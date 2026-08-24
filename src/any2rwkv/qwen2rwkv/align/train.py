"""The only Qwen3.5-2B -> Qwen2RWKV alignment command."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from ..gdn2rwkv import initialize_gdn_layer
from ..gqa2rwkv import initialize_gqa_layer
from .datasets import PackedSequences, build_packed_sequences
from .last_layer_cache import LastLayerCache
from .model_qwen import load_qwen_teacher
from .model_qwen2rwkv import build_qwen2rwkv

PROMPTS = (
    "请用三句话解释为什么天空是蓝色的。",
    "Solve x^2 - 5x + 6 = 0 and explain briefly.",
    "Write a Python function that returns the Fibonacci sequence up to n.",
)
LAYER_OUTPUT_NMSE_TARGET = 3e-3
VALIDATION_TMIX_PATIENCE = 5


def _distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def _position(source_text, hidden):
    batch, length = hidden.shape[:2]
    ids = torch.arange(length, device=hidden.device).view(1, -1).expand(batch, -1)
    return ids, source_text.rotary_emb(hidden, ids)


def _causal(hidden):
    length = hidden.shape[1]
    mask = torch.full(
        (length, length), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device
    ).triu(1)
    return mask.view(1, 1, length, length)


def _teacher_layer(source_text, layer_idx, hidden):
    layer = source_text.layers[layer_idx]
    positions, embeddings = _position(source_text, hidden)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if source_text.config.layer_types[layer_idx] == "full_attention":
        mask = _causal(hidden)
    return layer(
        hidden,
        position_embeddings=embeddings,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
    )


def _teacher_tmix_output(source_text, layer_idx, hidden):
    layer = source_text.layers[layer_idx]
    normalized = layer.input_layernorm(hidden)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if source_text.config.layer_types[layer_idx] == "linear_attention":
        return layer.linear_attn(normalized, attention_mask=mask)
    _, embeddings = _position(source_text, normalized)
    return layer.self_attn(
        normalized,
        position_embeddings=embeddings,
        attention_mask=_causal(normalized),
        past_key_values=None,
    )[0]


def _student_tmix_output(student, layer_idx, hidden):
    layer = student.model.layers[layer_idx]
    normalized = layer.input_layernorm(hidden)
    return layer.tmix(
        normalized,
        None,
        torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
    )


def _require_finite_tmix(tmix, layer_idx: int, stage: str, *, gradients: bool = False) -> None:
    for name, parameter in tmix.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None and not torch.isfinite(value).all():
            kind = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite layer {layer_idx} {kind} {name} during {stage}")


def _initialize_layer(source_text, student, layer_idx, init_hidden, validation_hidden):
    source_layer = source_text.layers[layer_idx]
    target = student.model.layers[layer_idx].tmix
    init_normalized = source_layer.input_layernorm(init_hidden)
    validation_normalized = source_layer.input_layernorm(validation_hidden)
    if source_text.config.layer_types[layer_idx] != "linear_attention":
        return initialize_gqa_layer()
    return initialize_gdn_layer(
        source_layer.linear_attn, target, init_normalized, validation_normalized
    )


def _schedule(
    optimizer,
    steps: int,
    *,
    warmup_fraction: float = 0.05,
    min_scale: float = 0.1,
):
    warmup = math.ceil(steps * warmup_fraction)

    def scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return min_scale + (1 - min_scale) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _mean(value: torch.Tensor, world: int):
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return float(value)


def _gather_init_hidden(hidden: torch.Tensor, world: int) -> torch.Tensor:
    if world == 1:
        return hidden
    gathered = [torch.empty_like(hidden) for _ in range(world)]
    dist.all_gather(gathered, hidden.contiguous())
    return torch.stack(gathered, dim=1).reshape(-1, *hidden.shape[1:])


def _validation_is_better(candidate: dict[str, float], best: dict[str, float]) -> bool:
    return candidate["layer_output_nmse"] < best["layer_output_nmse"]


def _require_finite_metrics(metrics: dict[str, float], layer_idx: int, split: str) -> None:
    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite layer {layer_idx} {split} metric {name}: {value}")


@torch.no_grad()
def _evaluate_layer(source_text, student, layer_idx, hidden, world, device):
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    layer = student.model.layers[layer_idx]
    layer.train()
    for batch in hidden.split(8):
        batch = batch.to(device)
        wanted_tmix = _teacher_tmix_output(source_text, layer_idx, batch).float()
        actual_tmix = _student_tmix_output(student, layer_idx, batch).float()
        source_layer = source_text.layers[layer_idx]
        wanted_residual = batch + wanted_tmix.to(batch)
        wanted_layer = (
            wanted_residual
            + source_layer.mlp(source_layer.post_attention_layernorm(wanted_residual))
        ).float()
        actual_residual = batch + actual_tmix.to(batch)
        actual_layer = (
            actual_residual + layer.mlp(layer.post_attention_layernorm(actual_residual))
        ).float()
        for offset, (actual, wanted) in enumerate(
            ((actual_tmix, wanted_tmix), (actual_layer, wanted_layer))
        ):
            totals[offset * 4] += (actual - wanted).double().square().sum()
            totals[offset * 4 + 1] += wanted.double().square().sum()
            totals[offset * 4 + 2] += (actual.double() * wanted.double()).sum()
            totals[offset * 4 + 3] += actual.double().square().sum()
    if world > 1:
        dist.all_reduce(totals)

    def metrics(offset):
        error, wanted_sq, dot, actual_sq = totals[offset : offset + 4]
        return float(error / wanted_sq.clamp_min(1e-24)), float(
            dot / (wanted_sq * actual_sq).clamp_min(1e-48).sqrt()
        )

    tmix_output_nmse, tmix_output_cosine = metrics(0)
    layer_output_nmse, layer_output_cosine = metrics(4)
    return {
        "tmix_output_nmse": tmix_output_nmse,
        "tmix_output_cosine": tmix_output_cosine,
        "layer_output_nmse": layer_output_nmse,
        "layer_output_cosine": layer_output_cosine,
        "rows_per_rank": int(hidden.shape[0]),
        "tokens_per_rank": int(hidden.shape[0] * hidden.shape[1]),
        "valid_tokens": int(hidden.shape[0] * hidden.shape[1] * world),
    }


def _completed_layers(output: Path) -> int:
    completed = 0
    while (output / f"layer_{completed:02d}.safetensors").is_file():
        completed += 1
    return completed


def _save_layer(output: Path, layer_idx: int, tmix) -> None:
    tensors = {name: value.detach().cpu().contiguous() for name, value in tmix.state_dict().items()}
    save_file(tensors, (output / f"layer_{layer_idx:02d}.safetensors").as_posix())


def _load_layer_checkpoint(output: Path, layer_idx: int, tmix) -> None:
    path = output / f"layer_{layer_idx:02d}.safetensors"
    state = load_file(path.as_posix())
    expected = set(tmix.state_dict())
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"layer {layer_idx} checkpoint schema is incompatible with the current "
            "source-shell TMix/readout runtime; use a new output directory "
            f"(missing={missing}, unexpected={unexpected})"
        )
    tmix.load_state_dict(state, strict=True)


def _rebuild_cache(student, ids, cache: LastLayerCache, completed: int, device):
    chunks = []
    with torch.no_grad():
        for batch in ids.split(8):
            chunks.append(student.model.embed_tokens(batch.to(device)).cpu())
    cache.store(torch.cat(chunks), "next")
    cache.advance()
    for index in range(completed):
        layer = student.model.layers[index].to(device).train()
        chunks = []
        with torch.no_grad():
            for hidden in cache.load().split(8):
                output = layer(
                    hidden.to(device),
                    None,
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device),
                )
                chunks.append(output.cpu())
        cache.store(torch.cat(chunks), "next")
        cache.advance()


class _LayerObjective(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        normalized = self.layer.input_layernorm(hidden)
        tmix_output = self.layer.tmix(
            normalized,
            None,
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
        )
        residual = hidden + tmix_output
        layer_output = residual + self.layer.mlp(self.layer.post_attention_layernorm(residual))
        return layer_output, tmix_output, tmix_output.float().new_zeros(())


def _layerwise(
    source_text,
    student,
    ids,
    output,
    rank,
    world,
    device,
    through_layer,
):
    cache = LastLayerCache(output / "cache", rank)
    completed = _completed_layers(output)
    for index in range(completed):
        _load_layer_checkpoint(output, index, student.model.layers[index].tmix)
    if completed > through_layer:
        return cache
    # The contiguous layer checkpoints are the sole resume authority. Rebuilding
    # avoids pairing a newly saved layer with a stale cache after an interrupted
    # save/cache-advance window.
    _rebuild_cache(student, ids, cache, completed, device)
    prefix_strict_pass = True
    strict_failures: list[str] = []
    for index in range(completed, through_layer + 1):
        hidden_cache = cache.load()
        if hidden_cache.shape[0] < 32:
            raise ValueError("each rank needs at least 32 packed rows for isolated data splits")
        init_local = hidden_cache[:8].to(device)
        validation_hidden = hidden_cache[8:24]
        validation_local = validation_hidden.to(device)
        train_hidden = hidden_cache[24:]
        init_hidden = _gather_init_hidden(init_local, world)
        validation_for_init = _gather_init_hidden(validation_local, world)
        metrics = (
            _initialize_layer(source_text, student, index, init_hidden, validation_for_init)
            if rank == 0
            else {}
        )
        if rank == 0:
            _require_finite_tmix(
                student.model.layers[index].tmix, index, "before-distillation initialization"
            )
        if world > 1:
            for parameter in student.model.layers[index].tmix.parameters():
                dist.broadcast(parameter.data, 0)
        student.requires_grad_(False)
        tmix = student.model.layers[index].tmix
        tmix.requires_grad_(True)
        layer = student.model.layers[index].to(device).train()
        wrapper = _LayerObjective(layer)
        if world > 1:
            wrapper = DistributedDataParallel(wrapper, device_ids=[device.index])
        lr = 1e-5
        corrective_lr = 1e-6
        model_parameters = [p for p in wrapper.parameters() if p.requires_grad]
        master_parameters = [
            nn.Parameter(parameter.detach().float().clone()) for parameter in model_parameters
        ]
        optimizer = torch.optim.AdamW(
            master_parameters,
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.1,
        )
        training_dataset = TensorDataset(train_hidden)
        loader = DataLoader(training_dataset, batch_size=8, shuffle=False)
        initial_epochs = 48
        corrective_epochs = 96
        max_epochs = initial_epochs + corrective_epochs
        scheduler = _schedule(optimizer, initial_epochs * len(loader))
        before_distillation_validation = _evaluate_layer(
            source_text, student, index, validation_hidden, world, device
        )
        _require_finite_metrics(
            before_distillation_validation, index, "before-distillation validation"
        )
        if rank == 0:
            print(
                {
                    "layer": index,
                    "split": "validation",
                    "stage": "before_distillation",
                    **before_distillation_validation,
                    **metrics,
                },
                flush=True,
            )
        best_validation = dict(before_distillation_validation)
        best_state = {
            name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
        }
        best_epoch = -1
        epochs_without_validation_tmix_improvement = 0
        for epoch in range(max_epochs):
            corrective = epoch >= initial_epochs
            if epoch == initial_epochs:
                if best_validation["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET:
                    break
                tmix.load_state_dict(best_state)
                _require_finite_tmix(tmix, index, "corrective alignment restore")
                master_parameters = [
                    nn.Parameter(parameter.detach().float().clone())
                    for parameter in model_parameters
                ]
                optimizer = torch.optim.AdamW(
                    master_parameters,
                    lr=corrective_lr,
                    betas=(0.9, 0.99),
                    weight_decay=0.0,
                )
                scheduler = _schedule(
                    optimizer,
                    corrective_epochs * len(loader),
                    warmup_fraction=0.0,
                )
                epochs_without_validation_tmix_improvement = 0
            layer_output_total = torch.zeros((), device=device)
            tmix_output_total = torch.zeros((), device=device)
            for training_batch in loader:
                hidden = training_batch[0].to(device)
                with torch.no_grad():
                    wanted_layer = _teacher_layer(source_text, index, hidden)
                    wanted_tmix = _teacher_tmix_output(source_text, index, hidden)
                actual_layer, actual_tmix, _ = wrapper(hidden)
                layer_output_loss = (
                    actual_layer.float() - wanted_layer.float()
                ).square().mean() / (wanted_layer.float().square().mean() + 1e-6)
                tmix_output_loss = (actual_tmix.float() - wanted_tmix.float()).square().mean() / (
                    wanted_tmix.float().square().mean() + 1e-6
                )
                layer_loss_weight = 4.0 if corrective else 1.0
                loss = layer_loss_weight * layer_output_loss + tmix_output_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite layer {index} loss")
                optimizer.zero_grad(set_to_none=True)
                for parameter in model_parameters:
                    parameter.grad = None
                loss.backward()
                _require_finite_tmix(tmix, index, f"epoch {epoch} backward", gradients=True)
                torch.nn.utils.clip_grad_norm_(model_parameters, 1.0)
                for parameter, master in zip(model_parameters, master_parameters, strict=True):
                    master.grad = (
                        None if parameter.grad is None else parameter.grad.detach().float()
                    )
                optimizer.step()
                with torch.no_grad():
                    for parameter, master in zip(model_parameters, master_parameters, strict=True):
                        parameter.copy_(master.to(parameter.dtype))
                _require_finite_tmix(tmix, index, f"epoch {epoch} update")
                scheduler.step()
                layer_output_total += layer_output_loss.detach() / len(loader)
                tmix_output_total += tmix_output_loss.detach() / len(loader)
            train_layer_output_nmse = _mean(layer_output_total, world)
            train_tmix_output_nmse = _mean(tmix_output_total, world)
            if not math.isfinite(train_layer_output_nmse) or not math.isfinite(
                train_tmix_output_nmse
            ):
                raise FloatingPointError(f"non-finite layer {index} epoch {epoch} train metrics")
            validation_metrics = _evaluate_layer(
                source_text, student, index, validation_hidden, world, device
            )
            _require_finite_metrics(validation_metrics, index, f"epoch {epoch} validation")
            previous_best_passed = best_validation["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET
            previous_best_tmix = best_validation["tmix_output_nmse"]
            improved = _validation_is_better(validation_metrics, best_validation)
            if improved:
                best_validation = dict(validation_metrics)
                best_state = {
                    name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
                }
                best_epoch = epoch
                if validation_metrics["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET:
                    if (
                        not previous_best_passed
                        or validation_metrics["tmix_output_nmse"] < previous_best_tmix
                    ):
                        epochs_without_validation_tmix_improvement = 0
                    else:
                        epochs_without_validation_tmix_improvement += 1
            elif best_validation["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET:
                epochs_without_validation_tmix_improvement += 1
            if rank == 0:
                print(
                    {
                        "layer": index,
                        "epoch": epoch,
                        "alignment_phase": "corrective" if corrective else "initial",
                        "phase_epoch": epoch - initial_epochs if corrective else epoch,
                        "train_tmix_output_nmse": train_tmix_output_nmse,
                        "train_layer_output_nmse": train_layer_output_nmse,
                        "validation_tmix_output_nmse": validation_metrics["tmix_output_nmse"],
                        "validation_layer_output_nmse": validation_metrics["layer_output_nmse"],
                        "best_validation_tmix_output_nmse": best_validation["tmix_output_nmse"],
                        "best_validation_layer_output_nmse": best_validation["layer_output_nmse"],
                        "epochs_without_validation_tmix_improvement": (
                            epochs_without_validation_tmix_improvement
                        ),
                    },
                    flush=True,
                )
            if best_validation[
                "layer_output_nmse"
            ] <= LAYER_OUTPUT_NMSE_TARGET and epochs_without_validation_tmix_improvement >= (
                12 if corrective else VALIDATION_TMIX_PATIENCE
            ):
                break
        tmix.load_state_dict(best_state)
        _require_finite_tmix(tmix, index, "best validation checkpoint")
        train_metrics = _evaluate_layer(source_text, student, index, train_hidden, world, device)
        _require_finite_metrics(train_metrics, index, "best checkpoint on train")
        validation_metrics = _evaluate_layer(
            source_text, student, index, validation_hidden, world, device
        )
        _require_finite_metrics(validation_metrics, index, "best checkpoint on validation")
        validation_pass = validation_metrics["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET
        layer_strict_pass = validation_pass
        prefix_strict_pass = prefix_strict_pass and layer_strict_pass
        if not layer_strict_pass:
            strict_failures.append(
                f"layer {index} validation={validation_metrics['layer_output_nmse']:.8g}"
            )
        if rank == 0:
            print(
                {
                    "layer": index,
                    "stage": "best_checkpoint_evaluation",
                    **{
                        f"before_distillation_validation_{name}": value
                        for name, value in before_distillation_validation.items()
                    },
                    **{f"train_{name}": value for name, value in train_metrics.items()},
                    **{f"validation_{name}": value for name, value in validation_metrics.items()},
                    "validation_pass": validation_pass,
                    "layer_strict_pass": layer_strict_pass,
                    "prefix_strict_pass": prefix_strict_pass,
                    "best_epoch": best_epoch,
                    "best_alignment_phase": (
                        "before_distillation"
                        if best_epoch < 0
                        else ("corrective" if best_epoch >= initial_epochs else "initial")
                    ),
                    "best_phase_epoch": (
                        -1
                        if best_epoch < 0
                        else best_epoch - initial_epochs
                        if best_epoch >= initial_epochs
                        else best_epoch
                    ),
                    "initialization_metrics": metrics,
                },
                flush=True,
            )
        if rank == 0 and prefix_strict_pass:
            _save_layer(output, index, tmix)
        if world > 1:
            dist.barrier()
        layer.train()
        chunks = []
        objective = wrapper.module if world > 1 else wrapper
        with torch.no_grad():
            for hidden in hidden_cache.split(8):
                chunks.append(objective(hidden.to(device))[0].cpu())
        cache.store(torch.cat(chunks), "next")
        cache.advance()
    if strict_failures:
        raise RuntimeError(
            f"strict layer-output NMSE target {LAYER_OUTPUT_NMSE_TARGET} failed after "
            f"measuring through layer {through_layer}: {'; '.join(strict_failures)}"
        )
    return cache


def convert_qwen3_5_2b(
    source: str,
    output: str,
    agentic: str,
    math_dataset: str,
    through_layer: int = 2,
) -> None:
    """Align only the validated three-layer GDN prefix.

    Layer 3 is the first GQA layer. It remains fail closed until a new method
    passes the strict feasibility and product gates documented in
    ``docs/gqa2rwkv.md``.
    """

    if not 0 <= through_layer <= 2:
        initialize_gqa_layer()
    rank, world, device = _distributed()
    try:
        output_path = Path(output).resolve()
        if output_path == Path(source).resolve():
            raise ValueError("output must not overwrite the source checkpoint")
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(source)
        packed_path = output_path / "packed_sequences.pt"
        packed_objects = [
            (
                torch.load(packed_path, map_location="cpu", weights_only=True)
                if packed_path.exists()
                else build_packed_sequences(tokenizer, agentic, math_dataset).input_ids
            )
            if rank == 0
            else None
        ]
        if rank == 0 and not packed_path.exists():
            torch.save(packed_objects[0], packed_path)
        if world > 1:
            dist.broadcast_object_list(packed_objects, src=0, device=device)
        packed = PackedSequences(packed_objects[0])
        source_outer, source_text = load_qwen_teacher(source, torch.bfloat16, device)
        student = build_qwen2rwkv(source_outer, source_text).to(device=device, dtype=torch.bfloat16)
        local_ids = packed.input_ids[rank::world].contiguous()
        _layerwise(
            source_text,
            student,
            local_ids,
            output_path,
            rank,
            world,
            device,
            through_layer,
        )
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--agentic", default="nvidia/Nemotron-SFT-Agentic-v2")
    parser.add_argument("--math", dest="math_dataset", default="nvidia/Nemotron-SFT-Math-v4")
    parser.add_argument("--through-layer", type=int, default=2)
    args = parser.parse_args()
    if "WORLD_SIZE" not in os.environ:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=8",
            "-m",
            "any2rwkv.qwen2rwkv.align.train",
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.run(command, check=False).returncode)
    convert_qwen3_5_2b(
        args.source,
        args.output,
        args.agentic,
        args.math_dataset,
        args.through_layer,
    )


if __name__ == "__main__":
    main()
