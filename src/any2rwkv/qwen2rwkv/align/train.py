"""The only Qwen3.5-2B -> Qwen2RWKV alignment command."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def _teacher_mixer(source_text, layer_idx, hidden):
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


def _student_mixer(student, layer_idx, hidden):
    layer = student.model.layers[layer_idx]
    normalized = layer.input_layernorm(hidden)
    v_first = None if layer_idx == 0 else torch.zeros_like(normalized)
    return layer.tmix(
        normalized,
        v_first,
        None,
        torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
    )[0]


def _freeze_value_residual(tmix) -> None:
    tmix.value_residual_scale.data.zero_()
    for parameter in (tmix.v0, tmix.v1, tmix.v2, tmix.value_residual_scale):
        parameter.requires_grad_(False)


def _initialize_layer(source_text, student, layer_idx, hidden):
    source_layer = source_text.layers[layer_idx]
    target = student.model.layers[layer_idx].tmix
    normalized = source_layer.input_layernorm(hidden)
    if source_text.config.layer_types[layer_idx] == "linear_attention":
        metrics = initialize_gdn_layer(source_layer.linear_attn, target, normalized)
    else:
        _, embeddings = _position(source_text, normalized)
        metrics = initialize_gqa_layer(source_layer.self_attn, target, normalized, embeddings)
    _freeze_value_residual(target)
    return metrics


def _schedule(optimizer, steps: int):
    warmup = max(1, math.ceil(steps * 0.05))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _mean(value: torch.Tensor, world: int):
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return float(value)


def _gather_calibration(hidden: torch.Tensor, world: int) -> torch.Tensor:
    if world == 1:
        return hidden
    gathered = [torch.empty_like(hidden) for _ in range(world)]
    dist.all_gather(gathered, hidden.contiguous())
    return torch.cat(gathered)


@torch.no_grad()
def _evaluate_layer(source_text, student, layer_idx, hidden, world, device):
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    layer = student.model.layers[layer_idx]
    layer.train()
    for batch in hidden.split(8):
        batch = batch.to(device)
        wanted_mixer = _teacher_mixer(source_text, layer_idx, batch).float()
        actual_mixer = _student_mixer(student, layer_idx, batch).float()
        wanted_block = _teacher_layer(source_text, layer_idx, batch).float()
        v_first = None if layer_idx == 0 else torch.zeros_like(batch)
        actual_block = layer(
            batch,
            v_first,
            None,
            torch.ones(batch.shape[:2], dtype=torch.bool, device=device),
        )[0].float()
        for offset, (actual, wanted) in enumerate(
            ((actual_mixer, wanted_mixer), (actual_block, wanted_block))
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

    mixer_nmse, mixer_cosine = metrics(0)
    block_nmse, block_cosine = metrics(4)
    return {
        "mixer_nmse": mixer_nmse,
        "mixer_cosine": mixer_cosine,
        "block_nmse": block_nmse,
        "block_cosine": block_cosine,
        "rows_per_rank": int(hidden.shape[0]),
        "tokens_per_rank": int(hidden.shape[0] * hidden.shape[1]),
    }


def _completed_layers(output: Path) -> int:
    completed = 0
    while (output / f"layer_{completed:02d}.safetensors").is_file():
        completed += 1
    return completed


def _save_layer(output: Path, layer_idx: int, tmix) -> None:
    tensors = {name: value.detach().cpu().contiguous() for name, value in tmix.state_dict().items()}
    save_file(tensors, (output / f"layer_{layer_idx:02d}.safetensors").as_posix())


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
                value_first = None if index == 0 else torch.zeros_like(hidden, device=device)
                output, _ = layer(
                    hidden.to(device),
                    value_first,
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
        value_first = None if self.layer.layer_idx == 0 else torch.zeros_like(hidden)
        return self.layer(
            hidden,
            value_first,
            None,
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
        )[0]


def _layerwise(source_text, student, ids, output, rank, world, device, through_layer):
    cache = LastLayerCache(output / "cache", rank)
    completed = _completed_layers(output)
    for index in range(completed):
        student.model.layers[index].tmix.load_state_dict(
            load_file((output / f"layer_{index:02d}.safetensors").as_posix())
        )
    if not cache.path("current").is_file():
        _rebuild_cache(student, ids, cache, completed, device)

    if completed > through_layer:
        return cache
    for index in range(completed, through_layer + 1):
        hidden_cache = cache.load()
        if hidden_cache.shape[0] < 32:
            raise ValueError("each rank needs at least 32 packed rows for isolated data splits")
        calibration_local = hidden_cache[:8].to(device)
        development = hidden_cache[8:16]
        frozen_final = hidden_cache[16:24]
        training = hidden_cache[24:]
        calibration = _gather_calibration(calibration_local, world)
        metrics = (
            _initialize_layer(source_text, student, index, calibration) if rank == 0 else {}
        )
        if world > 1:
            for parameter in student.model.layers[index].tmix.parameters():
                dist.broadcast(parameter.data, 0)
        student.requires_grad_(False)
        tmix = student.model.layers[index].tmix
        tmix.requires_grad_(True)
        _freeze_value_residual(tmix)
        layer = student.model.layers[index].to(device).train()
        wrapper = _LayerObjective(layer)
        if world > 1:
            wrapper = DistributedDataParallel(wrapper, device_ids=[device.index])
        lr = 1e-5 if student.config.source_layer_types[index] == "linear_attention" else 3e-5
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
        loader = DataLoader(TensorDataset(training), batch_size=8, shuffle=False)
        max_epochs = 48
        scheduler = _schedule(optimizer, max_epochs * len(loader))
        zero_development = _evaluate_layer(
            source_text, student, index, development, world, device
        )
        if rank == 0:
            print(
                {
                    "layer": index,
                    "split": "development",
                    "stage": "zero_step",
                    **zero_development,
                    **metrics,
                },
                flush=True,
            )
        best_nmse = zero_development["block_nmse"]
        best_state = {
            name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
        }
        last_train_nmse = math.inf
        for epoch in range(max_epochs):
            total = torch.zeros((), device=device)
            for (hidden,) in loader:
                hidden = hidden.to(device)
                with torch.no_grad():
                    wanted = _teacher_layer(source_text, index, hidden)
                    wanted_mixer = _teacher_mixer(source_text, index, hidden)
                actual = wrapper(hidden)
                actual_mixer = _student_mixer(student, index, hidden)
                block_loss = (actual.float() - wanted.float()).square().mean() / (
                    wanted.float().square().mean() + 1e-6
                )
                mixer_loss = (actual_mixer.float() - wanted_mixer.float()).square().mean() / (
                    wanted_mixer.float().square().mean() + 1e-6
                )
                loss = block_loss + mixer_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite layer {index} loss")
                optimizer.zero_grad(set_to_none=True)
                for parameter in model_parameters:
                    parameter.grad = None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_parameters, 1.0)
                for parameter, master in zip(
                    model_parameters, master_parameters, strict=True
                ):
                    master.grad = (
                        None if parameter.grad is None else parameter.grad.detach().float()
                    )
                optimizer.step()
                with torch.no_grad():
                    for parameter, master in zip(
                        model_parameters, master_parameters, strict=True
                    ):
                        parameter.copy_(master.to(parameter.dtype))
                scheduler.step()
                tmix.value_residual_scale.data.zero_()
                total += block_loss.detach() / len(loader)
            average = _mean(total, world)
            last_train_nmse = average
            development_metrics = _evaluate_layer(
                source_text, student, index, development, world, device
            )
            if development_metrics["block_nmse"] < best_nmse:
                best_nmse = development_metrics["block_nmse"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in tmix.state_dict().items()
                }
            if rank == 0:
                print(
                    {
                        "layer": index,
                        "epoch": epoch,
                        "train_block_nmse": average,
                        "development_block_nmse": development_metrics["block_nmse"],
                        "development_mixer_nmse": development_metrics["mixer_nmse"],
                    },
                    flush=True,
                )
            if best_nmse <= 1e-3:
                break
        tmix.load_state_dict(best_state)
        development_metrics = _evaluate_layer(
            source_text, student, index, development, world, device
        )
        if development_metrics["block_nmse"] > 1e-3:
            raise RuntimeError(
                f"layer {index} development block NMSE "
                f"{development_metrics['block_nmse']:.8g} exceeds 1e-3 after {max_epochs} epochs"
            )
        final_metrics = _evaluate_layer(
            source_text, student, index, frozen_final, world, device
        )
        if rank == 0:
            print(
                {
                    "layer": index,
                    "split": "frozen_final",
                    "train_block_nmse": last_train_nmse,
                    **final_metrics,
                },
                flush=True,
            )
        if final_metrics["block_nmse"] > 1e-3:
            raise RuntimeError(
                f"layer {index} frozen-final block NMSE "
                f"{final_metrics['block_nmse']:.8g} exceeds 1e-3"
            )
        if rank == 0:
            _save_layer(output, index, tmix)
        if world > 1:
            dist.barrier()
        layer.train()
        chunks = []
        with torch.no_grad():
            for hidden in hidden_cache.split(8):
                chunks.append(
                    wrapper.module(hidden.to(device)).cpu()
                    if world > 1
                    else wrapper(hidden.to(device)).cpu()
                )
        cache.store(torch.cat(chunks), "next")
        cache.advance()
    return cache


def _set_global_parameters(student):
    student.requires_grad_(False)
    ordinary, residual, scales = [], [], []
    for index, layer in enumerate(student.model.layers):
        tmix = layer.tmix
        tmix.requires_grad_(True)
        value_parameters = (tmix.v0, tmix.v1, tmix.v2)
        if index:
            residual.extend(value_parameters)
        scales.append(tmix.value_residual_scale)
        if index == 0:
            for parameter in value_parameters:
                parameter.requires_grad_(False)
            tmix.value_residual_scale.requires_grad_(False)
            tmix.value_residual_scale.data.zero_()
        excluded = {id(p) for p in (*value_parameters, tmix.value_residual_scale)}
        ordinary.extend(p for p in tmix.parameters() if id(p) not in excluded)
    return ordinary, residual, [p for p in scales if p.requires_grad]


def _global_kl(
    source_outer, source_text, student, ids, tokenizer, output, rank, world, device, epochs=3
):
    ordinary, residual, scales = _set_global_parameters(student)
    optimizer = torch.optim.AdamW(
        [
            {"params": ordinary, "lr": 1e-6, "weight_decay": 0.1},
            {"params": residual, "lr": 1e-5, "weight_decay": 0.1},
            {"params": scales, "lr": 1e-5, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.99),
    )
    wrapped = student.model
    if world > 1:
        wrapped = DistributedDataParallel(wrapped, device_ids=[device.index])
    loader = DataLoader(TensorDataset(ids), batch_size=8, shuffle=False)
    scheduler = _schedule(optimizer, epochs * len(loader))
    history = []
    student.train()
    for epoch in range(epochs):
        total = torch.zeros((), device=device)
        for (input_ids,) in loader:
            input_ids = input_ids.to(device)
            mask = torch.ones_like(input_ids)
            with torch.no_grad():
                teacher_hidden = source_text(
                    input_ids=input_ids, attention_mask=mask, use_cache=False
                ).last_hidden_state
            student_hidden = wrapped(
                input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state
            optimizer.zero_grad(set_to_none=True)
            loss_value = torch.zeros((), device=device)
            hidden_gradient = torch.zeros_like(student_hidden)
            positions = input_ids.shape[0] * (input_ids.shape[1] - 1)
            for start in range(0, input_ids.shape[1] - 1, 16):
                stop = min(start + 16, input_ids.shape[1] - 1)
                with torch.no_grad():
                    teacher_logits = source_outer.lm_head(teacher_hidden[:, start:stop]).float()
                    teacher_log = F.log_softmax(teacher_logits, -1)
                    teacher_prob = teacher_log.exp()
                hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
                student_log = F.log_softmax(student.lm_head(hidden_leaf).float(), -1)
                part = (teacher_prob * (teacher_log - student_log)).sum() / positions
                if not torch.isfinite(part):
                    raise FloatingPointError("non-finite global KL")
                hidden_gradient[:, start:stop] = torch.autograd.grad(part, hidden_leaf)[0]
                loss_value += part.detach()
            student_hidden.backward(hidden_gradient)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for scale in scales:
                scale.data.clamp_(0, 1)
            student.model.layers[0].tmix.value_residual_scale.data.zero_()
            total += loss_value / len(loader)
        average = _mean(total, world)
        history.append(average)
        if rank == 0:
            print({"global_epoch": epoch, "kl": average}, flush=True)
        if epoch >= 2 and all(
            (history[j - 1] - history[j]) / max(abs(history[j - 1]), 1e-12) < 0.005
            for j in (len(history) - 2, len(history) - 1)
        ):
            break
    if rank == 0:
        student.eval().half()
        student.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        student.to(dtype=torch.bfloat16)


def _accept(output: Path) -> bool:
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(output)
    model = Qwen2RWKVForCausalLM.from_pretrained(output, dtype=torch.float16).cuda().eval()
    records = []
    passed = True
    for prompt in PROMPTS:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"].cuda()
        with torch.no_grad():
            generated = model.generate(
                token_ids, use_cache=True, do_sample=False, max_new_tokens=128
            )
        answer_ids = generated[0, token_ids.shape[1] :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        tokens = answer_ids.tolist()
        coherent = bool(answer.strip()) and "�" not in answer
        if tokens:
            coherent = (
                coherent and max(tokens.count(token) for token in set(tokens)) < len(tokens) * 0.8
            )
        passed = passed and coherent
        records.append({"prompt": prompt, "token_ids": token_ids[0].tolist(), "output": answer})
    scales = [float(layer.tmix.value_residual_scale) for layer in model.model.layers]
    result = {"passed": passed, "generations": records, "value_residual_scale": scales}
    (output / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return passed


def _fresh_acceptance(output: Path) -> bool:
    command = [
        sys.executable,
        "-m",
        "any2rwkv.qwen2rwkv.align.train",
        "--accept-only",
        "--output",
        output.as_posix(),
    ]
    return subprocess.run(command, check=False).returncode == 0


def convert_qwen3_5_2b(
    source: str,
    output: str,
    agentic: str,
    math_dataset: str,
    through_layer: int = 23,
):
    rank, world, device = _distributed()
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
    if not 0 <= through_layer < student.config.num_hidden_layers:
        raise ValueError(
            f"through_layer must be in [0, {student.config.num_hidden_layers - 1}]"
        )
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
    if through_layer < student.config.num_hidden_layers - 1:
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    _global_kl(
        source_outer, source_text, student, local_ids, tokenizer, output_path, rank, world, device
    )
    if world > 1:
        dist.barrier()
    failed = torch.zeros((), dtype=torch.int32, device=device)
    if rank == 0 and not _fresh_acceptance(output_path):
        failed.fill_(1)
    if world > 1:
        dist.broadcast(failed, 0)
    if failed.item():
        _global_kl(
            source_outer,
            source_text,
            student,
            local_ids,
            tokenizer,
            output_path,
            rank,
            world,
            device,
            epochs=1,
        )
        if world > 1:
            dist.barrier()
        failed.zero_()
        if rank == 0 and not _fresh_acceptance(output_path):
            failed.fill_(1)
        if world > 1:
            dist.broadcast(failed, 0)
        if failed.item():
            raise RuntimeError("migration loop failed the only allowed generation acceptance")
    if world > 1:
        dist.destroy_process_group()


def continue_global_kl(source: str, output: str) -> None:
    """Run the single permitted corrective KL epoch from the saved final model."""
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVForCausalLM

    rank, world, device = _distributed()
    output_path = Path(output).resolve()
    packed_path = output_path / "packed_sequences.pt"
    if not packed_path.is_file():
        raise FileNotFoundError(f"missing packed sequences: {packed_path}")
    ids = torch.load(packed_path, map_location="cpu", weights_only=True)
    source_outer, source_text = load_qwen_teacher(source, torch.bfloat16, device)
    student = Qwen2RWKVForCausalLM.from_pretrained(
        output_path, dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(source)
    _global_kl(
        source_outer,
        source_text,
        student,
        ids[rank::world].contiguous(),
        tokenizer,
        output_path,
        rank,
        world,
        device,
        epochs=1,
    )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--agentic", default="nvidia/Nemotron-SFT-Agentic-v2")
    parser.add_argument("--math", dest="math_dataset", default="nvidia/Nemotron-SFT-Math-v4")
    parser.add_argument("--accept-only", action="store_true")
    parser.add_argument("--global-kl-only", action="store_true")
    parser.add_argument("--through-layer", type=int, default=23)
    args = parser.parse_args()
    if args.accept_only:
        raise SystemExit(0 if _accept(Path(args.output)) else 1)
    if args.source is None:
        parser.error("--source is required for conversion")
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
    if args.global_kl_only:
        continue_global_kl(args.source, args.output)
        return
    convert_qwen3_5_2b(
        args.source,
        args.output,
        args.agentic,
        args.math_dataset,
        args.through_layer,
    )


if __name__ == "__main__":
    main()
