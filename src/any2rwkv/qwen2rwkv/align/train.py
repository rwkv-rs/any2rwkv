"""The only Qwen3.5-2B -> Qwen2RWKV alignment command."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
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
from ..gqa2rwkv import (
    evaluate_gqa_recall,
    initialize_gqa_layer,
)
from .datasets import PackedSequences, build_packed_sequences
from .last_layer_cache import LastLayerCache
from .model_qwen import load_qwen_teacher
from .model_qwen2rwkv import build_qwen2rwkv

PROMPTS = (
    "请用三句话解释为什么天空是蓝色的。",
    "Solve x^2 - 5x + 6 = 0 and explain briefly.",
    "Write a Python function that returns the Fibonacci sequence up to n.",
)
VALIDATION_TMIX_PATIENCE = 5
GQA_PACKED_SHA256 = "1d039b73dcafd9783a7e872f682cf64728cb31f6090ef54c2882ca3bc0919336"
# Retention fractions have a much larger effect per unit than projection weights.
GQA_DECAY_LR_SCALE = 0.02


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


def _student_tmix_output(student, layer_idx, hidden, v_first=None):
    layer = student.model.layers[layer_idx]
    normalized = layer.input_layernorm(hidden)
    return layer.tmix(
        normalized,
        v_first,
        None,
        torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
    )[0]


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
    if source_text.config.layer_types[layer_idx] == "linear_attention":
        metrics = initialize_gdn_layer(
            source_layer.linear_attn, target, init_normalized, validation_normalized
        )
    else:
        _, init_embeddings = _position(source_text, init_normalized)
        metrics = initialize_gqa_layer(
            source_layer.self_attn,
            target,
            init_normalized,
            init_embeddings,
        )
    return metrics


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


def _hidden_batches(hidden, v_first=None):
    for start in range(0, len(hidden), 8):
        yield hidden[start : start + 8], None if v_first is None else v_first[start : start + 8]


@torch.no_grad()
def _evaluate_layer(source_text, student, layer_idx, hidden, world, device, v_first=None):
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    layer = student.model.layers[layer_idx]
    layer.train()
    for batch, first in _hidden_batches(hidden, v_first):
        batch = batch.to(device)
        wanted_tmix = _teacher_tmix_output(source_text, layer_idx, batch).float()
        actual_tmix = _student_tmix_output(
            student, layer_idx, batch, None if first is None else first.to(device)
        ).float()
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


def _require_gqa_packed_provenance(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != GQA_PACKED_SHA256:
        raise ValueError(
            "GQA prefix-cache mode requires the immutable packed tensor "
            f"SHA-256 {GQA_PACKED_SHA256}; got {actual} for {path}"
        )


def _save_layer(output: Path, layer_idx: int, tmix) -> None:
    _assert_pure_rwkv_state(tmix)
    tensors = {name: value.detach().cpu().contiguous() for name, value in tmix.state_dict().items()}
    path = output / f"layer_{layer_idx:02d}.safetensors"
    pending = path.with_suffix(".pending")
    save_file(tensors, pending.as_posix())
    pending.replace(path)


def _assert_pure_rwkv_state(module) -> None:
    forbidden = ("teacher", "sidecar", "beta", "branch_gate", "lora")
    names = tuple(module.state_dict())
    leaked = [name for name in names if any(token in name for token in forbidden)]
    if leaked:
        raise RuntimeError(f"pure RWKV checkpoint contains temporary parameters: {leaked}")


def _gqa_transferred_names(tmix):
    prefixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "feature_q_weight",
        "feature_k_weight",
    )
    return {name for name in tmix.state_dict() if name.split(".")[0] in prefixes}


def _load_gqa_initial_checkpoint(tmix, path: Path) -> None:
    """Explicit training-time expansion of a v2 layer; runtime loading stays strict."""
    saved = load_file(path.as_posix())
    state = tmix.state_dict()
    if set(saved) not in (set(state), _gqa_transferred_names(tmix)):
        raise ValueError("GQA initialization must be a complete v2 or v3 layer checkpoint")
    state.update(saved)
    tmix.load_state_dict(state, strict=True)


def _warm_start_checkpoint(path, layer_idx: int) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / f"layer_{layer_idx:02d}.safetensors"
    elif layer_idx != 3:
        return None
    if not candidate.is_file():
        raise FileNotFoundError(f"missing warm-start layer checkpoint: {candidate}")
    return candidate


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


@torch.no_grad()
def _cache_layer(layer, hidden, v_first, cache, device):
    outputs, first_values = [], []
    for batch, first in _hidden_batches(hidden, v_first):
        output, first = layer(batch.to(device), None if first is None else first.to(device))
        outputs.append(output.cpu())
        first_values.append(first.cpu())
    cache.store(torch.cat(outputs), v_first=torch.cat(first_values))
    cache.advance()


def _rebuild_cache(student, ids, cache: LastLayerCache, completed: int, device):
    with torch.no_grad():
        hidden = torch.cat(
            [student.model.embed_tokens(batch.to(device)).cpu() for batch in ids.split(8)]
        )
    cache.store(hidden)
    cache.advance()
    for index in range(completed):
        layer = student.model.layers[index].to(device).train()
        _cache_layer(layer, cache.load(), cache.load_v_first(), cache, device)


class _LayerObjective(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, teacher_tmix=None, gate=0.0, v_first=None):
        normalized = self.layer.input_layernorm(hidden)
        tmix_output, _ = self.layer.tmix(
            normalized,
            v_first,
            None,
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
        )
        mixed = (
            tmix_output if teacher_tmix is None else _mix_teacher(tmix_output, teacher_tmix, gate)
        )
        residual = hidden + mixed
        layer_output = residual + self.layer.mlp(self.layer.post_attention_layernorm(residual))
        return layer_output, tmix_output


@torch.no_grad()
def _evaluate_gqa_reference(source_text, student, layer_idx, hidden, world, device, v_first=None):
    target_layer = student.model.layers[layer_idx]
    reference_layer = copy.deepcopy(target_layer).to(device=device, dtype=torch.float32).eval()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for batch, first in _hidden_batches(hidden, v_first):
        batch = batch.to(device)
        wanted_tmix = _teacher_tmix_output(source_text, layer_idx, batch)
        source_layer = source_text.layers[layer_idx]
        wanted_residual = batch + wanted_tmix
        wanted_block = wanted_residual + source_layer.mlp(
            source_layer.post_attention_layernorm(wanted_residual)
        )
        reference_batch = batch.float()
        reference_tmix = reference_layer.tmix.reference_forward(
            reference_layer.input_layernorm(reference_batch),
            v_first=None if first is None else first.to(device=device, dtype=torch.float32),
        )
        reference_residual = reference_batch + reference_tmix
        reference_block = reference_residual + reference_layer.mlp(
            reference_layer.post_attention_layernorm(reference_residual)
        )
        native_tmix = _student_tmix_output(
            student, layer_idx, batch, None if first is None else first.to(device)
        )
        native_residual = batch + native_tmix
        native_block = native_residual + target_layer.mlp(
            target_layer.post_attention_layernorm(native_residual)
        )
        totals[0] += (reference_block - wanted_block.float()).double().square().sum()
        totals[1] += wanted_block.double().square().sum()
        totals[2] += (native_block.float() - reference_block).double().square().sum()
        totals[3] += reference_block.double().square().sum()
        totals[4] += (native_tmix.float() - reference_tmix).double().square().sum()
        totals[5] += reference_tmix.double().square().sum()
    if world > 1:
        dist.all_reduce(totals)
    return {
        "reference_layer_output_nmse": float(totals[0] / totals[1].clamp_min(1e-24)),
        "native_incremental_layer_output_nmse": float(totals[2] / totals[3].clamp_min(1e-24)),
        "native_incremental_tmix_output_nmse": float(totals[4] / totals[5].clamp_min(1e-24)),
    }


@torch.no_grad()
def _fp16_forward_mode(layer, hidden, config, chunk_size, cache=None, v_first=None):
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVCache

    cache = Qwen2RWKVCache(config) if cache is None else cache
    length = hidden.shape[1]
    chunk_size = length if chunk_size is None else chunk_size
    tmix_chunks = []
    block_chunks = []
    for start in range(0, length, chunk_size):
        chunk = hidden[:, start : start + chunk_size]
        normalized = layer.input_layernorm(chunk)
        tmix_output, _ = layer.tmix(
            normalized,
            None if v_first is None else v_first[:, start : start + chunk_size],
            cache,
            torch.ones(chunk.shape[:2], dtype=torch.bool, device=chunk.device),
        )
        residual = chunk + tmix_output
        block_output = residual + layer.mlp(layer.post_attention_layernorm(residual))
        cache.mark_updated(layer.layer_idx, chunk.shape[1])
        tmix_chunks.append(tmix_output)
        block_chunks.append(block_output)
    return (
        torch.cat(tmix_chunks, dim=1),
        torch.cat(block_chunks, dim=1),
        cache,
    )


def _fresh_process_gqa_strict_load(config, layer_idx: int, tmix) -> None:
    with tempfile.TemporaryDirectory(prefix="any2rwkv-gqa-strict-") as directory:
        root = Path(directory)
        config_path = root / "config.json"
        state_path = root / "layer.safetensors"
        config.to_json_file(config_path)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tmix.state_dict().items()},
            state_path.as_posix(),
        )
        script = "\n".join(
            (
                "import sys",
                "from safetensors.torch import load_file",
                "from any2rwkv.qwen2rwkv.transformers.modeling_qwen2rwkv import (",
                "    Qwen2RWKVConfig, Qwen2RWKVTimeMix)",
                "config = Qwen2RWKVConfig.from_json_file(sys.argv[1])",
                "module = Qwen2RWKVTimeMix(config, int(sys.argv[3])).half().eval()",
                "state = load_file(sys.argv[2])",
                "module.load_state_dict(state, strict=True)",
                "for name in state:",
                "    forbidden = ('lora', 'sidecar', 'teacher', 'beta', 'branch_gate')",
                "    assert not any(x in name for x in forbidden)",
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                config_path.as_posix(),
                state_path.as_posix(),
                str(layer_idx),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(
                "fresh-process strict pure RWKV layer load failed: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )


def _gqa_cache_resource_metrics(cache, layer_idx: int) -> dict[str, float]:
    layer = cache.layers[layer_idx]
    recurrent = layer.recurrent_states[0]
    batch = recurrent.shape[0]
    recurrent_bytes = recurrent.numel() * recurrent.element_size() // batch
    control_bytes = (
        cache.elapsed[layer_idx].numel() * cache.elapsed[layer_idx].element_size() // batch
    )
    if recurrent_bytes != 2 * 1024 * 1024:
        raise RuntimeError(f"pure RWKV persistent state changed: recurrent={recurrent_bytes}")
    shift = layer.conv_states[0]
    shift_bytes = shift.numel() * shift.element_size() // batch
    return {
        "token_shift_bytes_per_sequence": float(shift_bytes),
        "recurrent_bytes_per_sequence": float(recurrent_bytes),
        "control_bytes_per_sequence": float(control_bytes),
        "total_fixed_state_bytes_per_sequence": float(
            recurrent_bytes + shift_bytes + control_bytes
        ),
    }


@torch.no_grad()
def _gqa_fp16_soak(layer, seed_hidden, config, v_first=None) -> dict[str, float]:
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVCache

    cache = Qwen2RWKVCache(config)
    completed = 0
    resource_metrics = {}
    while completed < 8192:
        length = min(256, 8192 - completed)
        hidden = seed_hidden.expand(1, length, -1).contiguous()
        output, _ = layer.tmix(
            layer.input_layernorm(hidden),
            None if v_first is None else v_first.expand(1, length, -1).contiguous(),
            cache,
            torch.ones(1, length, dtype=torch.bool, device=hidden.device),
        )
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite FP16 output during 8192-token pure RWKV soak")
        cache.mark_updated(layer.layer_idx, length)
        if not torch.isfinite(cache.layers[layer.layer_idx].recurrent_states[0]).all():
            raise FloatingPointError("non-finite FP16 recurrent state during 8192-token soak")
        completed += length
        if completed in (512, 4096, 8192):
            resource_metrics = _gqa_cache_resource_metrics(cache, layer.layer_idx)
            if int(cache.elapsed[layer.layer_idx][0]) != completed:
                raise RuntimeError("FlashRWKV2 elapsed state diverged during pure RWKV soak")
    return {"soak_tokens": 8192.0, **resource_metrics}


@torch.no_grad()
def _evaluate_gqa_fp16_cache(student, layer_idx, hidden, rank, world, device, v_first=None):
    source_layer = student.model.layers[layer_idx]
    runtime_layer = copy.deepcopy(source_layer).to(device=device, dtype=torch.float16).eval()
    names = ("chunk64", "chunk128", "chunk256", "decode")
    totals = torch.zeros(len(names), 2, 2, dtype=torch.float64, device=device)
    first_batch = first_value = None
    for batch, first in _hidden_batches(hidden, v_first):
        batch = batch.to(device=device, dtype=torch.float16)
        first = None if first is None else first.to(device=device, dtype=torch.float16)
        if first_batch is None:
            first_batch, first_value = batch, first
        full_tmix, full_block, cache = _fp16_forward_mode(
            runtime_layer, batch, student.config, None, v_first=first
        )
        _gqa_cache_resource_metrics(cache, layer_idx)
        for index, chunk_size in enumerate((64, 128, 256, 1)):
            tmix, block, _ = _fp16_forward_mode(
                runtime_layer, batch, student.config, chunk_size, v_first=first
            )
            for output_idx, (actual, wanted) in enumerate(((tmix, full_tmix), (block, full_block))):
                totals[index, output_idx, 0] += (
                    (actual.float() - wanted.float()).double().square().sum()
                )
                totals[index, output_idx, 1] += wanted.double().square().sum()
    if world > 1:
        dist.all_reduce(totals)
    metrics = {
        f"{name}_{output}_relative_l2": float((error / wanted.clamp_min(1e-24)).sqrt())
        for name, results in zip(names, totals, strict=True)
        for output, (error, wanted) in zip(("tmix", "block"), results, strict=True)
    }
    for name, value in metrics.items():
        if not math.isfinite(value) or value > GQA_RUNTIME_RELATIVE_L2_LIMIT:
            raise RuntimeError(f"FP16 pure RWKV cache parity failed for {name}: {value:.8g}")
    # Broadcast rank-zero failures before any rank can save a layer checkpoint.
    outcome = [None]
    if rank == 0:
        try:
            _fresh_process_gqa_strict_load(student.config, layer_idx, runtime_layer.tmix)
            outcome[0] = _gqa_fp16_soak(
                runtime_layer,
                first_batch[:1, :1],
                student.config,
                None if first_value is None else first_value[:1, :1],
            )
        except Exception as error:
            outcome[0] = str(error)
    if world > 1:
        dist.broadcast_object_list(outcome, src=0)
    if isinstance(outcome[0], str):
        raise RuntimeError(outcome[0])
    metrics.update(outcome[0])
    return metrics


def _mix_teacher(student_output: torch.Tensor, teacher_output: torch.Tensor, gate: float):
    return teacher_output.detach() * gate + student_output * (1 - gate)


@torch.no_grad()
def _project_gqa_feature_geometry(parameters, master_parameters) -> None:
    """Keep learned feature maps full-rank while allowing subspace rotation."""
    for parameter, master in zip(parameters, master_parameters, strict=True):
        if master.ndim != 3 or master.shape[-2:] != (256, 64):
            continue
        orthogonal, _ = torch.linalg.qr(master, mode="reduced")
        master.copy_(orthogonal)
        parameter.copy_(orthogonal.to(parameter.dtype))


def _nmse_loss(actual: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    return (actual.float() - wanted.float()).square().mean() / (
        wanted.float().square().mean() + 1e-6
    )


GQA_GATE_SCHEDULE = (0.9, 0.75, 0.5, 0.25, 0.1, 0.0)
GQA_RUNTIME_RELATIVE_L2_LIMIT = 1e-3


@torch.no_grad()
def _refresh_gqa_stage_cache(
    source_text, student, layer_idx, hidden_cache, cache, device, gate, v_first=None
):
    objective = _LayerObjective(student.model.layers[layer_idx])
    chunks = []
    for hidden, first in _hidden_batches(hidden_cache, v_first):
        hidden = hidden.to(device)
        teacher = _teacher_tmix_output(source_text, layer_idx, hidden) if gate else None
        chunks.append(
            objective(hidden, teacher, gate, None if first is None else first.to(device))[0].cpu()
        )
    # Keep the fixed prefix input in `current`; only gate=0 may advance `next`.
    cache.store(torch.cat(chunks), "next", v_first=v_first)


def _require_gqa_acceptance(native, reference, recall, layer_idx):
    for split, metrics in (("native", native), ("reference", reference), ("recall", recall)):
        _require_finite_metrics(metrics, layer_idx, split)
    failures = []
    for name in ("native_incremental_layer_output_nmse", "native_incremental_tmix_output_nmse"):
        if reference[name] > 1e-3:
            failures.append(f"{name} {reference[name]:.8g} > 0.001")
    for name, value in recall.items():
        if value < 0.9:
            failures.append(f"{name} {value:.8g} < 0.9")
    if failures:
        raise RuntimeError(f"layer {layer_idx} pure RWKV acceptance failed: " + "; ".join(failures))


def _run_gqa_phase(
    source_text,
    student,
    layer_idx,
    train_hidden,
    validation_hidden,
    world,
    device,
    *,
    gate: float,
    phase: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    train_v_first=None,
    validation_v_first=None,
    train_dynamics=True,
):
    layer = student.model.layers[layer_idx]
    tmix = layer.tmix
    student.requires_grad_(False)
    if phase == "attention_transfer":
        parameters = tmix.attention_transfer_parameters()
    elif phase == "distillation":
        transferred = _gqa_transferred_names(tmix)
        parameters = [
            parameter
            for name, parameter in tmix.named_parameters()
            if train_dynamics or name in transferred
        ]
    else:
        raise ValueError(f"unknown GQA alignment phase {phase!r}")
    for parameter in parameters:
        parameter.requires_grad_(True)
    wrapper = tmix
    if world > 1:
        wrapper = DistributedDataParallel(wrapper, device_ids=[device.index])
    steps_per_epoch = math.ceil(len(train_hidden) / 8)
    master_parameters = [
        nn.Parameter(parameter.detach().float().clone()) for parameter in parameters
    ]
    decay_parameters = {id(tmix.w0), id(tmix.w2)}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    master
                    for parameter, master in zip(parameters, master_parameters)
                    if id(parameter) not in decay_parameters
                ],
                "lr": lr,
            },
            {
                "params": [
                    master
                    for parameter, master in zip(parameters, master_parameters)
                    if id(parameter) in decay_parameters
                ],
                "lr": lr * GQA_DECAY_LR_SCALE,
            },
        ],
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )
    scheduler = _schedule(optimizer, epochs * steps_per_epoch, min_scale=0.0)
    best_validation = _evaluate_layer(
        source_text, student, layer_idx, validation_hidden, world, device, validation_v_first
    )
    best_state = {name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()}
    best_recall_pass = min(evaluate_gqa_recall(tmix, use_flash=True).values()) >= 0.9
    best_epoch = -1
    stale = 0
    for epoch in range(epochs):
        train_mixed = torch.zeros((), device=device)
        train_pure = torch.zeros((), device=device)
        for hidden_cpu, first in _hidden_batches(train_hidden, train_v_first):
            hidden = hidden_cpu.to(device)
            with torch.no_grad():
                teacher_tmix = _teacher_tmix_output(source_text, layer_idx, hidden)
            pure_tmix, _ = wrapper(
                layer.input_layernorm(hidden), None if first is None else first.to(device)
            )
            mixed_tmix = _mix_teacher(pure_tmix.float(), teacher_tmix.float(), gate)
            mixed_loss = _nmse_loss(mixed_tmix, teacher_tmix)
            pure_loss = _nmse_loss(pure_tmix, teacher_tmix)
            loss = mixed_loss + 4.0 * pure_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite layer {layer_idx} {phase} loss")
            optimizer.zero_grad(set_to_none=True)
            for parameter in parameters:
                parameter.grad = None
            loss.backward()
            _require_finite_tmix(
                tmix, layer_idx, f"{phase} gate {gate} epoch {epoch}", gradients=True
            )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            for parameter, master in zip(parameters, master_parameters, strict=True):
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
            optimizer.step()
            with torch.no_grad():
                _project_gqa_feature_geometry(parameters, master_parameters)
                for parameter, master in zip(parameters, master_parameters, strict=True):
                    if parameter is tmix.w0:
                        master.clamp_(0, 1 - 1e-4)
                    elif parameter is tmix.a0:
                        master.clamp_(0, 1)
                    parameter.copy_(master.to(parameter.dtype))
            scheduler.step()
            train_mixed += mixed_loss.detach() / steps_per_epoch
            train_pure += pure_loss.detach() / steps_per_epoch
        validation = _evaluate_layer(
            source_text, student, layer_idx, validation_hidden, world, device, validation_v_first
        )
        _require_finite_metrics(validation, layer_idx, f"{phase} gate {gate} epoch {epoch}")
        recall = evaluate_gqa_recall(tmix, use_flash=True)
        recall_pass = min(recall.values()) >= 0.9
        if recall_pass and (
            not best_recall_pass or _validation_is_better(validation, best_validation)
        ):
            best_validation = dict(validation)
            best_recall_pass = True
            best_state = {
                name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        train_mixed_nmse = _mean(train_mixed, world)
        train_pure_nmse = _mean(train_pure, world)
        if world == 1 or dist.get_rank() == 0:
            print(
                {
                    "layer": layer_idx,
                    "alignment_phase": phase,
                    "gate": gate,
                    "phase_epoch": epoch,
                    "train_mixed_tmix_nmse": train_mixed_nmse,
                    "train_pure_tmix_nmse": train_pure_nmse,
                    "validation_pure_tmix_nmse": validation["tmix_output_nmse"],
                    "validation_pure_block_nmse": validation["layer_output_nmse"],
                    "best_pure_block_nmse": best_validation["layer_output_nmse"],
                    "recall_min_hit_at_1": min(recall.values()),
                    "epochs_without_pure_improvement": stale,
                },
                flush=True,
            )
        if stale >= patience:
            break
    tmix.load_state_dict(best_state, strict=True)
    return best_validation, best_epoch


def _align_gqa_layer(
    source_text,
    student,
    layer_idx,
    train_hidden,
    validation_hidden,
    world,
    device,
    *,
    cache=None,
    hidden_cache=None,
    v_first=None,
    skip_transfer=False,
    epochs=8,
    learning_rate=3e-5,
    train_dynamics=True,
    output=None,
):
    tmix = student.model.layers[layer_idx].tmix
    train_v_first = None if v_first is None else v_first[24:]
    validation_v_first = None if v_first is None else v_first[8:24]
    initial_state = {
        name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
    }
    before_distillation = _evaluate_layer(
        source_text, student, layer_idx, validation_hidden, world, device, validation_v_first
    )
    if world == 1 or dist.get_rank() == 0:
        print(
            {
                "layer": layer_idx,
                "stage": "gqa_initial_validation",
                "validation": before_distillation,
                "train_dynamics": train_dynamics,
                "decay_learning_rate_scale": GQA_DECAY_LR_SCALE,
            },
            flush=True,
        )
    try:
        stage_history = []
        attention_validation, attention_epoch = before_distillation, -1
        if not skip_transfer:
            attention_validation, attention_epoch = _run_gqa_phase(
                source_text,
                student,
                layer_idx,
                train_hidden,
                validation_hidden,
                world,
                device,
                gate=GQA_GATE_SCHEDULE[0],
                phase="attention_transfer",
                epochs=16,
                patience=4,
                lr=1e-2,
                weight_decay=0.0,
                train_v_first=train_v_first,
                validation_v_first=validation_v_first,
                train_dynamics=train_dynamics,
            )
            stage_history.append(
                {
                    "gate": GQA_GATE_SCHEDULE[0],
                    "phase": "attention_transfer",
                    **attention_validation,
                }
            )
            if cache is not None:
                _refresh_gqa_stage_cache(
                    source_text,
                    student,
                    layer_idx,
                    hidden_cache,
                    cache,
                    device,
                    GQA_GATE_SCHEDULE[0],
                    v_first,
                )
        for gate in GQA_GATE_SCHEDULE:
            validation, _ = _run_gqa_phase(
                source_text,
                student,
                layer_idx,
                train_hidden,
                validation_hidden,
                world,
                device,
                gate=gate,
                phase="distillation",
                epochs=epochs,
                patience=4,
                lr=learning_rate,
                weight_decay=0.1,
                train_v_first=train_v_first,
                validation_v_first=validation_v_first,
                train_dynamics=train_dynamics,
            )
            stage_history.append({"gate": gate, "phase": "distillation", **validation})
            if cache is not None:
                _refresh_gqa_stage_cache(
                    source_text, student, layer_idx, hidden_cache, cache, device, gate, v_first
                )
        if output is not None and (world == 1 or dist.get_rank() == 0):
            # A gate-zero research candidate is distinct from an accepted layer.
            # Preserve it so runtime fixes can be evaluated without retraining.
            _assert_pure_rwkv_state(tmix)
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in tmix.state_dict().items()
                },
                str(output / "gqa_candidate.safetensors"),
            )
        native = _evaluate_layer(
            source_text, student, layer_idx, validation_hidden, world, device, validation_v_first
        )
        reference = _evaluate_gqa_reference(
            source_text, student, layer_idx, validation_hidden, world, device, validation_v_first
        )
        recall = evaluate_gqa_recall(tmix, use_flash=True)
        if world == 1 or dist.get_rank() == 0:
            print(
                {
                    "layer": layer_idx,
                    "stage": "pure_rwkv_acceptance",
                    "native": native,
                    "reference": reference,
                    "recall": recall,
                },
                flush=True,
            )
        _require_gqa_acceptance(native, reference, recall, layer_idx)
        fp16_cache = _evaluate_gqa_fp16_cache(
            student,
            layer_idx,
            validation_hidden,
            dist.get_rank() if world > 1 else 0,
            world,
            device,
            validation_v_first,
        )
        return {
            "attention_transfer_best_epoch": attention_epoch,
            "attention_transfer_best_validation": attention_validation,
            "stage_history": stage_history,
            "reference_validation": reference,
            "native_validation": native,
            "before_distillation_validation": before_distillation,
            "recall_validation": recall,
            "fp16_cache_validation": fp16_cache,
        }
    except Exception:
        tmix.load_state_dict(initial_state, strict=True)
        raise


def _layerwise(
    source_text,
    student,
    ids,
    output,
    rank,
    world,
    device,
    through_layer,
    *,
    prefix_cache: Path | None = None,
    gqa_initial_checkpoint=None,
    gqa_epochs=8,
    gqa_learning_rate=3e-5,
    gqa_train_dynamics=True,
):
    cache = LastLayerCache(output / "cache", rank)
    if prefix_cache is None:
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
    else:
        completed = 3
        if through_layer != completed:
            raise ValueError("GQA prefix-cache mode is restricted to layer 3")
        if _completed_layers(output):
            raise ValueError("GQA prefix-cache mode requires a fresh output directory")
        reused_cache = LastLayerCache(prefix_cache, rank)
        reused_hidden = reused_cache.load()
        reused_first = reused_cache.load_v_first()
        if reused_first is None:
            raise ValueError("prefix cache must be rebuilt with first-layer values for RWKV")
        if reused_hidden.shape != (ids.shape[0], ids.shape[1], student.config.hidden_size):
            raise ValueError(
                "GQA prefix cache shape does not match the immutable packed rows: "
                f"cache={tuple(reused_hidden.shape)} ids={tuple(ids.shape)}"
            )
        cache.store(reused_hidden, "next", v_first=reused_first)
        cache.advance()
        prefix_strict_pass = False
        if rank == 0:
            print(
                {
                    "stage": "gqa_prefix_cache_reuse",
                    "prefix_cache": prefix_cache.as_posix(),
                    "start_layer": completed,
                    "formal_prefix_acceptance": False,
                },
                flush=True,
            )
    strict_failures: list[str] = []
    for index in range(completed, through_layer + 1):
        hidden_cache = cache.load()
        first_cache = cache.load_v_first()
        if hidden_cache.shape[0] < 32:
            raise ValueError("each rank needs at least 32 packed rows for isolated data splits")
        init_local = hidden_cache[:8].to(device)
        validation_hidden = hidden_cache[8:24]
        validation_local = validation_hidden.to(device)
        train_hidden = hidden_cache[24:]
        init_hidden = _gather_init_hidden(init_local, world)
        validation_for_init = _gather_init_hidden(validation_local, world)
        is_gdn = student.config.source_layer_types[index] == "linear_attention"
        warm_start = _warm_start_checkpoint(gqa_initial_checkpoint, index)
        metrics = (
            _initialize_layer(source_text, student, index, init_hidden, validation_for_init)
            if rank == 0 and warm_start is None
            else {}
        )
        if rank == 0 and warm_start is not None:
            if is_gdn:
                _load_layer_checkpoint(warm_start.parent, index, student.model.layers[index].tmix)
            else:
                _load_gqa_initial_checkpoint(student.model.layers[index].tmix, warm_start)
            print(
                {
                    "stage": "gqa_initial_checkpoint",
                    "path": str(warm_start),
                    "layer": index,
                    "warm_start_mode": "full_layer" if is_gdn else "transferred_gqa",
                    "train_dynamics": gqa_train_dynamics,
                    "epochs_per_gate": gqa_epochs,
                    "learning_rate": gqa_learning_rate,
                },
                flush=True,
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
        if is_gdn and warm_start is not None:
            if rank == 0:
                _save_layer(output, index, tmix)
            if world > 1:
                dist.barrier()
            _cache_layer(layer, hidden_cache, first_cache, cache, device)
            continue
        if not is_gdn:
            result = _align_gqa_layer(
                source_text,
                student,
                index,
                train_hidden,
                validation_hidden,
                world,
                device,
                cache=cache,
                hidden_cache=hidden_cache,
                v_first=first_cache,
                skip_transfer=warm_start is not None,
                epochs=gqa_epochs,
                learning_rate=gqa_learning_rate,
                train_dynamics=gqa_train_dynamics,
                output=output,
            )
            train_metrics = _evaluate_layer(
                source_text,
                student,
                index,
                train_hidden,
                world,
                device,
                None if first_cache is None else first_cache[24:],
            )
            validation_metrics = result["native_validation"]
            layer_strict_pass = (
                validation_metrics["layer_output_nmse"]
                <= result["before_distillation_validation"]["layer_output_nmse"]
            )
            if prefix_cache is None:
                prefix_strict_pass = prefix_strict_pass and layer_strict_pass
            artifact_pass = layer_strict_pass if prefix_cache is not None else prefix_strict_pass
            if rank == 0:
                print(
                    {
                        "layer": index,
                        "stage": "pure_rwkv_gqa_best_checkpoint",
                        **{f"train_{name}": value for name, value in train_metrics.items()},
                        **{
                            f"validation_{name}": value
                            for name, value in validation_metrics.items()
                        },
                        **{
                            f"reference_{name}": value
                            for name, value in result["reference_validation"].items()
                        },
                        **{
                            f"runtime_{name}": value
                            for name, value in result["fp16_cache_validation"].items()
                        },
                        "attention_transfer_best_epoch": result["attention_transfer_best_epoch"],
                        "stage_history": result["stage_history"],
                        "initialization_metrics": metrics,
                        "prefix_strict_pass": prefix_strict_pass,
                        "layer_artifact_pass": artifact_pass,
                    },
                    flush=True,
                )
                if artifact_pass:
                    _save_layer(output, index, tmix)
            if world > 1:
                dist.barrier()
            _cache_layer(layer, hidden_cache, first_cache, cache, device)
            continue
        wrapper = _LayerObjective(layer)
        if world > 1:
            wrapper = DistributedDataParallel(wrapper, device_ids=[device.index])
        lr = 1e-5
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
        for epoch in range(initial_epochs):
            layer_output_total = torch.zeros((), device=device)
            tmix_output_total = torch.zeros((), device=device)
            for training_batch in loader:
                hidden = training_batch[0].to(device)
                with torch.no_grad():
                    wanted_layer = _teacher_layer(source_text, index, hidden)
                    wanted_tmix = _teacher_tmix_output(source_text, index, hidden)
                actual_layer, actual_tmix = wrapper(hidden)
                layer_output_loss = (
                    actual_layer.float() - wanted_layer.float()
                ).square().mean() / (wanted_layer.float().square().mean() + 1e-6)
                tmix_output_loss = (actual_tmix.float() - wanted_tmix.float()).square().mean() / (
                    wanted_tmix.float().square().mean() + 1e-6
                )
                loss = layer_output_loss + tmix_output_loss
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
            previous_best_tmix = best_validation["tmix_output_nmse"]
            improved = _validation_is_better(validation_metrics, best_validation)
            if improved:
                best_validation = dict(validation_metrics)
                best_state = {
                    name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
                }
                best_epoch = epoch
                if validation_metrics["tmix_output_nmse"] < previous_best_tmix:
                    epochs_without_validation_tmix_improvement = 0
                else:
                    epochs_without_validation_tmix_improvement += 1
            else:
                epochs_without_validation_tmix_improvement += 1
            if rank == 0:
                print(
                    {
                        "layer": index,
                        "epoch": epoch,
                        "alignment_phase": "distillation",
                        "phase_epoch": epoch,
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
            if epochs_without_validation_tmix_improvement >= VALIDATION_TMIX_PATIENCE:
                break
        tmix.load_state_dict(best_state)
        _require_finite_tmix(tmix, index, "best validation checkpoint")
        train_metrics = _evaluate_layer(source_text, student, index, train_hidden, world, device)
        _require_finite_metrics(train_metrics, index, "best checkpoint on train")
        validation_metrics = _evaluate_layer(
            source_text, student, index, validation_hidden, world, device
        )
        _require_finite_metrics(validation_metrics, index, "best checkpoint on validation")
        validation_pass = (
            validation_metrics["layer_output_nmse"]
            <= before_distillation_validation["layer_output_nmse"]
        )
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
                        "before_distillation" if best_epoch < 0 else "distillation"
                    ),
                    "best_phase_epoch": best_epoch,
                    "initialization_metrics": metrics,
                },
                flush=True,
            )
        if rank == 0 and prefix_strict_pass:
            _save_layer(output, index, tmix)
        if world > 1:
            dist.barrier()
        layer.train()
        _cache_layer(layer, hidden_cache, first_cache, cache, device)
    if strict_failures:
        raise RuntimeError(
            "layer validation regressed relative to initialization after "
            f"measuring through layer {through_layer}: {'; '.join(strict_failures)}"
        )
    return cache


def _set_global_parameters(student):
    student.requires_grad_(False)
    ordinary = []
    for layer in student.model.layers:
        tmix = layer.tmix
        tmix.requires_grad_(True)
        ordinary.extend(tmix.parameters())
    return ordinary


def _global_kl(
    source_outer, source_text, student, ids, tokenizer, output, rank, world, device, epochs=3
):
    ordinary = _set_global_parameters(student)
    master_parameters = [nn.Parameter(parameter.detach().float().clone()) for parameter in ordinary]
    decay_ids = {
        id(parameter)
        for name, parameter in student.named_parameters()
        if name.endswith((".tmix.w0", ".tmix.w2"))
    }
    bounded = {
        id(parameter): (1 - 1e-4 if name.endswith(".w0") else 1)
        for name, parameter in student.named_parameters()
        if name.endswith((".tmix.w0", ".tmix.a0"))
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    master
                    for parameter, master in zip(ordinary, master_parameters)
                    if id(parameter) not in decay_ids
                ],
                "lr": 1e-6,
            },
            {
                "params": [
                    master
                    for parameter, master in zip(ordinary, master_parameters)
                    if id(parameter) in decay_ids
                ],
                "lr": 1e-6 * GQA_DECAY_LR_SCALE,
            },
        ],
        weight_decay=0.1,
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
            for parameter in ordinary:
                parameter.grad = None
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
            torch.nn.utils.clip_grad_norm_(ordinary, 1.0, error_if_nonfinite=True)
            for parameter, master in zip(ordinary, master_parameters, strict=True):
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
            optimizer.step()
            with torch.no_grad():
                for parameter, master in zip(ordinary, master_parameters, strict=True):
                    if id(parameter) in bounded:
                        master.clamp_(0, bounded[id(parameter)])
                    parameter.copy_(master.to(parameter.dtype))
            scheduler.step()
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
        _assert_pure_rwkv_state(student)
        for index, layer in enumerate(student.model.layers):
            if student.config.source_layer_types[index] == "full_attention":
                recall = evaluate_gqa_recall(layer.tmix, use_flash=True)
                if min(recall.values()) < 0.9:
                    raise RuntimeError(f"global KL regressed layer {index} recall: {recall}")
        student.generation_config.eos_token_id = list(
            dict.fromkeys((student.config.eos_token_id, tokenizer.eos_token_id))
        )
        student.generation_config.pad_token_id = tokenizer.pad_token_id
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
                token_ids, use_cache=True, do_sample=False, max_new_tokens=512
            )
        answer_ids = generated[0, token_ids.shape[1] :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        tokens = answer_ids.tolist()
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        finished = bool(tokens) and tokens[-1] in eos
        coherent = bool(answer.strip()) and "�" not in answer
        if tokens:
            coherent = (
                coherent and max(tokens.count(token) for token in set(tokens)) < len(tokens) * 0.8
            )
        passed = passed and coherent and finished
        records.append(
            {
                "prompt": prompt,
                "token_ids": token_ids[0].tolist(),
                "output": answer,
                "generated_tokens": len(tokens),
                "finish_reason": "eos" if finished else "length",
            }
        )
    result = {"scope": "generation_smoke", "passed": passed, "generations": records}
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
    *,
    gqa_initial_checkpoint=None,
    gqa_epochs=8,
    gqa_learning_rate=3e-5,
    gqa_train_dynamics=True,
):
    rank, world, device = _distributed()
    torch.manual_seed(42)
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
        raise ValueError(f"through_layer must be in [0, {student.config.num_hidden_layers - 1}]")
    _layerwise(
        source_text,
        student,
        local_ids,
        output_path,
        rank,
        world,
        device,
        through_layer,
        gqa_initial_checkpoint=gqa_initial_checkpoint,
        gqa_epochs=gqa_epochs,
        gqa_learning_rate=gqa_learning_rate,
        gqa_train_dynamics=gqa_train_dynamics,
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
            raise RuntimeError("generation smoke test failed after global KL refinement")
    if world > 1:
        dist.destroy_process_group()


def convert_gqa_from_prefix_cache(
    source: str,
    output: str,
    prefix_cache: str,
    *,
    gqa_initial_checkpoint=None,
    gqa_epochs=8,
    gqa_learning_rate=3e-5,
    gqa_train_dynamics=True,
) -> None:
    """Run only layer-3 GQA work from a verified post-layer-2 rank-local cache."""
    rank, world, device = _distributed()
    torch.manual_seed(42)
    output_path = Path(output).resolve()
    prefix_path = Path(prefix_cache).resolve()
    if output_path == Path(source).resolve() or output_path == prefix_path:
        raise ValueError("GQA output must not overwrite the source or prefix cache")
    output_path.mkdir(parents=True, exist_ok=True)
    packed_path = output_path / "packed_sequences.pt"
    if not packed_path.is_file():
        raise FileNotFoundError("GQA prefix-cache mode requires immutable packed_sequences.pt")
    _require_gqa_packed_provenance(packed_path)
    packed_objects = [
        torch.load(packed_path, map_location="cpu", weights_only=True) if rank == 0 else None
    ]
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
        3,
        prefix_cache=prefix_path,
        gqa_initial_checkpoint=gqa_initial_checkpoint,
        gqa_epochs=gqa_epochs,
        gqa_learning_rate=gqa_learning_rate,
        gqa_train_dynamics=gqa_train_dynamics,
    )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def continue_global_kl(source: str, output: str) -> None:
    """Run another global KL epoch from the saved final model."""
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVForCausalLM

    rank, world, device = _distributed()
    output_path = Path(output).resolve()
    packed_path = output_path / "packed_sequences.pt"
    if not packed_path.is_file():
        raise FileNotFoundError(f"missing packed sequences: {packed_path}")
    ids = torch.load(packed_path, map_location="cpu", weights_only=True)
    source_outer, source_text = load_qwen_teacher(source, torch.bfloat16, device)
    student = Qwen2RWKVForCausalLM.from_pretrained(output_path, dtype=torch.bfloat16).to(device)
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
    parser.add_argument("--gqa-prefix-cache")
    parser.add_argument(
        "--gqa-initial-checkpoint",
        help="layer-3 initialization file or a complete directory of layer checkpoints",
    )
    parser.add_argument(
        "--gqa-epochs",
        type=int,
        default=8,
        help="epochs per gate; zero replays acceptance without optimization",
    )
    parser.add_argument("--gqa-learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--gqa-freeze-dynamics",
        action="store_true",
        help="matched-budget ablation: train only transferred parameters",
    )
    args = parser.parse_args()
    if args.gqa_epochs < 0 or args.gqa_learning_rate <= 0:
        parser.error("GQA epochs must be nonnegative and learning rate must be positive")
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
    if args.gqa_prefix_cache is not None:
        convert_gqa_from_prefix_cache(
            args.source,
            args.output,
            args.gqa_prefix_cache,
            gqa_initial_checkpoint=args.gqa_initial_checkpoint,
            gqa_epochs=args.gqa_epochs,
            gqa_learning_rate=args.gqa_learning_rate,
            gqa_train_dynamics=not args.gqa_freeze_dynamics,
        )
        return
    convert_qwen3_5_2b(
        args.source,
        args.output,
        args.agentic,
        args.math_dataset,
        args.through_layer,
        gqa_initial_checkpoint=args.gqa_initial_checkpoint,
        gqa_epochs=args.gqa_epochs,
        gqa_learning_rate=args.gqa_learning_rate,
        gqa_train_dynamics=not args.gqa_freeze_dynamics,
    )


if __name__ == "__main__":
    main()
