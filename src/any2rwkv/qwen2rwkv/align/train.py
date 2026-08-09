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
    ORACLE_BLOCK_NMSE_GATE,
    exact_gqa_teacher_targets,
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
LAYER_OUTPUT_NMSE_TARGET = 3e-3
VALIDATION_TMIX_PATIENCE = 5
GQA_PACKED_SHA256 = "1d039b73dcafd9783a7e872f682cf64728cb31f6090ef54c2882ca3bc0919336"


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
        None,
        torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
    )[0]


def _freeze_value_residual(tmix) -> None:
    if not hasattr(tmix, "value_residual_scale"):
        return
    tmix.value_residual_scale.data.zero_()
    for parameter in (tmix.v0, tmix.v1, tmix.v2, tmix.value_residual_scale):
        parameter.requires_grad_(False)


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
        _, validation_embeddings = _position(source_text, validation_normalized)
        metrics = initialize_gqa_layer(
            source_layer.self_attn,
            target,
            init_normalized,
            init_embeddings,
            validation_normalized,
            validation_embeddings,
        )
        with torch.no_grad():
            probe = init_hidden[:1, : min(128, init_hidden.shape[1])]
            wanted_block = _teacher_layer(source_text, layer_idx, probe).float()
            target_layer = student.model.layers[layer_idx]
            normalized = target_layer.input_layernorm(probe)
            tmix_output = target_layer.tmix.reference_forward(normalized)
            residual = probe + tmix_output.to(probe)
            actual_block = (
                residual
                + target_layer.mlp(target_layer.post_attention_layernorm(residual))
            ).float()
            block_nmse = float(
                (actual_block - wanted_block).square().sum()
                / wanted_block.square().sum().clamp_min(1e-24)
            )
        if not math.isfinite(block_nmse) or block_nmse > 1e-6:
            raise RuntimeError(
                "bounded Hedgehog exact-prefix complete-Block identity failed: "
                f"block_nmse={block_nmse:.8g}"
            )
        metrics["hedgehog_exact_prefix_block_nmse"] = block_nmse
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
        return min_scale + (1 - min_scale) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

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
            + source_layer.mlp(
                source_layer.post_attention_layernorm(wanted_residual)
            )
        ).float()
        actual_residual = batch + actual_tmix.to(batch)
        actual_layer = (
            actual_residual
            + layer.mlp(layer.post_attention_layernorm(actual_residual))
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
                output, _ = layer(
                    hidden.to(device),
                    None,
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

    def forward(self, hidden, exact_attention=None):
        normalized = self.layer.input_layernorm(hidden)
        tmix_output, _ = self.layer.tmix(
            normalized,
            None,
            None,
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device),
        )
        if exact_attention is not None:
            actual = self.layer.tmix._last_attention_heads.transpose(1, 2).float()
            wanted = exact_attention.float()
            attention_loss = (actual - wanted).square().sum() / wanted.square().sum().clamp_min(
                1e-12
            )
            return hidden, tmix_output, attention_loss
        residual = hidden + tmix_output
        layer_output = residual + self.layer.mlp(self.layer.post_attention_layernorm(residual))
        return layer_output, tmix_output, tmix_output.float().new_zeros(())


@torch.no_grad()
def _precompute_gqa_teacher_targets(source_text, layer_idx, hidden, device):
    exact_chunks = []
    source_layer = source_text.layers[layer_idx]
    for batch in hidden.split(8):
        batch = batch.to(device)
        normalized = source_layer.input_layernorm(batch)
        _, embeddings = _position(source_text, normalized)
        exact = exact_gqa_teacher_targets(
            source_layer.self_attn,
            normalized,
            embeddings,
        )
        exact_chunks.append(exact.to(torch.bfloat16).cpu())
    return torch.cat(exact_chunks)


@torch.no_grad()
def _evaluate_gqa_reference(source_text, student, layer_idx, hidden, world, device):
    totals = torch.zeros(16, dtype=torch.float64, device=device)
    target_layer = student.model.layers[layer_idx]
    target_layer.train()
    saved_attention = target_layer.tmix._last_attention_heads
    saved_sidecar_metrics = target_layer.tmix._last_sidecar_metrics
    target_layer.tmix._last_attention_heads = None
    target_layer.tmix._last_sidecar_metrics = {}
    try:
        reference_layer_module = copy.deepcopy(target_layer).to(
            device=device, dtype=torch.float32
        ).eval()
    finally:
        target_layer.tmix._last_attention_heads = saved_attention
        target_layer.tmix._last_sidecar_metrics = saved_sidecar_metrics
    for batch in hidden.split(8):
        batch = batch.to(device)
        reference_batch = batch.float()
        normalized = reference_layer_module.input_layernorm(reference_batch)
        reference_attention, reference_gate = (
            reference_layer_module.tmix.attention_heads_reference(normalized)
        )
        reference_attention = reference_attention.float()
        reference_gate = reference_gate.float()
        reference_mixed = reference_attention.transpose(1, 2).reshape(
            *reference_batch.shape
        ) * torch.sigmoid(reference_gate)
        reference_tmix = reference_layer_module.tmix.o_proj(reference_mixed).float()
        native_tmix = _student_tmix_output(student, layer_idx, batch).float()
        source_layer = source_text.layers[layer_idx]
        source_normalized = source_layer.input_layernorm(batch)
        _, embeddings = _position(source_text, source_normalized)
        wanted_attention_bt = exact_gqa_teacher_targets(
            source_layer.self_attn,
            source_normalized,
            embeddings,
        )
        projected = source_layer.self_attn.q_proj(source_normalized).view(
            batch.shape[0],
            batch.shape[1],
            source_layer.self_attn.config.num_attention_heads,
            2 * source_layer.self_attn.head_dim,
        )
        _, wanted_gate = projected.chunk(2, dim=-1)
        wanted_tmix = source_layer.self_attn.o_proj(
            wanted_attention_bt.reshape(*batch.shape)
            * torch.sigmoid(wanted_gate.reshape(*batch.shape))
        ).float()

        def complete(layer, layer_input, tmix_output):
            residual = layer_input + tmix_output.to(layer_input)
            return residual + layer.mlp(
                layer.post_attention_layernorm(residual)
            )

        reference_layer = complete(
            reference_layer_module, reference_batch, reference_tmix
        ).float()
        native_layer = complete(target_layer, batch, native_tmix).float()
        wanted_layer = complete(
            source_layer, batch, wanted_tmix
        ).float()
        pairs = (
            (reference_attention, wanted_attention_bt.transpose(1, 2)),
            (reference_tmix, wanted_tmix),
            (reference_layer, wanted_layer),
            (native_layer, reference_layer),
        )
        for offset, (actual, wanted) in enumerate(pairs):
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

    reference_attention_nmse, reference_attention_cosine = metrics(0)
    reference_tmix_nmse, reference_tmix_cosine = metrics(4)
    reference_block_nmse, reference_block_cosine = metrics(8)
    native_incremental_block_nmse, _ = metrics(12)
    del reference_layer_module
    return {
        "reference_attention_nmse": reference_attention_nmse,
        "reference_attention_cosine": reference_attention_cosine,
        "reference_tmix_output_nmse": reference_tmix_nmse,
        "reference_tmix_output_cosine": reference_tmix_cosine,
        "reference_layer_output_nmse": reference_block_nmse,
        "reference_layer_output_cosine": reference_block_cosine,
        "native_incremental_layer_output_nmse": native_incremental_block_nmse,
    }


@torch.no_grad()
def _fp16_forward_mode(layer, hidden, config, chunk_size, cache=None):
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVCache

    cache = Qwen2RWKVCache(config) if cache is None else cache
    length = hidden.shape[1]
    chunk_size = length if chunk_size is None else chunk_size
    tmix_chunks = []
    block_chunks = []
    attention_chunks = []
    for start in range(0, length, chunk_size):
        chunk = hidden[:, start : start + chunk_size]
        normalized = layer.input_layernorm(chunk)
        tmix_output, _ = layer.tmix(
            normalized,
            None,
            cache,
            torch.ones(chunk.shape[:2], dtype=torch.bool, device=chunk.device),
        )
        residual = chunk + tmix_output
        block_output = residual + layer.mlp(
            layer.post_attention_layernorm(residual)
        )
        cache.layers[layer.layer_idx].mark_updated(chunk.shape[1])
        tmix_chunks.append(tmix_output)
        block_chunks.append(block_output)
        attention_chunks.append(layer.tmix._last_attention_heads)
    return (
        torch.cat(tmix_chunks, dim=1),
        torch.cat(block_chunks, dim=1),
        torch.cat(attention_chunks, dim=2),
        cache,
    )


def _fresh_process_gqa_strict_load(config, layer_idx: int, tmix) -> None:
    with tempfile.TemporaryDirectory(prefix="any2rwkv-gqa-strict-") as directory:
        root = Path(directory)
        config_path = root / "config.json"
        state_path = root / "layer.safetensors"
        config.to_json_file(config_path)
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in tmix.state_dict().items()
            },
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
                "assert not any('lora' in name for name in module.state_dict())",
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
                "fresh-process strict GQA layer load failed: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )


def _gqa_cache_resource_metrics(cache, layer_idx: int) -> dict[str, float]:
    layer = cache.layers[layer_idx]
    tensors = (layer.gqa_keys, layer.gqa_values, layer.gqa_scores, layer.gqa_positions)
    if any(value is None for value in tensors):
        raise RuntimeError("bounded GQA sidecar was not initialized")
    batch = layer.gqa_keys.shape[0]
    valid = layer.gqa_positions >= 0
    if valid.sum(-1).max() > 128 or layer.gqa_positions.shape[1] != 128:
        raise RuntimeError("bounded GQA sidecar exceeded 128 exact slots")
    sidecar_bytes = (
        layer.gqa_keys.numel() * layer.gqa_keys.element_size()
        + layer.gqa_values.numel() * layer.gqa_values.element_size()
    ) // batch
    recurrent = layer.recurrent_states[0]
    recurrent_bytes = recurrent.numel() * recurrent.element_size() // batch
    control_bytes = (
        layer.gqa_scores.numel() * layer.gqa_scores.element_size()
        + layer.gqa_positions.numel() * layer.gqa_positions.element_size()
        + cache.elapsed[layer_idx].numel() * cache.elapsed[layer_idx].element_size()
    ) // batch
    if sidecar_bytes != 256 * 1024 or recurrent_bytes != 2 * 1024 * 1024:
        raise RuntimeError(
            "bounded GQA persistent payload changed: "
            f"sidecar={sidecar_bytes}, recurrent={recurrent_bytes}"
        )
    return {
        "sidecar_slots": float(valid.sum(-1).max()),
        "sidecar_kv_bytes_per_sequence": float(sidecar_bytes),
        "recurrent_bytes_per_sequence": float(recurrent_bytes),
        "control_bytes_per_sequence": float(control_bytes),
        "total_fixed_state_bytes_per_sequence": float(
            sidecar_bytes + recurrent_bytes + control_bytes
        ),
    }


@torch.no_grad()
def _gqa_fp16_soak(layer, seed_hidden, config) -> dict[str, float]:
    from ..transformers.modeling_qwen2rwkv import Qwen2RWKVCache

    cache = Qwen2RWKVCache(config)
    checkpoints = {512, 4096, 8192}
    completed = 0
    resource_metrics = {}
    while completed < 8192:
        length = min(256, 8192 - completed)
        hidden = seed_hidden.expand(1, length, -1).contiguous()
        normalized = layer.input_layernorm(hidden)
        output, _ = layer.tmix(
            normalized,
            None,
            cache,
            torch.ones(1, length, dtype=torch.bool, device=hidden.device),
        )
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite FP16 output during 8192-token cache soak")
        cache.layers[layer.layer_idx].mark_updated(length)
        completed += length
        if completed in checkpoints:
            resource_metrics = _gqa_cache_resource_metrics(
                cache, layer.layer_idx
            )
            if int(cache.elapsed[layer.layer_idx][0]) != completed:
                raise RuntimeError(
                    "FlashRWKV2 elapsed state diverged during cache soak: "
                    f"elapsed={int(cache.elapsed[layer.layer_idx][0])}, "
                    f"expected={completed}"
                )
    return {"soak_tokens": 8192.0, **resource_metrics}


@torch.no_grad()
def _evaluate_gqa_fp16_cache(
    source_text,
    student,
    layer_idx,
    hidden,
    rank,
    world,
    device,
):
    source_layer_copy = student.model.layers[layer_idx]
    saved_attention = source_layer_copy.tmix._last_attention_heads
    saved_sidecar_metrics = source_layer_copy.tmix._last_sidecar_metrics
    source_layer_copy.tmix._last_attention_heads = None
    source_layer_copy.tmix._last_sidecar_metrics = {}
    try:
        runtime_layer = copy.deepcopy(source_layer_copy).to(
            device=device, dtype=torch.float16
        ).eval()
    finally:
        source_layer_copy.tmix._last_attention_heads = saved_attention
        source_layer_copy.tmix._last_sidecar_metrics = saved_sidecar_metrics
    pair_names = (
        "fp16_attention",
        "fp16_tmix",
        "fp16_block",
        "chunk64_block_incremental",
        "chunk128_block_incremental",
        "chunk256_block_incremental",
        "decode_block_incremental",
        "cache_repeat_select_block_incremental",
        "cache_reset_block_incremental",
    )
    totals = torch.zeros(len(pair_names), 2, dtype=torch.float64, device=device)

    def accumulate(index, actual, wanted):
        totals[index, 0].add_((actual.float() - wanted.float()).double().square().sum())
        totals[index, 1].add_(wanted.float().double().square().sum())

    first_batch = None
    for batch in hidden.split(8):
        batch = batch.to(device)
        if first_batch is None:
            first_batch = batch
        source_layer = source_text.layers[layer_idx]
        source_normalized = source_layer.input_layernorm(batch)
        _, embeddings = _position(source_text, source_normalized)
        wanted_attention_bt = exact_gqa_teacher_targets(
            source_layer.self_attn,
            source_normalized,
            embeddings,
        )
        projected = source_layer.self_attn.q_proj(source_normalized).view(
            batch.shape[0],
            batch.shape[1],
            source_layer.self_attn.config.num_attention_heads,
            2 * source_layer.self_attn.head_dim,
        )
        _, gate = projected.chunk(2, dim=-1)
        wanted_tmix = source_layer.self_attn.o_proj(
            wanted_attention_bt.reshape(*batch.shape)
            * torch.sigmoid(gate.reshape(*batch.shape))
        )
        wanted_residual = batch + wanted_tmix
        wanted_block = wanted_residual + source_layer.mlp(
            source_layer.post_attention_layernorm(wanted_residual)
        )
        wanted_attention = wanted_attention_bt.transpose(1, 2)
        batch = batch.half()
        full_tmix, full_block, full_attention, full_cache = _fp16_forward_mode(
            runtime_layer, batch, student.config, None
        )
        accumulate(0, full_attention, wanted_attention)
        accumulate(1, full_tmix, wanted_tmix)
        accumulate(2, full_block, wanted_block)
        _gqa_cache_resource_metrics(full_cache, layer_idx)
        for metric_index, chunk_size in enumerate((64, 128, 256, 1), start=3):
            _, chunk_block, _, _ = _fp16_forward_mode(
                runtime_layer, batch, student.config, chunk_size
            )
            accumulate(metric_index, chunk_block, full_block)

    probe = first_batch[:2].half()
    prefix = probe[:, :128]
    continuation = probe[:, 128:192]
    _, baseline_prefix, _, baseline_cache = _fp16_forward_mode(
        runtime_layer, prefix, student.config, 64
    )
    _, baseline_continuation, _, baseline_cache = _fp16_forward_mode(
        runtime_layer,
        continuation,
        student.config,
        64,
        cache=baseline_cache,
    )
    _, _, _, reordered_cache = _fp16_forward_mode(
        runtime_layer, prefix, student.config, 64
    )
    reordered_cache.batch_repeat_interleave(2)
    reordered_cache.batch_select_indices(
        torch.tensor([1, 2], dtype=torch.long, device=device)
    )
    _, reordered_continuation, _, reordered_cache = _fp16_forward_mode(
        runtime_layer,
        continuation,
        student.config,
        64,
        cache=reordered_cache,
    )
    accumulate(7, reordered_continuation, baseline_continuation)
    reordered_cache.reset()
    _, reset_prefix, _, reset_cache = _fp16_forward_mode(
        runtime_layer,
        prefix,
        student.config,
        64,
        cache=reordered_cache,
    )
    accumulate(8, reset_prefix, baseline_prefix)
    _gqa_cache_resource_metrics(reset_cache, layer_idx)

    if world > 1:
        dist.all_reduce(totals)
    metrics = {
        f"{name}_nmse": float(error / wanted.clamp_min(1e-24))
        for name, (error, wanted) in zip(pair_names, totals, strict=True)
    }
    for name in pair_names[3:]:
        if metrics[f"{name}_nmse"] > 1e-3:
            raise RuntimeError(
                f"FP16 cache parity gate failed for {name}: "
                f"{metrics[f'{name}_nmse']:.8g} > 0.001"
            )

    if rank == 0:
        _fresh_process_gqa_strict_load(
            student.config, layer_idx, runtime_layer.tmix
        )
        soak = _gqa_fp16_soak(
            runtime_layer,
            first_batch[:1, :1].half(),
            student.config,
        )
    else:
        soak = {}
    if world > 1:
        dist.barrier()
    return {**metrics, **soak}


def _run_gqa_phase(
    source_text,
    student,
    layer_idx,
    train_hidden,
    validation_hidden,
    world,
    device,
    *,
    phase: str,
    train_attention: torch.Tensor | None,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
):
    layer = student.model.layers[layer_idx]
    tmix = layer.tmix
    student.requires_grad_(False)
    if phase == "attention_transfer":
        parameters = tmix.attention_transfer_parameters()
        if train_attention is None:
            raise ValueError("attention transfer requires exact teacher heads")
        dataset = TensorDataset(train_hidden, train_attention)
    elif phase == "low_rank_correction":
        parameters = tmix.lora_parameters()
        if not parameters:
            raise RuntimeError("low-rank correction requires enabled LoRA parameters")
        dataset = TensorDataset(train_hidden)
    else:
        raise ValueError(f"unknown GQA alignment phase {phase!r}")
    for parameter in parameters:
        parameter.requires_grad_(True)
    wrapper = _LayerObjective(layer)
    if world > 1:
        wrapper = DistributedDataParallel(wrapper, device_ids=[device.index])
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    master_parameters = [
        nn.Parameter(parameter.detach().float().clone()) for parameter in parameters
    ]
    optimizer = torch.optim.AdamW(
        master_parameters,
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )
    scheduler = _schedule(optimizer, epochs * len(loader), min_scale=0.0)
    best_validation = _evaluate_layer(
        source_text, student, layer_idx, validation_hidden, world, device
    )
    best_state = {
        name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
    }
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(epochs):
        attention_total = torch.zeros((), device=device)
        block_total = torch.zeros((), device=device)
        tmix_total = torch.zeros((), device=device)
        for training_batch in loader:
            hidden = training_batch[0].to(device)
            exact_attention = (
                training_batch[1].to(device)
                if phase == "attention_transfer"
                else None
            )
            with torch.no_grad():
                if phase == "low_rank_correction":
                    wanted_tmix = _teacher_tmix_output(
                        source_text, layer_idx, hidden
                    )
                    source_layer = source_text.layers[layer_idx]
                    wanted_residual = hidden + wanted_tmix
                    wanted_layer = wanted_residual + source_layer.mlp(
                        source_layer.post_attention_layernorm(wanted_residual)
                    )
                else:
                    wanted_tmix = None
                    wanted_layer = None
            actual_layer, actual_tmix, attention_loss = wrapper(
                hidden, exact_attention
            )
            if phase == "attention_transfer":
                loss = attention_loss
                block_loss = loss.detach().new_zeros(())
                tmix_loss = loss.detach().new_zeros(())
            else:
                block_loss = (
                    (actual_layer.float() - wanted_layer.float()).square().mean()
                    / (wanted_layer.float().square().mean() + 1e-6)
                )
                tmix_loss = (
                    (actual_tmix.float() - wanted_tmix.float()).square().mean()
                    / (wanted_tmix.float().square().mean() + 1e-6)
                )
                loss = 4 * block_loss + tmix_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite layer {layer_idx} {phase} loss"
                )
            optimizer.zero_grad(set_to_none=True)
            for parameter in parameters:
                parameter.grad = None
            loss.backward()
            _require_finite_tmix(
                tmix,
                layer_idx,
                f"{phase} epoch {epoch} backward",
                gradients=True,
            )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            for parameter, master in zip(parameters, master_parameters, strict=True):
                master.grad = (
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().float()
                )
            optimizer.step()
            with torch.no_grad():
                for parameter, master in zip(
                    parameters, master_parameters, strict=True
                ):
                    parameter.copy_(master.to(parameter.dtype))
            scheduler.step()
            attention_total += attention_loss.detach() / len(loader)
            block_total += block_loss.detach() / len(loader)
            tmix_total += tmix_loss.detach() / len(loader)
        train_attention_nmse = _mean(attention_total, world)
        train_block_nmse = _mean(block_total, world)
        train_tmix_nmse = _mean(tmix_total, world)
        validation = _evaluate_layer(
            source_text, student, layer_idx, validation_hidden, world, device
        )
        _require_finite_metrics(validation, layer_idx, f"{phase} epoch {epoch}")
        if _validation_is_better(validation, best_validation):
            best_validation = dict(validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in tmix.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if world == 1 or dist.get_rank() == 0:
            print(
                {
                    "layer": layer_idx,
                    "alignment_phase": phase,
                    "phase_epoch": epoch,
                    "train_attention_nmse": train_attention_nmse,
                    "train_tmix_output_nmse": train_tmix_nmse,
                    "train_layer_output_nmse": train_block_nmse,
                    "validation_tmix_output_nmse": validation["tmix_output_nmse"],
                    "validation_layer_output_nmse": validation["layer_output_nmse"],
                    "best_validation_layer_output_nmse": best_validation[
                        "layer_output_nmse"
                    ],
                    "epochs_without_block_improvement": epochs_without_improvement,
                    **{
                        f"sidecar_{name}": float(value)
                        for name, value in tmix._last_sidecar_metrics.items()
                    },
                },
                flush=True,
            )
        if epochs_without_improvement >= patience:
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
):
    tmix = student.model.layers[layer_idx].tmix
    initial_state = {
        name: value.detach().cpu().clone() for name, value in tmix.state_dict().items()
    }
    try:
        train_attention = _precompute_gqa_teacher_targets(
            source_text, layer_idx, train_hidden, device
        )
        attention_validation, attention_epoch = _run_gqa_phase(
            source_text,
            student,
            layer_idx,
            train_hidden,
            validation_hidden,
            world,
            device,
            phase="attention_transfer",
            train_attention=train_attention,
            epochs=16,
            patience=4,
            lr=1e-2,
            weight_decay=0.0,
        )
        del train_attention
        student.requires_grad_(False)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(0)
            tmix.enable_lora(rank=16, alpha=32.0)
        if world > 1:
            for parameter in tmix.lora_parameters():
                dist.broadcast(parameter.data, 0)
        correction_validation, correction_epoch = _run_gqa_phase(
            source_text,
            student,
            layer_idx,
            train_hidden,
            validation_hidden,
            world,
            device,
            phase="low_rank_correction",
            train_attention=None,
            epochs=48,
            patience=8,
            lr=3e-5,
            weight_decay=0.1,
        )
        tmix.merge_lora()
        reference = _evaluate_gqa_reference(
            source_text, student, layer_idx, validation_hidden, world, device
        )
        native = _evaluate_layer(
            source_text, student, layer_idx, validation_hidden, world, device
        )
        sidecar = {
            name: float(value)
            for name, value in tmix._last_sidecar_metrics.items()
        }
        _require_finite_metrics(native, layer_idx, "final GQA BF16 validation")
        if reference["reference_layer_output_nmse"] > ORACLE_BLOCK_NMSE_GATE:
            raise RuntimeError(
                "bounded Hedgehog/H2O reference complete-Block gate failed: "
                f"{reference['reference_layer_output_nmse']:.8g} > "
                f"{ORACLE_BLOCK_NMSE_GATE:.8g}"
            )
        if native["layer_output_nmse"] > LAYER_OUTPUT_NMSE_TARGET:
            raise RuntimeError(
                "bounded Hedgehog/H2O BF16 complete-Block gate failed: "
                f"{native['layer_output_nmse']:.8g} > {LAYER_OUTPUT_NMSE_TARGET:.8g}"
            )
        if reference["native_incremental_layer_output_nmse"] > 1e-3:
            raise RuntimeError(
                "FlashRWKV2 BF16 incremental complete-Block gate failed: "
                f"{reference['native_incremental_layer_output_nmse']:.8g} > 0.001"
            )
        fp16_cache = _evaluate_gqa_fp16_cache(
            source_text,
            student,
            layer_idx,
            validation_hidden,
            dist.get_rank() if world > 1 else 0,
            world,
            device,
        )
        return {
            "attention_transfer_best_epoch": attention_epoch,
            "attention_transfer_best_validation": attention_validation,
            "correction_best_epoch": correction_epoch,
            "correction_best_validation": correction_validation,
            "reference_validation": reference,
            "native_validation": native,
            "fp16_cache_validation": fp16_cache,
            "sidecar_validation": sidecar,
        }
    except Exception as error:
        if world == 1 or dist.get_rank() == 0:
            failure = {
                "layer": layer_idx,
                "stage": "bounded_hedgehog_alignment_failed",
                "error": repr(error),
                "artifact_saved": False,
                **{
                    f"sidecar_{name}": float(value)
                    for name, value in tmix._last_sidecar_metrics.items()
                },
            }
            for local_name in (
                "attention_validation",
                "correction_validation",
                "reference",
                "native",
            ):
                values = locals().get(local_name)
                if values is not None:
                    failure.update(
                        {
                            f"{local_name}_{name}": value
                            for name, value in values.items()
                        }
                    )
            print(failure, flush=True)
        tmix.drop_lora()
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
        reused_hidden = LastLayerCache(prefix_cache, rank).load()
        if reused_hidden.shape != (ids.shape[0], ids.shape[1], student.config.hidden_size):
            raise ValueError(
                "GQA prefix cache shape does not match the immutable packed rows: "
                f"cache={tuple(reused_hidden.shape)} ids={tuple(ids.shape)}"
            )
        cache.store(reused_hidden, "next")
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
        if student.config.source_layer_types[index] == "full_attention":
            _freeze_value_residual(tmix)
        layer = student.model.layers[index].to(device).train()
        is_gdn = student.config.source_layer_types[index] == "linear_attention"
        if not is_gdn:
            result = _align_gqa_layer(
                source_text,
                student,
                index,
                train_hidden,
                validation_hidden,
                world,
                device,
            )
            train_metrics = _evaluate_layer(
                source_text, student, index, train_hidden, world, device
            )
            validation_metrics = result["native_validation"]
            layer_strict_pass = (
                validation_metrics["layer_output_nmse"] <= LAYER_OUTPUT_NMSE_TARGET
            )
            if prefix_cache is None:
                prefix_strict_pass = prefix_strict_pass and layer_strict_pass
            artifact_pass = (
                layer_strict_pass if prefix_cache is not None else prefix_strict_pass
            )
            if rank == 0:
                print(
                    {
                        "layer": index,
                        "stage": "bounded_hedgehog_best_checkpoint",
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
                        **{
                            f"sidecar_{name}": value
                            for name, value in result["sidecar_validation"].items()
                        },
                        "attention_transfer_best_epoch": result[
                            "attention_transfer_best_epoch"
                        ],
                        "correction_best_epoch": result["correction_best_epoch"],
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
            objective = _LayerObjective(layer)
            chunks = []
            with torch.no_grad():
                for hidden in hidden_cache.split(8):
                    chunks.append(objective(hidden.to(device))[0].cpu())
            cache.store(torch.cat(chunks), "next")
            cache.advance()
            continue
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
                if hasattr(tmix, "value_residual_scale"):
                    tmix.value_residual_scale.data.zero_()
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


def _set_global_parameters(student):
    student.requires_grad_(False)
    ordinary, residual, scales = [], [], []
    seen_gqa = False
    for index, layer in enumerate(student.model.layers):
        tmix = layer.tmix
        tmix.requires_grad_(True)
        if student.config.source_layer_types[index] == "linear_attention":
            ordinary.extend(tmix.parameters())
            continue
        if not hasattr(tmix, "value_residual_scale"):
            ordinary.extend(tmix.parameters())
            seen_gqa = True
            continue
        value_parameters = (tmix.v0, tmix.v1, tmix.v2)
        if seen_gqa:
            residual.extend(value_parameters)
        scales.append(tmix.value_residual_scale)
        if not seen_gqa:
            for parameter in value_parameters:
                parameter.requires_grad_(False)
            tmix.value_residual_scale.requires_grad_(False)
            tmix.value_residual_scale.data.zero_()
        excluded = {id(p) for p in (*value_parameters, tmix.value_residual_scale)}
        ordinary.extend(p for p in tmix.parameters() if id(p) not in excluded)
        seen_gqa = True
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
            first_gqa = next(
                layer.tmix
                for index, layer in enumerate(student.model.layers)
                if student.config.source_layer_types[index] == "full_attention"
            )
            if hasattr(first_gqa, "value_residual_scale"):
                first_gqa.value_residual_scale.data.zero_()
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
    scales = [
        float(layer.tmix.value_residual_scale)
        for layer in model.model.layers
        if hasattr(layer.tmix, "value_residual_scale")
    ]
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


def convert_gqa_from_prefix_cache(source: str, output: str, prefix_cache: str) -> None:
    """Run only layer-3 GQA work from a verified post-layer-2 rank-local cache."""
    rank, world, device = _distributed()
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
    )
    if world > 1:
        dist.barrier()
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
    if args.gqa_prefix_cache is not None:
        convert_gqa_from_prefix_cache(args.source, args.output, args.gqa_prefix_cache)
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
