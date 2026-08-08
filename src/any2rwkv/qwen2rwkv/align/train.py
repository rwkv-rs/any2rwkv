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


def _layerwise(source_text, student, ids, output, rank, world, device):
    cache = LastLayerCache(output / "cache", rank)
    completed = _completed_layers(output)
    for index in range(completed):
        student.model.layers[index].tmix.load_state_dict(
            load_file((output / f"layer_{index:02d}.safetensors").as_posix())
        )
    if not cache.path("current").is_file():
        _rebuild_cache(student, ids, cache, completed, device)

    for index in range(completed, 24):
        hidden_cache = cache.load()
        calibration = hidden_cache[:8].to(device)
        metrics = _initialize_layer(source_text, student, index, calibration)
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
        optimizer = torch.optim.AdamW(
            [p for p in wrapper.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.1,
        )
        loader = DataLoader(TensorDataset(hidden_cache), batch_size=8, shuffle=False)
        scheduler = _schedule(optimizer, 12 * len(loader))
        history = []
        for epoch in range(12):
            total = torch.zeros((), device=device)
            for (hidden,) in loader:
                hidden = hidden.to(device)
                with torch.no_grad():
                    wanted = _teacher_layer(source_text, index, hidden)
                actual = wrapper(hidden)
                loss = (actual.float() - wanted.float()).square().mean() / (
                    wanted.float().square().mean() + 1e-6
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite layer {index} loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tmix.value_residual_scale.data.zero_()
                total += loss.detach() / len(loader)
            average = _mean(total, world)
            history.append(average)
            if epoch >= 2 and all(
                (history[j - 1] - history[j]) / max(abs(history[j - 1]), 1e-12) < 0.005
                for j in (len(history) - 2, len(history) - 1)
            ):
                break
        if rank == 0:
            _save_layer(output, index, tmix)
            print({"layer": index, "nmse": history[-1], **metrics}, flush=True)
        if world > 1:
            dist.barrier()
        layer.train()
        chunks = []
        with torch.no_grad():
            for (hidden,) in loader:
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


def convert_qwen3_5_2b(source: str, output: str, agentic: str, math_dataset: str):
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
    _layerwise(source_text, student, local_ids, output_path, rank, world, device)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--agentic", default="nvidia/Nemotron-SFT-Agentic-v2")
    parser.add_argument("--math", dest="math_dataset", default="nvidia/Nemotron-SFT-Math-v4")
    parser.add_argument("--accept-only", action="store_true")
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
    convert_qwen3_5_2b(args.source, args.output, args.agentic, args.math_dataset)


if __name__ == "__main__":
    main()
