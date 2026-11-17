"""Greedy GQA-only priming using local, pre-tokenized training/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import sys
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..gqa_priming import PRIMED_GQA_SCHEMA, PrimedQwen, source_outputs


@dataclass(frozen=True)
class AlignmentOptions:
    epochs: int = 100
    extra_epochs: int = 20
    target_nmse: float = 0.0025
    learning_rate: float = 1e-4
    copied_lr_scale: float = 0.1
    batch_size: int = 16
    eval_batch_size: int = 16
    group_size: int = 20
    seed: int = 42

    def validate(self):
        if min(self.epochs, self.batch_size, self.eval_batch_size, self.group_size) <= 0:
            raise ValueError("epochs, batch sizes and rotation group size must be positive")
        if self.extra_epochs < 0 or any(
            not math.isfinite(value) or value <= 0
            for value in (self.target_nmse, self.learning_rate, self.copied_lr_scale)
        ):
            raise ValueError("invalid exploration budget, target or learning rate")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(module, frozen_only=False):
    parameters = dict(module.named_parameters())
    return fingerprint(
        {
            name: tensor_hash(value)
            for name, value in module.state_dict().items()
            if not frozen_only or name not in parameters or not parameters[name].requires_grad
        }
    )


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def atomic_torch(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, pending)
    os.replace(pending, path)


@contextmanager
def output_lock(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another writer owns the priming output") from error
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another writer owns the priming output") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def errors(actual, target):
    return (actual.float() - target.float()).square().sum(), target.float().square().sum()


def metrics(total):
    block = total[0] / max(total[1], 1e-12)
    mixer = total[2] / max(total[3], 1e-12)
    if not all(math.isfinite(value) and value >= 0 for value in (block, mixer)):
        raise FloatingPointError("nonfinite validation NMSE")
    return {"block_nmse": block, "mixer_nmse": mixer, "loss": 0.5 * (block + mixer)}


def local_loss(actual, mixed, target, target_mixed):
    block_error, block_energy = errors(actual, target)
    mix_error, mix_energy = errors(mixed, target_mixed)
    return 0.5 * (
        block_error / block_energy.clamp_min(1e-12) + mix_error / mix_energy.clamp_min(1e-12)
    )


def should_train(epoch, first_hit, options):
    return epoch < options.epochs and (
        first_hit is None or epoch < first_hit + options.extra_epochs
    )


@torch.no_grad()
def measure(student, source, index, hidden, first, batch_size, device):
    total = [0.0] * 4
    student.eval()
    for start in range(0, len(hidden), batch_size):
        part = hidden[start : start + batch_size].to(device)
        first_part = None if first is None else first[start : start + batch_size].to(device)
        with autocast(device):
            target, target_mixed = source_outputs(
                source.model.layers[index], source.model.rotary_emb, part
            )
            actual, mixed, _ = student.step_layer(index, part, first_part)
        for column, value in enumerate((*errors(actual, target), *errors(mixed, target_mixed))):
            total[column] += float(value)
    return metrics(total)


@torch.no_grad()
def advance(student, index, hidden, first, batch_size, device):
    outputs, values = [], []
    for start in range(0, len(hidden), batch_size):
        part = hidden[start : start + batch_size].to(device)
        first_part = None if first is None else first[start : start + batch_size].to(device)
        with autocast(device):
            output, _, value = student.step_layer(index, part, first_part)
        outputs.append(output.detach().cpu())
        if value is not None:
            values.append(value.detach().cpu())
    return torch.cat(outputs), torch.cat(values) if values else None


def validate_tokens(training, validation, vocabulary, group_size):
    for tokens in (training, validation):
        if tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape) <= 0:
            raise ValueError("packed splits must be nonempty LongTensor [rows, context]")
        if int(tokens.min()) < 0 or int(tokens.max()) >= vocabulary:
            raise ValueError("packed token outside the source vocabulary")
    if training.shape[1] != validation.shape[1] or len(training) % group_size:
        raise ValueError("splits need equal context and complete training rotation groups")
    hashes = {tensor_hash(row) for row in training}
    if len(hashes) != len(training) or len({tensor_hash(row) for row in validation}) != len(
        validation
    ):
        raise ValueError("duplicate packed rows within a split")
    if hashes.intersection(tensor_hash(row) for row in validation):
        raise ValueError("training and validation packed rows overlap")


def load_layer(directory, identity, mixer):
    report = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if (
        report["identity"] != identity
        or file_hash(directory / "best.safetensors") != report["sha256"]
    ):
        raise ValueError("completed priming layer identity/checksum mismatch")
    mixer.load_state_dict(load_file(directory / "best.safetensors"), strict=True)
    mixer.set_trainable(False)
    return report["report"]


def train_layer(
    student, source, index, hidden, first, validation_rows, directory, identity, options
):
    directory.mkdir(parents=True, exist_ok=True)
    mixer = student.layers[index].mixer
    if (directory / "complete.json").exists():
        return load_layer(directory, identity, mixer)
    student.requires_grad_(False)
    mixer.set_trainable(True)
    frozen = state_hash(student, frozen_only=True)
    original = state_hash(source)
    device = next(mixer.parameters()).device
    copied = ("q_proj.", "k_proj.", "v_proj.", "o_proj.", "q_norm.", "k_norm.")
    groups = [[], []]
    for name, parameter in mixer.named_parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.float32:
                raise ValueError("priming requires FP32 master weights")
            groups[int(name.startswith(copied))].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups[0], "lr": options.learning_rate},
            {"params": groups[1], "lr": options.learning_rate * options.copied_lr_scale},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    rows = len(hidden) - validation_rows
    steps = math.ceil(rows / options.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.45 * (1 + math.cos(math.pi * min(step / (steps * options.epochs), 1))),
    )
    progress = directory / "progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        if (
            saved["identity"] != identity
            or saved["frozen"] != frozen
            or saved["original"] != original
        ):
            raise ValueError("priming resume identity/frozen parent mismatch")
        mixer.load_state_dict(saved["current"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, history, best_state, first_hit = (
            saved[key] for key in ("epoch", "history", "best_state", "first_hit")
        )
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        if epoch > options.epochs or [row["epoch"] for row in history] != list(range(epoch + 1)):
            raise ValueError("invalid priming resume history")
    else:
        epoch = 0
        measured = measure(
            student,
            source,
            index,
            hidden[:validation_rows],
            None if first is None else first[:validation_rows],
            options.eval_batch_size,
            device,
        )
        history, best_state = [{"epoch": 0, **measured}], cpu_state(mixer)
        first_hit = (
            0
            if max(measured["block_nmse"], measured["mixer_nmse"]) <= options.target_nmse
            else None
        )
    best = min(history, key=lambda row: row["loss"])
    print(
        json.dumps({"stage": "layer_begin", "layer": index, "epoch": epoch, "best": best}),
        flush=True,
    )
    while should_train(epoch, first_hit, options):
        generator = torch.Generator().manual_seed(options.seed + 10000 * index + epoch)
        order = (
            torch.randperm(rows // options.group_size, generator=generator)[:, None]
            * options.group_size
            + torch.arange(options.group_size)
        ).flatten() + validation_rows
        for indices in order.split(options.batch_size):
            part = hidden[indices].to(device)
            first_part = None if first is None else first[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), autocast(device):
                target, target_mixed = source_outputs(
                    source.model.layers[index], source.model.rotary_emb, part
                )
            with autocast(device):
                actual, mixed, _ = student.step_layer(index, part, first_part)
                loss = local_loss(actual, mixed, target, target_mixed)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite priming training loss")
            loss.backward()
            parameters = [parameter for parameter in mixer.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        epoch += 1
        measured = measure(
            student,
            source,
            index,
            hidden[:validation_rows],
            None if first is None else first[:validation_rows],
            options.eval_batch_size,
            device,
        )
        history.append({"epoch": epoch, **measured})
        if measured["loss"] < best["loss"]:
            best, best_state = history[-1], cpu_state(mixer)
        if (
            first_hit is None
            and max(measured["block_nmse"], measured["mixer_nmse"]) <= options.target_nmse
        ):
            first_hit = epoch
        if state_hash(student, frozen_only=True) != frozen or state_hash(source) != original:
            raise ValueError("frozen FFN, other layer or teacher weights changed")
        atomic_torch(
            progress,
            {
                "identity": identity,
                "frozen": frozen,
                "original": original,
                "epoch": epoch,
                "current": cpu_state(mixer),
                "best_state": best_state,
                "history": history,
                "first_hit": first_hit,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
        )
        print(
            json.dumps(
                {"stage": "layer_epoch", "layer": index, **history[-1], "best_epoch": best["epoch"]}
            ),
            flush=True,
        )
    mixer.load_state_dict(best_state, strict=True)
    mixer.set_trainable(False)
    report = {
        "layer": index,
        "epochs": epoch,
        "best": best,
        "first_hit": first_hit,
        "target_met": max(best["block_nmse"], best["mixer_nmse"]) <= options.target_nmse,
        "frozen_weights_unchanged": True,
        "history": history,
    }
    pending = directory / "best.safetensors.tmp"
    save_file(best_state, pending)
    os.replace(pending, directory / "best.safetensors")
    write_json(
        directory / "complete.json",
        {
            "identity": identity,
            "sha256": file_hash(directory / "best.safetensors"),
            "report": report,
        },
    )
    return report


def _align(
    source,
    training,
    validation,
    output,
    *,
    selected=None,
    rank=32,
    agqa=True,
    backend="flash",
    retained_mixers=None,
    options=None,
):
    """Train ascending GQA layers only; freeze each best before advancing the prefix.

    Both local targets use the student prefix. This is layerwise alignment, not
    the paper's independent-stream, end-to-end final-hidden MSE objective.
    """
    options = options or AlignmentOptions()
    options.validate()
    validate_tokens(training, validation, source.config.vocab_size, options.group_size)
    device = next(source.parameters()).device
    if backend == "flash" and (device.type != "cuda" or training.shape[1] % 16):
        raise ValueError("Flash alignment needs a CUDA source and context divisible by 16")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this layerwise entry is single-process; do not launch with torchrun")
    source.eval().requires_grad_(False)
    torch.manual_seed(options.seed)
    student = PrimedQwen(source, selected, rank, agqa, backend, retained_mixers)
    manifest = {
        "schema": PRIMED_GQA_SCHEMA,
        "source_sha256": state_hash(source),
        "source_config": json.loads(json.dumps(source.config.to_dict())),
        "selected": student.selected,
        "rank": rank,
        "agqa": agqa,
        "backend": backend,
        "options": asdict(options),
        "training": tensor_hash(training),
        "validation": tensor_hash(validation),
        "retained_mixers": {
            str(index): state_hash(module) for index, module in (retained_mixers or {}).items()
        },
        "torch": str(torch.__version__),
        "device_type": device.type,
        "implementation": file_hash(Path(__file__)),
        "mixer_implementation": file_hash(Path(__file__).parents[1] / "gqa_priming.py"),
    }
    identity = fingerprint(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "output belongs to a different experiment; preserve it and choose a new directory"
            )
    else:
        if any(output.iterdir()):
            raise ValueError("refusing to adopt a nonempty output without an experiment identity")
        write_json(manifest_path, manifest)
    tokens = torch.cat((validation.cpu(), training.cpu()))
    with torch.no_grad():
        hidden = torch.cat(
            [
                source.model.embed_tokens(part.to(device)).float().cpu()
                for part in tokens.split(options.eval_batch_size)
            ]
        )
    first, reports = None, []
    for index in range(max(student.selected) + 1):
        if index in student.selected:
            reports.append(
                train_layer(
                    student,
                    source,
                    index,
                    hidden,
                    first,
                    len(validation),
                    output / f"layer_{index:02d}",
                    identity,
                    options,
                )
            )
        if index < max(student.selected):
            hidden, first = advance(student, index, hidden, first, options.eval_batch_size, device)
    if state_hash(source) != manifest["source_sha256"]:
        raise ValueError("source backbone changed during priming")
    write_json(
        output / "complete.json",
        {"identity": identity, "layers": reports, "global_training": False},
    )
    return student, reports


def align(source, training, validation, output, **kwargs):
    """Align ascending GQA layers under an exclusive, process-lifetime output lock."""
    with output_lock(output):
        return _align(source, training, validation, output, **kwargs)


def load_aligned(source, directory, *, retained_mixers=None, backend="flash"):
    """Strictly assemble GQA deltas on the exact original/retained frozen parent."""
    directory = Path(directory)
    manifest = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    retained = {str(index): state_hash(module) for index, module in (retained_mixers or {}).items()}
    if (
        manifest["schema"] != PRIMED_GQA_SCHEMA
        or manifest["source_sha256"] != state_hash(source)
        or manifest["source_config"] != json.loads(json.dumps(source.config.to_dict()))
        or manifest["retained_mixers"] != retained
    ):
        raise ValueError("priming schema, source config/weights or retained parent mismatch")
    if not (directory / "complete.json").is_file():
        raise ValueError("priming run is not complete")
    identity = fingerprint(manifest)
    if json.loads((directory / "complete.json").read_text())["identity"] != identity:
        raise ValueError("completion identity mismatch")
    source.eval().requires_grad_(False)
    model = PrimedQwen(
        source, manifest["selected"], manifest["rank"], manifest["agqa"], backend, retained_mixers
    )
    for index in model.selected:
        load_layer(directory / f"layer_{index:02d}", identity, model.layers[index].mixer)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--packed",
        type=Path,
        required=True,
        help="local torch file containing disjoint training/validation LongTensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="3,7,11,15,19,23")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--no-agqa", action="store_true")
    parser.add_argument("--backend", choices=("flash", "reference"), default="flash")
    parser.add_argument("--device", default="cuda")
    for name, field in AlignmentOptions.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(field.default), default=field.default
        )
    args = parser.parse_args()
    if not args.source.is_dir() or args.source.resolve() == args.output.resolve():
        parser.error("source must be a local model directory, distinct from output")
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    config = getattr(config, "text_config", config)
    source = Qwen3_5ForCausalLM.from_pretrained(
        args.source,
        config=config,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        key_mapping={r"^model\.language_model\.": "model."},
    ).to(args.device)
    packed = torch.load(args.packed, map_location="cpu", weights_only=True)
    options = AlignmentOptions(
        **{name: getattr(args, name) for name in AlignmentOptions.__dataclass_fields__}
    )
    align(
        source,
        packed["training"],
        packed["validation"],
        args.output,
        selected=[int(index) for index in args.layers.split(",")],
        rank=args.rank,
        agqa=not args.no_agqa,
        backend=args.backend,
        options=options,
    )


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def logged_main():
    directory = Path(os.environ.get("ANY2RWKV_LOG_DIR", "testlog"))
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"primed_gqa_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}_p{os.getpid()}.log"
    )
    stdout, stderr = sys.stdout, sys.stderr
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(number, frame):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, terminate)
    with path.open("x", encoding="utf-8", buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        status = 0
        try:
            print(f"=== START {datetime.now().astimezone().isoformat()} ===", flush=True)
            print(
                f"host={socket.gethostname()} pid={os.getpid()} "
                f"cwd={Path.cwd()} log={path.resolve()}",
                flush=True,
            )
            main()
        except SystemExit as error:
            status = error.code if isinstance(error.code, int) else 1
        except KeyboardInterrupt:
            status = 130
        except Exception:
            status = 1
            traceback.print_exc()
        finally:
            print(
                f"=== END {datetime.now().astimezone().isoformat()} status={status} ===", flush=True
            )
            sys.stdout, sys.stderr = stdout, stderr
            signal.signal(signal.SIGTERM, previous_handler)
    return status


if __name__ == "__main__":
    raise SystemExit(logged_main())
