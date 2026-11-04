import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from any2rwkv.qwen2rwkv.qwen3_5.align import primed_gqa as alignment
from any2rwkv.qwen2rwkv.qwen3_5.gqa_priming import (
    W_SCALE,
    AdaptiveHeads,
    FrozenGDN,
    PrimedQwen,
    PrimedState,
    PrimedTimeMix,
    reference_wkv,
    run_wkv,
    source_outputs,
)


@pytest.fixture
def source():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"] * 3,
        attention_dropout=0.0,
        pad_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "eager"
    return Qwen3_5ForCausalLM(config).eval().requires_grad_(False)


def test_recurrence_matches_explicit_dplr_and_gradients():
    torch.manual_seed(31)
    inputs = [torch.randn(2, 7, 2, 4).mul(0.1).requires_grad_() for _ in range(6)]
    actual, actual_state = reference_wkv(*inputs)
    state, outputs = torch.zeros(2, 2, 4, 4), []
    for index in range(7):
        read, decay, key, value, left, right = [tensor[:, index] for tensor in inputs]
        transition = torch.diag_embed((W_SCALE * decay.sigmoid()).exp())
        transition = transition + right.unsqueeze(-1) * left.unsqueeze(-2)
        state = transition @ state + key.unsqueeze(-1) * value.unsqueeze(-2)
        outputs.append((read.unsqueeze(-2) @ state).squeeze(-2))
    expected = torch.stack(outputs, 1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, state)
    actual_gradients = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_agqa_starts_as_exact_repeat_and_then_learns():
    adapter = AdaptiveHeads(2, 8, 4, 3)
    hidden = torch.randn(2, 5, 2, 4)
    torch.testing.assert_close(adapter(hidden), hidden.repeat_interleave(4, 2), rtol=0, atol=0)
    adapter(hidden).square().sum().backward()
    assert adapter.up.weight.grad.abs().sum() > 0
    assert adapter.repeat.grad.abs().sum() > 0
    assert torch.equal(adapter.down.weight.grad, torch.zeros_like(adapter.down.weight))


def test_source_outputs_matches_native_block(source):
    hidden = torch.randn(2, 7, 16)
    actual, _ = source_outputs(source.model.layers[0], source.model.rotary_emb, hidden)
    positions = torch.arange(7).expand(2, -1)
    mask = torch.full((7, 7), torch.finfo(hidden.dtype).min).triu(1)[None, None]
    expected = source.model.layers[0](
        hidden,
        position_embeddings=source.model.rotary_emb(hidden, positions),
        attention_mask=mask,
        past_key_values=None,
    )
    torch.testing.assert_close(actual, expected)


def test_source_copy_and_initial_control_contract(source):
    mixer = PrimedTimeMix(
        source.model.layers[0].self_attn, source.config, rank=4, backend="reference"
    )
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        actual, expected = getattr(mixer, name), getattr(source.model.layers[0].self_attn, name)
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
        assert actual.weight.data_ptr() != expected.weight.data_ptr()
    assert all(torch.count_nonzero(shift) == 0 for shift in mixer.shifts.values())
    assert torch.all(mixer.decay.bias == -6)
    assert torch.all(mixer.erase.bias == -2)
    assert torch.count_nonzero(mixer.r_k) == 0
    assert not any("feature" in name or "teacher" in name for name in mixer.state_dict())


def test_nontrivial_mixer_cache_chunks_match_full_and_stay_fixed(source):
    mixer = PrimedTimeMix(
        source.model.layers[0].self_attn, source.config, rank=4, backend="reference"
    )
    with torch.no_grad():
        for shift in mixer.shifts.values():
            shift.uniform_(-0.2, 0.2)
        mixer.r_k.uniform_(-0.1, 0.1)
        mixer.key_heads.up.weight.normal_(std=0.01)
        mixer.value_heads.up.weight.normal_(std=0.01)
        mixer.value_mix.up.weight.normal_(std=0.01)
        hidden, first = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        positions = torch.arange(9).expand(2, -1)
        expected, _ = mixer(hidden, positions, first)
        cache, chunks, sizes = PrimedState(), [], []
        for start, end in ((0, 3), (3, 4), (4, 9)):
            output, _ = mixer(
                hidden[:, start:end], positions[:, start:end], first[:, start:end], cache
            )
            chunks.append(output)
            sizes.append(cache.matrix.numel() + cache.previous.numel())
        torch.testing.assert_close(torch.cat(chunks, 1), expected, atol=1e-6, rtol=1e-5)
        assert cache.length == 9 and len(set(sizes)) == 1
        assert cache.matrix.dtype == torch.float32
        with pytest.raises(ValueError, match="positions"):
            mixer(hidden[:, :1], positions[:, :1], cache=cache)
    with pytest.raises(ValueError, match="no_grad"):
        mixer(hidden[:, :1], positions[:, :1], cache=PrimedState())


def test_first_gqa_values_cross_retained_layer_and_keep_gradients(source):
    student = PrimedQwen(source, selected=[0, 2], rank=4, backend="reference")
    student.layers[0].mixer.set_trainable(True)
    student.layers[2].mixer.set_trainable(True)
    with torch.no_grad():
        student.layers[2].mixer.value_mix.bias.fill_(0.3)
    hidden = student.embed_tokens(torch.tensor([[1, 2, 3, 4]]))
    hidden, _, first = student.step_layer(0, hidden)
    first.retain_grad()
    hidden, _, carried = student.step_layer(1, hidden, first)
    assert carried is first
    result, _, _ = student.step_layer(2, hidden, carried)
    result.square().sum().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert student.layers[0].mixer.v_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in source.parameters())


def test_no_agqa_stays_frozen_when_current_mixer_unfrozen(source):
    mixer = PrimedTimeMix(
        source.model.layers[0].self_attn, source.config, rank=4, agqa=False, backend="reference"
    )
    mixer.set_trainable(True)
    assert mixer.q_proj.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in mixer.key_heads.parameters())
    assert not any(parameter.requires_grad for parameter in mixer.value_heads.parameters())


@pytest.mark.parametrize(
    "change",
    [
        {"head_dim": 7},
        {"num_key_value_heads": 3},
        {"attention_dropout": 0.1},
        {"rope_parameters": {"rope_type": "linear", "rope_theta": 10000}},
    ],
)
def test_unsupported_geometry_is_rejected(source, change):
    config = deepcopy(source.config)
    for name, value in change.items():
        setattr(config, name, value)
    with pytest.raises(ValueError):
        PrimedTimeMix(source.model.layers[0].self_attn, config, rank=4, backend="reference")


def test_flash_never_silently_falls_back_to_reference(source):
    mixer = PrimedTimeMix(source.model.layers[0].self_attn, source.config, rank=4)
    with pytest.raises(ValueError, match="CUDA BF16"):
        mixer(torch.randn(1, 16, 16), torch.arange(16)[None])


def test_gate_hit_exploration_and_hard_epoch_cap():
    options = alignment.AlignmentOptions()
    assert alignment.should_train(99, None, options)
    assert not alignment.should_train(100, None, options)
    assert alignment.should_train(24, 5, options)
    assert not alignment.should_train(25, 5, options)
    assert alignment.should_train(99, 95, options)
    assert not alignment.should_train(100, 95, options)
    assert not alignment.should_train(20, 0, options)


def packed():
    generator = torch.Generator().manual_seed(9)
    return torch.randint(1, 41, (4, 8), generator=generator), torch.randint(
        1, 41, (2, 8), generator=generator
    )


def tiny_options():
    return alignment.AlignmentOptions(
        epochs=2, extra_epochs=1, target_nmse=1e-12, batch_size=2, eval_batch_size=2, group_size=2
    )


def test_alignment_freezes_backbone_saves_and_loads_exactly(source, tmp_path):
    training, validation = packed()
    before = alignment.state_hash(source)
    student, reports = alignment.align(
        source,
        training,
        validation,
        tmp_path / "run",
        selected=[0, 2],
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    assert alignment.state_hash(source) == before
    assert [report["layer"] for report in reports] == [0, 2]
    assert all(report["epochs"] == 2 and report["frozen_weights_unchanged"] for report in reports)
    assert not any(parameter.requires_grad for parameter in student.parameters())
    loaded = alignment.load_aligned(source, tmp_path / "run", backend="reference")
    with torch.no_grad():
        torch.testing.assert_close(loaded(validation), student(validation), rtol=0, atol=0)
    resumed, reused_reports = alignment.align(
        source,
        training,
        validation,
        tmp_path / "run",
        selected=[0, 2],
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    assert reports == reused_reports
    assert alignment.state_hash(resumed) == alignment.state_hash(student)
    with torch.no_grad():
        source.model.layers[1].mlp.down_proj.weight.add_(0.1)
    with pytest.raises(ValueError, match="parent mismatch"):
        alignment.load_aligned(source, tmp_path / "run", backend="reference")


def test_epoch_resume_matches_uninterrupted_training(source, tmp_path, monkeypatch):
    training, validation = packed()
    baseline, history = alignment.align(
        source,
        training,
        validation,
        tmp_path / "baseline",
        selected=[0, 2],
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    save = alignment.atomic_torch

    def interrupt_after_save(path, payload):
        save(path, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(alignment, "atomic_torch", interrupt_after_save)
    with pytest.raises(KeyboardInterrupt):
        alignment.align(
            source,
            training,
            validation,
            tmp_path / "interrupted",
            selected=[0, 2],
            rank=4,
            backend="reference",
            options=tiny_options(),
        )
    monkeypatch.setattr(alignment, "atomic_torch", save)
    resumed, resumed_history = alignment.align(
        source,
        training,
        validation,
        tmp_path / "interrupted",
        selected=[0, 2],
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    assert resumed_history == history
    assert alignment.state_hash(resumed) == alignment.state_hash(baseline)


def test_changed_data_and_corrupt_checkpoint_are_rejected(source, tmp_path):
    training, validation = packed()
    alignment.align(
        source,
        training,
        validation,
        tmp_path / "run",
        selected=[0],
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    with pytest.raises(ValueError, match="different experiment"):
        alignment.align(
            source,
            training.flip(0),
            validation,
            tmp_path / "run",
            selected=[0],
            rank=4,
            backend="reference",
            options=tiny_options(),
        )
    with (tmp_path / "run/layer_00/best.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        alignment.load_aligned(source, tmp_path / "run", backend="reference")


def test_dataset_overlap_is_rejected():
    training, _ = packed()
    with pytest.raises(ValueError, match="overlap"):
        alignment.validate_tokens(training, training[:1], 41, 2)


def test_native_gdn_layers_are_retained_and_frozen(source, tmp_path):
    config = deepcopy(source.config)
    config.num_hidden_layers = 4
    config.layer_types = [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
    ]
    config.linear_num_key_heads = config.linear_num_value_heads = 2
    config.linear_key_head_dim = config.linear_value_head_dim = 8
    config.linear_conv_kernel_dim = 4
    hybrid = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    training, validation = packed()
    original = alignment.state_hash(hybrid)
    student, reports = alignment.align(
        hybrid,
        training,
        validation,
        tmp_path / "hybrid",
        rank=4,
        backend="reference",
        options=tiny_options(),
    )
    assert student.selected == [2, 3]
    assert student.layers[0] is hybrid.model.layers[0]
    assert student.layers[1] is hybrid.model.layers[1]
    assert alignment.state_hash(hybrid) == original
    assert all(report["frozen_weights_unchanged"] for report in reports)
    with torch.no_grad():
        assert torch.isfinite(student(validation)).all()


def test_output_has_one_writer_and_releases_after_failure(tmp_path):
    output = tmp_path / "run"
    with alignment.output_lock(output):
        with pytest.raises(RuntimeError, match="another writer"):
            with alignment.output_lock(output):
                pytest.fail("second writer acquired lock")
    with alignment.output_lock(output):
        pass


def test_default_geometry_uses_one_d256_state_per_query_head():
    config = Qwen3_5TextConfig(
        hidden_size=2048,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        layer_types=["full_attention"],
        vocab_size=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    )
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
        mixer = PrimedTimeMix(model.model.layers[0].self_attn, config)
    assert (mixer.heads, mixer.kv_heads, mixer.width) == (8, 2, 256)
    assert sum(parameter.numel() for parameter in mixer.parameters()) == 15266336
    assert mixer.heads * mixer.width * mixer.width * 4 == 2 * 1024**2


def test_converted_gdn_adapter_retains_author_shell_and_never_sets_first(source):
    try:
        from any2rwkv.qwen2rwkv.qwen3_5.transformers.modeling_qwen2rwkv import (
            Qwen2RWKVGatedDeltaNet,
        )
    except ImportError as error:
        pytest.skip(
            f"author GDN adapter needs the pinned Transformers RWKV implementation: {error}"
        )
    config = deepcopy(source.config)
    config.layer_types = ["linear_attention"] * 3
    config.linear_num_key_heads = config.linear_num_value_heads = 2
    config.linear_key_head_dim = config.linear_value_head_dim = 8
    config.linear_conv_kernel_dim = 4
    native = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    shell = Qwen2RWKVGatedDeltaNet(config, 0).float()
    shell.load_state_dict(native.model.layers[0].linear_attn.state_dict(), strict=True)
    frozen = FrozenGDN(shell, backend="reference")
    hidden = torch.randn(2, 8, 16)
    before = alignment.state_hash(shell)
    with torch.no_grad():
        query, key, value, beta, decay, gate = shell._source_activations(hidden, None, None)
        inputs = shell._wkv_inputs(query, key, value, beta, decay, training=False)
        raw, _ = reference_wkv(*(tensor.view_as(value) for tensor in inputs))
        expected = shell._source_boundary(raw, gate)
        actual, first = frozen(hidden, torch.arange(8).expand(2, -1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert first is None
        carried = torch.randn_like(hidden)
        assert frozen(hidden, torch.arange(8).expand(2, -1), carried)[1] is carried
    assert alignment.state_hash(shell) == before
    assert not any(parameter.requires_grad for parameter in frozen.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashRWKV2 requires CUDA")
@pytest.mark.parametrize("width", [128, 256])
def test_cuda_flash_bf16_forward_backward_and_cached_chunks(width):
    flash = pytest.importorskip("flashrwkv2")
    if flash.__version__ != "0.1.0a17":
        pytest.skip("priming uses FlashRWKV2 0.1.0a17; default runtime uses a different pin")
    torch.manual_seed(41)
    inputs = [
        (torch.randn(1, 16, 2, width, device="cuda") * 0.02).bfloat16().requires_grad_()
        for _ in range(6)
    ]
    actual = run_wkv(inputs, "flash")
    expected, _ = reference_wkv(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)
    actual_gradients = torch.autograd.grad(actual.float().square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.float().square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=4e-2)
    with torch.no_grad():
        state = PrimedState()
        left = run_wkv([tensor[:, :7] for tensor in inputs], "flash", state)
        right = run_wkv([tensor[:, 7:] for tensor in inputs], "flash", state)
        torch.testing.assert_close(torch.cat((left, right), 1), actual, atol=2e-4, rtol=2e-2)
        assert state.matrix.dtype == torch.float32
        assert state.matrix.shape == (1, 2, width, width)


def test_local_cli_logs_and_reuses_completed_layers(source, tmp_path):
    model_path = tmp_path / "source"
    source.save_pretrained(model_path)
    training, validation = packed()
    token_path = tmp_path / "tokens.pt"
    torch.save({"training": training, "validation": validation}, token_path)
    output = tmp_path / "aligned"
    command = [
        sys.executable,
        "-u",
        "-m",
        "any2rwkv.qwen2rwkv.qwen3_5.align.primed_gqa",
        "--source",
        str(model_path),
        "--packed",
        str(token_path),
        "--output",
        str(output),
        "--layers",
        "0,2",
        "--rank",
        "4",
        "--backend",
        "reference",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--group-size",
        "2",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1] / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    environment["ANY2RWKV_LOG_DIR"] = str(tmp_path / "logs")
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "=== START" in result.stdout and "status=0" in result.stdout
    logs = list((tmp_path / "logs").glob("primed_gqa_*_p*.log"))
    assert len(logs) == 2
    assert (output / "complete.json").is_file()
    assert all("cwd=" in path.read_text(encoding="utf-8") for path in logs)
