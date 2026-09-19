from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from any2rwkv.qwen2rwkv.qwen3_5.align import train
from any2rwkv.qwen2rwkv.qwen3_5.align.train import (
    GQA_GATE_SCHEDULE,
    _fp16_forward_mode,
    _fresh_process_gqa_strict_load,
    _gqa_fp16_soak,
    _LayerObjective,
    _mix_teacher,
    _nmse_loss,
    _refresh_gqa_stage_cache,
    _require_gqa_acceptance,
)
from any2rwkv.qwen2rwkv.qwen3_5.gqa2rwkv import _source_qkv, evaluate_gqa_recall
from any2rwkv.qwen2rwkv.qwen3_5.transformers.modeling_qwen2rwkv import (
    Qwen2RWKVConfig,
    Qwen2RWKVDecoderLayer,
    Qwen2RWKVForCausalLM,
    Qwen2RWKVTimeMix,
    _FP32RotaryEmbedding,
)


def _config() -> Qwen2RWKVConfig:
    return Qwen2RWKVConfig(
        vocab_size=32,
        hidden_size=2048,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        source_layer_types=["full_attention"],
        layer_types=["full_attention"],
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
        attention_bias=False,
    )


def test_gqa_reference_is_explicit_numerator_denominator_recurrence() -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    generator = torch.Generator().manual_seed(0)
    feature_query = torch.rand(1, 8, 7, 128, generator=generator)
    feature_key = torch.rand(1, 8, 7, 128, generator=generator)
    value_heads = torch.rand(1, 8, 7, 256, generator=generator)
    numerator, denominator = module._state_reference(feature_query, feature_key, value_heads)

    causal_weights = (feature_query @ feature_key.transpose(-1, -2)).tril()
    assert torch.allclose(numerator, causal_weights @ value_heads)
    assert torch.allclose(denominator, causal_weights.sum(-1))
    assert numerator.dtype == denominator.dtype == torch.float32


def test_gqa_feature_state_geometry_and_readout_contract() -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    module.reset_parameters()
    assert module.feature_q_weight.shape == (8, 256, 64)
    assert module.feature_k_weight.shape == (8, 256, 64)
    assert module.kernel_heads == 16
    assert module.recurrent_width == 16 * 256
    assert module.feature_q_weight.dtype == torch.float32
    identity = torch.eye(64)
    assert torch.equal(module.feature_q_weight[0, :64], identity)
    query = torch.randn(2, 8, 3, 256)
    key = torch.randn(2, 2, 3, 256)
    query_feature, key_feature = module._features(query, key)
    assert query_feature.shape == (2, 8, 3, 128)
    assert key_feature.shape == (2, 8, 3, 128)
    assert torch.all(query_feature > 0)
    assert torch.all(key_feature > 0)
    gate = torch.zeros(2, 3, 2048)
    output = module._readout(
        torch.randn(2, 8, 3, 256),
        torch.ones(2, 8, 3),
        gate,
    )
    assert output.shape == (2, 3, 2048)
    assert output.dtype == torch.float32


def test_gqa_fp32_reference_applies_source_gate_and_o_projection() -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    hidden = torch.randn(1, 3, 2048)
    heads, gate = module.attention_heads_reference(hidden)
    expected = module.o_proj(heads.transpose(1, 2).reshape_as(hidden) * torch.sigmoid(gate).float())
    assert torch.allclose(module.reference_forward(hidden), expected)


def test_source_projection_norm_rope_and_gate_transfer() -> None:
    config = _config()
    source = Qwen3_5Attention(config, 0).float().eval()
    module = Qwen2RWKVTimeMix(config, 0).float()
    module.load_source_attention(source)
    hidden = torch.randn(2, 7, 2048)
    positions = torch.arange(11, 18).expand(2, -1)
    embeddings = module.rotary_emb(hidden, positions)
    wanted = _source_qkv(source, hidden, embeddings)
    actual = module._project_qkv(hidden, positions)
    for actual_tensor, wanted_tensor in zip(actual, wanted, strict=True):
        assert torch.equal(actual_tensor, wanted_tensor)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        assert torch.equal(getattr(source, name).weight, getattr(module, name).weight)


def test_rope_rebuild_ignores_uninitialized_nonpersistent_buffers() -> None:
    rope = _FP32RotaryEmbedding(_config())
    with torch.no_grad():
        rope.inv_freq.fill_(float("nan"))
        rope.original_inv_freq.fill_(float("nan"))
    hidden = torch.zeros(1, 4, 2048)
    positions = torch.arange(4).view(1, -1)
    cos, sin = rope(hidden, positions)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
    assert cos.dtype == sin.dtype == hidden.dtype


def test_gqa_schema_rejects_bounded_artifact() -> None:
    values = _config().to_dict()
    values["gqa_readout_mode"] = "hedgehog_h2o_shared_norm"
    with pytest.raises(ValueError, match="pure RWKV"):
        Qwen2RWKVConfig.from_dict(values)
    values = _config().to_dict()
    values["gqa_sidecar_capacity"] = 128
    with pytest.raises(ValueError, match="removed fields"):
        Qwen2RWKVConfig.from_dict(values)
    values = _config().to_dict()
    values["gqa_checkpoint_schema"] = "gqa_rwkv_feature_state_d256x2_v2"
    with pytest.raises(ValueError, match="schema"):
        Qwen2RWKVConfig.from_dict(values)


def test_fresh_process_strict_load_has_no_distillation_parameters() -> None:
    config = _config()
    module = Qwen2RWKVTimeMix(config, 0).half().eval()
    _fresh_process_gqa_strict_load(config, 0, module)
    forbidden = ("teacher", "sidecar", "beta_logit", "branch_gate", "lora")
    assert not any(token in name for name in module.state_dict() for token in forbidden)


def test_gate_schedule_mixes_frozen_teacher_and_pure_student() -> None:
    assert GQA_GATE_SCHEDULE == (0.9, 0.75, 0.5, 0.25, 0.1, 0.0)
    student = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4, requires_grad=True)
    assert torch.equal(_mix_teacher(student, teacher, 1.0), teacher.detach())
    assert torch.equal(_mix_teacher(student, teacher, 0.0), student)
    mixed = _mix_teacher(student, teacher, 0.25)
    mixed.square().mean().backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_block_gate_and_final_cache_use_pure_student(monkeypatch) -> None:
    layer = Qwen2RWKVDecoderLayer(_config(), 0).float().train()
    monkeypatch.setattr(
        layer.tmix,
        "_training_forward",
        lambda hidden, first: (layer.tmix.reference_forward(hidden), first),
    )
    hidden = torch.randn(2, 3, 2048)
    teacher = torch.randn_like(hidden, requires_grad=True)
    objective = _LayerObjective(layer)
    pure, _ = objective(hidden)
    mixed, _ = objective(hidden, teacher, 1.0)
    residual = hidden + teacher.detach()
    expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
    assert torch.equal(mixed, expected)
    assert torch.equal(objective(hidden, teacher, 0.0)[0], pure)

    def forbidden_teacher(*args):
        raise AssertionError("gate=0 cache refresh accessed the GQA teacher")

    monkeypatch.setattr(train, "_teacher_tmix_output", forbidden_teacher)
    student = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    stored = []
    cache = SimpleNamespace(store=lambda output, slot, **kwargs: stored.append((output, slot)))
    _refresh_gqa_stage_cache(None, student, 0, hidden, cache, torch.device("cpu"), 0.0)
    assert stored[0][1] == "next"
    assert torch.equal(stored[0][0], pure)


@pytest.mark.parametrize("gate", GQA_GATE_SCHEDULE)
def test_gate_loss_keeps_direct_student_gradient(gate: float) -> None:
    student = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4)
    pure_loss = _nmse_loss(student, teacher)
    loss = _nmse_loss(_mix_teacher(student, teacher, gate), teacher) + 4 * pure_loss
    expected = (4 + (1 - gate) ** 2) * pure_loss
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        torch.autograd.grad(loss, student, retain_graph=True)[0],
        torch.autograd.grad(expected, student)[0],
    )


@pytest.mark.parametrize("distance", (128, 256, 512, 1024))
def test_feature_state_long_distance_recall(distance: int) -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    metrics = evaluate_gqa_recall(module, (distance,))
    assert metrics[f"teacher_recall_{distance}_hit_at_1"] == 1
    assert metrics[f"recall_{distance}_hit_at_1"] >= 0.9


def test_recall_fixture_detects_feature_collapse() -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    with torch.no_grad():
        module.feature_q_weight.zero_()
        module.feature_k_weight.zero_()
    metrics = evaluate_gqa_recall(module, (128,))
    assert metrics["teacher_recall_128_hit_at_1"] == 1
    assert metrics["recall_128_hit_at_1"] == 0


def test_recall_fixture_detects_learned_forgetting() -> None:
    module = Qwen2RWKVTimeMix(_config(), 0).float()
    with torch.no_grad():
        module.w0.fill_(0.5)
    metrics = evaluate_gqa_recall(module, (128,))
    assert metrics["teacher_recall_128_hit_at_1"] == 1
    assert metrics["recall_128_hit_at_1"] == 0


@pytest.mark.parametrize("failure", ("numerics", "recall", "nan"))
def test_gqa_acceptance_rejects_failed_pure_checkpoint(failure: str) -> None:
    native = {"layer_output_nmse": 1e-3}
    reference = dict.fromkeys(
        ("native_incremental_layer_output_nmse", "native_incremental_tmix_output_nmse"), 1e-4
    )
    recall = {f"recall_{distance}_hit_at_1": 1.0 for distance in (128, 256, 512, 1024)}
    if failure == "numerics":
        reference["native_incremental_tmix_output_nmse"] = 2e-3
    elif failure == "recall":
        recall["recall_1024_hit_at_1"] = 0.875
    else:
        native["layer_output_nmse"] = float("nan")
    with pytest.raises((RuntimeError, FloatingPointError)):
        _require_gqa_acceptance(native, reference, recall, 3)


def test_source_block_nmse_has_no_arbitrary_acceptance_cutoff() -> None:
    reference = dict.fromkeys(
        ("native_incremental_layer_output_nmse", "native_incremental_tmix_output_nmse"), 1e-4
    )
    recall = {f"recall_{distance}_hit_at_1": 1.0 for distance in (128, 256, 512, 1024)}
    _require_gqa_acceptance({"layer_output_nmse": 0.05}, reference, recall, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashRWKV2 requires CUDA")
def test_flash_bf16_matches_fp32_reference_and_recall() -> None:
    torch.manual_seed(0)
    module = Qwen2RWKVTimeMix(_config(), 0).bfloat16().cuda()
    query = torch.randn(1, 8, 64, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 64, 256, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(query)
    features = module._features(query, key)
    native = module._state_training(*features, values)
    reference = module._state_reference(*features, values)
    for actual, wanted in zip(native, reference, strict=True):
        assert _nmse_loss(actual, wanted) <= 1e-3
    (native[0] / native[1][..., None]).square().mean().backward()
    for parameter in module.attention_transfer_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert min(evaluate_gqa_recall(module, use_flash=True).values()) >= 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashRWKV2 requires CUDA")
@torch.no_grad()
def test_flash_fp16_prefill_chunks_decode_and_fixed_state_soak() -> None:
    torch.manual_seed(0)
    config = _config()
    layer = Qwen2RWKVDecoderLayer(config, 0).half().cuda().eval()
    hidden = torch.randn(1, 512, 2048, device="cuda", dtype=torch.float16)
    full_tmix, full_block, full_cache = _fp16_forward_mode(layer, hidden, config, None)
    for chunk in (64, 128, 256, 1):
        tmix, block, cache = _fp16_forward_mode(layer, hidden, config, chunk)
        for actual, wanted in ((tmix, full_tmix), (block, full_block)):
            assert _nmse_loss(actual, wanted).sqrt() <= 1e-3
        assert int(cache.elapsed[0][0]) == int(full_cache.elapsed[0][0]) == 512
        assert cache.get_seq_length(0) == 512
    metrics = _gqa_fp16_soak(layer, hidden[:, :1], config)
    assert metrics["soak_tokens"] == 8192
    assert metrics["recurrent_bytes_per_sequence"] == 2 * 1024 * 1024
    assert metrics["token_shift_bytes_per_sequence"] == 2048 * 2


def test_rwkv_extension_preserves_transferred_function() -> None:
    torch.manual_seed(42)
    config = _config()
    source = Qwen3_5Attention(config, 0).float().eval()
    module = Qwen2RWKVTimeMix(config, 0).float()
    module.load_source_attention(source)
    hidden = torch.randn(2, 17, 2048)
    positions = torch.arange(17).expand(2, -1)
    query, key, value, gate = _source_qkv(source, hidden, module.rotary_emb(hidden, positions))
    feature_query, feature_key = module._features(query, key)
    weights = (feature_query @ feature_key.transpose(-1, -2)).tril()
    heads = weights @ value.repeat_interleave(4, dim=1) / weights.sum(-1, keepdim=True)
    expected = source.o_proj(heads.transpose(1, 2).reshape_as(hidden) * gate.sigmoid())
    actual = module.reference_forward(hidden, v_first=torch.randn_like(hidden))
    assert _nmse_loss(actual, expected) < 1e-12


def test_training_expansion_preserves_v2_checkpoint(tmp_path) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(42)
    original = Qwen2RWKVTimeMix(_config(), 0).float()
    transferred = train._gqa_transferred_names(original)
    state = {name: value for name, value in original.state_dict().items() if name in transferred}
    checkpoint = tmp_path / "layer_03.safetensors"
    save_file(state, str(checkpoint))
    expanded = Qwen2RWKVTimeMix(_config(), 0).float()
    train._load_gqa_initial_checkpoint(expanded, checkpoint)
    hidden = torch.randn(1, 17, 2048)
    first = torch.randn_like(hidden)
    assert torch.equal(
        expanded.reference_forward(hidden, v_first=first),
        original.reference_forward(hidden, v_first=first),
    )
    state.pop("q_proj.weight")
    save_file(state, str(checkpoint))
    with pytest.raises(ValueError, match="complete v2 or v3"):
        train._load_gqa_initial_checkpoint(expanded, checkpoint)


@pytest.mark.parametrize("finished", [False, True])
def test_generation_smoke_rejects_truncated_answers(tmp_path, monkeypatch, finished) -> None:
    import json

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2]])},
        decode=lambda *args, **kwargs: "A readable answer.",
    )
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[9]),
        generate=lambda *args, **kwargs: torch.tensor([[1, 2, 3, 9 if finished else 4]]),
    )
    model.cuda = model.eval = lambda: model
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(Qwen2RWKVForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    assert train._accept(tmp_path) is finished
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert all(
        row["finish_reason"] == ("eos" if finished else "length") for row in report["generations"]
    )


def test_value_residual_cache_round_trip(tmp_path) -> None:
    from any2rwkv.qwen2rwkv.qwen3_5.align.last_layer_cache import LastLayerCache

    cache = LastLayerCache(tmp_path, 0)
    hidden = torch.randn(2, 16, 2048).bfloat16()
    first = torch.randn_like(hidden)
    cache.store(hidden, v_first=first)
    cache.advance()
    assert torch.equal(cache.load(), hidden)
    assert torch.equal(cache.load_v_first(), first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashRWKV2 requires CUDA")
def test_rwkv_zero_initialization_has_native_gradients() -> None:
    torch.manual_seed(42)
    module = Qwen2RWKVTimeMix(_config(), 0).bfloat16().cuda().train()
    hidden = torch.randn(2, 32, 2048, device="cuda", dtype=torch.bfloat16)
    first = torch.randn_like(hidden)
    output, _ = module(hidden, first)
    output.float().square().mean().backward()
    for name in ("x_r", "x_k", "x_v", "x_g", "w0", "w2", "a0", "a2", "v0", "v2", "r_k", "norm_mix"):
        gradient = dict(module.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashRWKV2 requires CUDA")
@torch.no_grad()
def test_trained_rwkv_dynamics_match_reference_and_cached_decode() -> None:
    import copy

    torch.manual_seed(42)
    config = _config()
    layer = Qwen2RWKVDecoderLayer(config, 0).bfloat16().cuda().train()
    tmix = layer.tmix
    for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "v0", "norm_mix"):
        getattr(tmix, name).fill_(0.03)
    tmix.w0.fill_(0.001)
    tmix.a0.fill_(0.01)
    tmix.k_a.fill_(0.02)
    tmix.r_k.fill_(0.0001)
    for name in ("w2", "a2", "v2"):
        getattr(tmix, name).normal_(std=1e-4)
    hidden = torch.randn(1, 256, 2048, device="cuda", dtype=torch.bfloat16)
    first = torch.randn_like(hidden)
    reference = copy.deepcopy(tmix).float()
    actual = tmix(hidden, first)[0]
    expected = reference.reference_forward(hidden.float(), v_first=first.float())
    assert _nmse_loss(actual, expected) < 1e-3
    layer.half().eval()
    hidden, first = hidden.half(), first.half()
    full, _, _ = _fp16_forward_mode(layer, hidden, config, None, v_first=first)
    for chunk in (64, 128, 1):
        cached, _, cache = _fp16_forward_mode(layer, hidden, config, chunk, v_first=first)
        assert _nmse_loss(cached, full).sqrt() < 1e-3
        assert cache.layers[0].conv_states[0].shape == (1, 2048, 1)
