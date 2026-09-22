"""CPU tests for NVFP4 fake-quant QAT hooks.

The fake modules below mimic the one Transformer Engine behavior the hooks rely on:
forward reads its weights only through ``_get_weight_tensors()``. Module names follow
Megatron's Qwen3.5 layout so the default include pattern is exercised as-is.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.quant import fake_quantize_nvfp4, quantize_nvfp4
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from slime.backends.megatron_utils.nvfp4_qat import (
    DEFAULT_INCLUDE,
    install_nvfp4_qat,
    is_nvfp4_qat_target,
    megatron_amax_group,
    nvfp4_qdq_for_sync,
    nvfp4_sync_family_amax,
    set_nvfp4_qat_enabled,
)


@pytest.fixture(autouse=True)
def _reenable_qat_after_each_test():
    yield
    set_nvfp4_qat_enabled(True)


HIDDEN = 32
FFN = 64
NUM_EXPERTS = 3


class _FakeTELinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_names = ["weight"]
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.05)

    def _get_weight_tensors(self):
        return [getattr(self, name) for name in self.weight_names]

    def forward(self, x):
        return F.linear(x, torch.cat(self._get_weight_tensors()))


class _FakeTEGroupedLinear(nn.Module):
    def __init__(self, num_gemms, in_features, out_features):
        super().__init__()
        self.num_gemms = num_gemms
        for i in range(num_gemms):
            weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.05 * (i + 1)
            setattr(self, f"weight{i}", nn.Parameter(weight))

    def _get_weight_tensors(self):
        return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

    def forward(self, x):
        return torch.stack([F.linear(x, w) for w in self._get_weight_tensors()])


class _FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = _FakeTELinear(HIDDEN, 3 * HIDDEN)
        self.linear_proj = _FakeTELinear(HIDDEN, HIDDEN)


class _GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False, dtype=torch.bfloat16)
        self.in_proj_z = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)
        self.in_proj_b = nn.Linear(HIDDEN, 4, bias=False, dtype=torch.bfloat16)
        self.in_proj_a = nn.Linear(HIDDEN, 4, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)


class _LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _GatedDeltaNet()


class _DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = _FakeTELinear(HIDDEN, 2 * FFN)
        self.linear_fc2 = _FakeTELinear(FFN, HIDDEN)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = _FakeTEGroupedLinear(NUM_EXPERTS, HIDDEN, 2 * FFN)
        self.linear_fc2 = _FakeTEGroupedLinear(NUM_EXPERTS, FFN, HIDDEN)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear on purpose: the router must be skipped by name, not by type.
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False, dtype=torch.bfloat16)
        self.experts = _Experts()
        self.shared_experts = _DenseMLP()


class _Layer(nn.Module):
    def __init__(self, attention, mlp):
        super().__init__()
        self.self_attention = attention
        self.mlp = mlp


class _Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embedding = nn.Embedding(16, HIDDEN, dtype=torch.bfloat16)
        self.decoder = _Decoder(layers)
        self.output_layer = nn.Linear(HIDDEN, 16, bias=False, dtype=torch.bfloat16)


def _dense_model():
    torch.manual_seed(0)
    return _Model([_Layer(_LinearAttention(), _DenseMLP()), _Layer(_FullAttention(), _DenseMLP())])


def _moe_model():
    torch.manual_seed(0)
    return _Model([_Layer(_LinearAttention(), _MoE()), _Layer(_FullAttention(), _MoE())])


def test_default_include_selects_the_twelve_dense_linear_kinds():
    installed = install_nvfp4_qat(_dense_model())

    gdn = [
        f"decoder.layers.0.self_attention.linear_attn.{n}"
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    ]
    attn = ["decoder.layers.1.self_attention.linear_qkv", "decoder.layers.1.self_attention.linear_proj"]
    mlp = [f"decoder.layers.{i}.mlp.{n}" for i in (0, 1) for n in ("linear_fc1", "linear_fc2")]
    assert installed == sorted(gdn + attn + mlp)


def test_default_include_covers_experts_and_shared_experts_but_not_router():
    installed = install_nvfp4_qat(_moe_model())

    assert "decoder.layers.0.mlp.experts.linear_fc1" in installed
    assert "decoder.layers.0.mlp.shared_experts.linear_fc2" in installed
    assert not [name for name in installed if "router" in name]


def test_exclude_narrows_to_routed_experts_only():
    installed = install_nvfp4_qat(_moe_model(), exclude=[r"self_attention", r"shared_experts"])

    assert installed == sorted(
        f"decoder.layers.{i}.mlp.experts.{n}" for i in (0, 1) for n in ("linear_fc1", "linear_fc2")
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("decoder.layers.3.mlp.linear_fc1", True),
        ("module.module.decoder.layers.3.mlp.linear_fc1.weight", True),
        ("module.module.decoder.layers.3.mlp.experts.linear_fc1.weight7", True),
        ("module.module.language_model.decoder.layers.0.self_attention.linear_proj.weight", True),
        ("module.module.decoder.layers.0.self_attention.linear_attn.in_proj_a.weight", True),
        ("module.module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight", False),
        ("module.module.decoder.layers.0.self_attention.linear_qkv.bias", False),
        ("module.module.decoder.layers.0.mlp.router.weight", False),
        ("module.module.mtp.layers.0.mlp.linear_fc1.weight", False),
        ("module.module.embedding.word_embeddings.weight", False),
        ("module.module.output_layer.weight", False),
    ],
)
def test_target_predicate_agrees_for_module_and_parameter_names(name, expected):
    assert is_nvfp4_qat_target(name) is expected


def _input():
    torch.manual_seed(1)
    return torch.randn(5, HIDDEN, dtype=torch.bfloat16)


def _amax(*weights):
    return torch.stack([w.detach().abs().amax().float() for w in weights]).amax()


def test_te_dense_forward_uses_the_fake_quantized_weight():
    model = _dense_model()
    layer = model.decoder.layers[1].self_attention.linear_proj
    unquantized = layer(_input())
    install_nvfp4_qat(model)

    expected = F.linear(_input(), fake_quantize_nvfp4(layer.weight.detach()))
    assert torch.equal(layer(_input()), expected)
    assert not torch.equal(expected, unquantized)


def test_plain_linear_forward_uses_the_fake_quantized_weight():
    model = _dense_model()
    layer = model.decoder.layers[0].self_attention.linear_attn.out_proj
    install_nvfp4_qat(model)

    expected = F.linear(_input(), fake_quantize_nvfp4(layer.weight.detach()))
    assert torch.equal(layer(_input()), expected)


def test_grouped_forward_quantizes_each_expert_with_its_own_amax():
    model = _moe_model()
    layer = model.decoder.layers[0].mlp.experts.linear_fc1
    install_nvfp4_qat(model)

    weights = [getattr(layer, f"weight{i}").detach() for i in range(NUM_EXPERTS)]
    expected = torch.stack([F.linear(_input(), fake_quantize_nvfp4(w)) for w in weights])
    shared = torch.stack([F.linear(_input(), fake_quantize_nvfp4(w, tensor_amax=_amax(*weights))) for w in weights])
    assert torch.equal(layer(_input()), expected)
    assert not torch.equal(expected, shared)


def test_gdn_input_projections_share_one_family_amax():
    model = _dense_model()
    gdn = model.decoder.layers[0].self_attention.linear_attn
    with torch.no_grad():
        gdn.in_proj_qkv.weight.mul_(4)  # make the family amax differ from in_proj_z's own amax
    install_nvfp4_qat(model)

    family = _amax(gdn.in_proj_qkv.weight, gdn.in_proj_z.weight, gdn.in_proj_b.weight, gdn.in_proj_a.weight)
    expected = F.linear(_input(), fake_quantize_nvfp4(gdn.in_proj_z.weight.detach(), tensor_amax=family))
    own = F.linear(_input(), fake_quantize_nvfp4(gdn.in_proj_z.weight.detach()))
    assert torch.equal(gdn.in_proj_z(_input()), expected)
    assert not torch.equal(expected, own)


@pytest.mark.parametrize(
    "pick",
    [
        lambda m: m.decoder.layers[1].mlp.linear_fc2,
        lambda m: m.decoder.layers[0].self_attention.linear_attn.out_proj,
    ],
    ids=["te_dense", "plain_linear"],
)
def test_gradient_reaches_the_high_precision_weight_through_the_ste(pick):
    model = _dense_model()
    layer = pick(model)
    install_nvfp4_qat(model)
    x = torch.randn(5, layer.weight.shape[1], dtype=torch.bfloat16)

    layer(x).sum().backward()

    reference = layer.weight.detach().clone().requires_grad_()
    F.linear(x, fake_quantize_nvfp4(reference)).sum().backward()
    assert layer.weight.grad is not None
    assert torch.equal(layer.weight.grad, reference.grad)


class _Holder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer


@pytest.mark.parametrize("partition_dim", [0, 1], ids=["column_parallel", "row_parallel"])
def test_tp_shard_quantizes_on_the_full_tensor_grid(monkeypatch, partition_dim):
    torch.manual_seed(2)
    full = torch.randn(64, 64, dtype=torch.bfloat16) * 0.05
    full[40, 50] = 3.0  # the global max lives in shard 1, so shard 0 is wrong without the reduction
    shard, other_shard = full.chunk(2, dim=partition_dim)
    expected_weight = fake_quantize_nvfp4(full).chunk(2, dim=partition_dim)[0]

    tp_group = object()

    def all_reduce_max_with_other_rank(tensor, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is tp_group
        tensor.copy_(torch.maximum(tensor, _amax(other_shard)))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce_max_with_other_rank)

    layer = _FakeTELinear(shard.shape[1], shard.shape[0])
    layer.weight.data = shard.clone()
    install_nvfp4_qat(_Holder(layer), include=[r"^layer$"], amax_group=lambda name, param: tp_group)

    x = torch.randn(3, shard.shape[1], dtype=torch.bfloat16)
    assert torch.equal(layer(x), F.linear(x, expected_weight))
    assert not torch.equal(expected_weight, fake_quantize_nvfp4(shard))

    synced_full = nvfp4_qdq_for_sync("module.module.layer.weight", full, include=[r"^layer$"])
    assert torch.equal(synced_full.chunk(2, dim=partition_dim)[0], layer._get_weight_tensors()[0])


def _backup(model):
    """Shape of ``weights_backuper.get("actor")``: global Megatron parameter names to tensors."""
    return {f"module.module.{name}": param.detach().clone() for name, param in model.named_parameters()}


GDN = "module.module.decoder.layers.0.self_attention.linear_attn"


def test_sync_weight_equals_the_weight_the_hooked_te_forward_used():
    model = _moe_model()
    install_nvfp4_qat(model)
    backup = _backup(model)

    qkv = "module.module.decoder.layers.1.self_attention.linear_qkv.weight"
    used = model.decoder.layers[1].self_attention.linear_qkv._get_weight_tensors()[0]
    assert torch.equal(nvfp4_qdq_for_sync(qkv, backup[qkv]), used)

    experts = model.decoder.layers[0].mlp.experts.linear_fc2
    for i, used in enumerate(experts._get_weight_tensors()):
        name = f"module.module.decoder.layers.0.mlp.experts.linear_fc2.weight{i}"
        assert torch.equal(nvfp4_qdq_for_sync(name, backup[name]), used)


def test_sync_weight_of_a_gdn_projection_uses_the_family_amax():
    model = _dense_model()
    gdn = model.decoder.layers[0].self_attention.linear_attn
    with torch.no_grad():
        gdn.in_proj_qkv.weight.mul_(4)
    install_nvfp4_qat(model)
    backup = _backup(model)

    with nvfp4_sync_family_amax(backup.items()):
        synced = nvfp4_qdq_for_sync(f"{GDN}.in_proj_z.weight", backup[f"{GDN}.in_proj_z.weight"])

    assert torch.equal(F.linear(_input(), synced), gdn.in_proj_z(_input()))


def test_sync_of_a_family_member_without_its_family_amax_raises():
    weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="nvfp4_sync_family_amax"):
        nvfp4_qdq_for_sync(f"{GDN}.in_proj_z.weight", weight)

    with nvfp4_sync_family_amax(_backup(_dense_model()).items()):
        pass
    with pytest.raises(RuntimeError, match="nvfp4_sync_family_amax"):
        nvfp4_qdq_for_sync(f"{GDN}.in_proj_z.weight", weight)


def test_sync_returns_unquantized_parameters_as_is():
    router = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16)
    assert nvfp4_qdq_for_sync("module.module.decoder.layers.0.mlp.router.weight", router) is router


def test_synced_weight_packs_to_the_same_nvfp4_codes_as_the_master_weight():
    torch.manual_seed(3)
    master = torch.randn(64, 256, dtype=torch.bfloat16) * 0.05
    master.view(-1)[torch.randint(0, master.numel(), (8,))] *= 40  # heavy tail, like real LLM weights

    synced = nvfp4_qdq_for_sync("module.module.decoder.layers.0.mlp.linear_fc2.weight", master)

    from_master, from_synced = quantize_nvfp4(master), quantize_nvfp4(synced)
    assert torch.equal(from_synced.packed_weight, from_master.packed_weight)
    assert torch.equal(from_synced.weight_scale.view(torch.uint8), from_master.weight_scale.view(torch.uint8))
    assert torch.equal(from_synced.weight_scale_2, from_master.weight_scale_2)
    assert torch.equal(from_master.reconstruction, synced)


def test_disabling_qat_makes_every_hooked_forward_use_the_high_precision_weight():
    model = _moe_model()
    layers = [
        model.decoder.layers[1].self_attention.linear_proj,
        model.decoder.layers[0].self_attention.linear_attn.out_proj,
        model.decoder.layers[0].mlp.experts.linear_fc1,
    ]
    high_precision = [layer(_input()) for layer in layers]
    install_nvfp4_qat(model)
    quantized = [layer(_input()) for layer in layers]

    set_nvfp4_qat_enabled(False)
    assert all(torch.equal(layer(_input()), out) for layer, out in zip(layers, high_precision, strict=True))

    set_nvfp4_qat_enabled(True)
    assert all(torch.equal(layer(_input()), out) for layer, out in zip(layers, quantized, strict=True))
    assert not any(torch.equal(q, h) for q, h in zip(quantized, high_precision, strict=True))


def test_install_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="matched no modules"):
        install_nvfp4_qat(_dense_model(), include=[r"^transformer\.h\.\d+\.attn$"])


def test_install_raises_on_a_selected_module_that_is_not_a_linear():
    with pytest.raises(TypeError, match="not a supported linear"):
        install_nvfp4_qat(_dense_model(), include=[r"^decoder\.layers\.0\.mlp$"])


def test_install_raises_when_in_features_is_not_a_multiple_of_the_block_size():
    # Padding a shard would move its block boundaries off the gathered tensor's grid.
    with pytest.raises(ValueError, match="multiple of 16"):
        install_nvfp4_qat(_Holder(_FakeTELinear(24, 32)), include=[r"^layer$"])


def test_install_raises_when_a_dense_te_linear_holds_several_weights():
    layer = _FakeTELinear(HIDDEN, HIDDEN)
    layer.weight_names = ["weight", "weight"]
    with pytest.raises(ValueError, match="several weight tensors"):
        install_nvfp4_qat(_Holder(layer), include=[r"^layer$"])


def test_install_raises_when_exclude_splits_a_fused_family():
    with pytest.raises(ValueError, match="Incomplete fused NVFP4 family"):
        install_nvfp4_qat(_dense_model(), exclude=[r"in_proj_z$"])


def test_installing_twice_does_not_wrap_the_hooks_again():
    model = _dense_model()
    te_layer = model.decoder.layers[1].mlp.linear_fc1
    plain_layer = model.decoder.layers[0].self_attention.linear_attn.out_proj
    first = install_nvfp4_qat(model)
    hooks = (te_layer._get_weight_tensors, plain_layer.forward)

    assert install_nvfp4_qat(model) == first
    assert (te_layer._get_weight_tensors, plain_layer.forward) == hooks


def _qwen3_5_args(nvfp4_qat):
    return SimpleNamespace(
        nvfp4_qat=nvfp4_qat,
        nvfp4_qat_include=list(DEFAULT_INCLUDE),
        nvfp4_qat_exclude=[],
        vocab_size=64,
        hidden_size=32,
        kv_channels=8,
        num_attention_heads=4,
        num_query_groups=2,
        q_lora_rank=None,
    )


def test_convert_to_hf_splits_the_fake_quantized_qkv_when_qat_is_on():
    torch.manual_seed(4)
    name = "module.module.decoder.layers.3.self_attention.linear_qkv.weight"
    qkv = torch.randn(96, 32, dtype=torch.bfloat16) * 0.05  # 2 groups x (2 q + 2 gate + k + v) x head_dim 8

    synced = dict(convert_to_hf(_qwen3_5_args(nvfp4_qat=True), "qwen35", name, qkv))

    # Quantizing the fused tensor first gives q/k/v one shared tensor scale, as the serving runtime needs.
    expected = dict(convert_to_hf(_qwen3_5_args(nvfp4_qat=False), "qwen35", name, fake_quantize_nvfp4(qkv)))
    unquantized = dict(convert_to_hf(_qwen3_5_args(nvfp4_qat=False), "qwen35", name, qkv))
    assert synced.keys() == expected.keys()
    assert all(torch.equal(synced[key], expected[key]) for key in expected)
    assert not torch.equal(
        synced["model.language_model.layers.3.self_attn.k_proj.weight"],
        unquantized["model.language_model.layers.3.self_attn.k_proj.weight"],
    )


def test_convert_to_hf_leaves_norms_in_high_precision_when_qat_is_on():
    name = "module.module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"
    norm = torch.randn(32, dtype=torch.bfloat16)

    ((_, synced),) = convert_to_hf(_qwen3_5_args(nvfp4_qat=True), "qwen35", name, norm)

    assert torch.equal(synced, norm)


@pytest.fixture
def parallel_state(monkeypatch):
    """Stand-in for megatron.core.parallel_state with TP=2 and expert-TP=1."""
    state = ModuleType("megatron.core.parallel_state")
    state.get_tensor_model_parallel_world_size = lambda: 2
    state.get_tensor_model_parallel_group = lambda: "tp_group"
    state.get_expert_tensor_parallel_world_size = lambda: 1
    state.get_expert_tensor_parallel_group = lambda: "etp_group"
    core = ModuleType("megatron.core")
    core.parallel_state = state
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", state)
    return state


def _megatron_weight(**attributes):
    weight = nn.Parameter(torch.zeros(16, 16))
    for key, value in attributes.items():
        setattr(weight, key, value)
    return weight


def test_megatron_amax_group_is_the_tp_group_for_a_sharded_dense_weight(parallel_state):
    weight = _megatron_weight(tensor_model_parallel=True)
    assert megatron_amax_group("decoder.layers.0.mlp.linear_fc1", weight) == "tp_group"


def test_megatron_amax_group_is_none_for_weights_every_rank_holds_in_full(parallel_state):
    duplicated = _megatron_weight(tensor_model_parallel=True, parallel_mode="duplicated")
    replicated = _megatron_weight(tensor_model_parallel=False)
    assert megatron_amax_group("decoder.layers.0.self_attention.linear_attn.in_proj_z", duplicated) is None
    assert megatron_amax_group("decoder.layers.0.self_attention.linear_attn.out_proj", replicated) is None


def test_megatron_amax_group_uses_the_expert_tp_group_for_routed_experts(parallel_state):
    weight = _megatron_weight(tensor_model_parallel=True)
    name = "decoder.layers.0.mlp.experts.linear_fc1"
    assert megatron_amax_group(name, weight) is None  # expert-TP size 1: each rank holds whole experts

    parallel_state.get_expert_tensor_parallel_world_size = lambda: 2
    assert megatron_amax_group(name, weight) == "etp_group"


def test_nested_sync_contexts_restore_the_outer_family_amax():
    # The disk weight updater saves an HF checkpoint (its own context) inside the actor's sync context.
    outer = _backup(_dense_model())
    # x3, not x2: a power-of-two amax change is absorbed exactly by the FP8 block scales.
    inner = {name: weight * 3 for name, weight in outer.items()}
    name = f"{GDN}.in_proj_z.weight"

    with nvfp4_sync_family_amax(outer.items()):
        expected = nvfp4_qdq_for_sync(name, outer[name])
        with nvfp4_sync_family_amax(inner.items()):
            assert not torch.equal(nvfp4_qdq_for_sync(name, outer[name]), expected)
        assert torch.equal(nvfp4_qdq_for_sync(name, outer[name]), expected)
