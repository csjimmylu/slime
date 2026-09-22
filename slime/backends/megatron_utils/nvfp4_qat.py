"""NVFP4 W4A16 fake-quant QAT for the Megatron actor.

The forward pass of every selected linear uses ``v = Q(w)`` while the optimizer keeps
updating the high-precision ``w``; weight sync sends the same ``v`` to the rollout engine. The
QDQ math is imported from ``qatfactory.quant`` (shared with the other baselines) and never
reimplemented here; this module only decides which layers are quantized, which amax each one
uses, and installs the hooks.

Nothing at module level imports Megatron or Transformer Engine, so the logic is testable on CPU.
"""

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from qatfactory.quant import (
    build_nvfp4_fused_family_names,
    fake_quantize_nvfp4,
    nvfp4_fused_family_candidates,
)
from qatfactory.quant.formats.nvfp4 import NVFP4_BLOCK_SIZE
from torch import nn


# Megatron module names, after stripping wrapper prefixes. An allowlist: embeddings, the output
# layer, norms, the MoE router, MTP and vision towers stay in high precision by not being listed.
DEFAULT_INCLUDE = (
    r"^decoder\.layers\.\d+\.("
    r"self_attention\.(linear_qkv|linear_proj)"
    r"|self_attention\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    r"|mlp\.(linear_fc1|linear_fc2)"
    r"|mlp\.shared_experts\.(linear_fc1|linear_fc2)"
    r"|mlp\.experts\.(linear_fc1|linear_fc2)"
    r")$",
)

# Maps (module name, weight) to the process group the weight is sharded over, or None when every
# rank holds the full tensor.
AmaxGroupResolver = Callable[[str, torch.Tensor], "dist.ProcessGroup | None"]

_WRAPPER_PREFIX = re.compile(r"^(module\.)*(language_model\.)?")
_WEIGHT_SUFFIX = re.compile(r"\.weight\d*$")

# The actor process swaps ref / teacher weights into the one model object it owns, so the hooks
# have to be switchable: those forwards must see the high-precision weights.
_enabled = True

# Family amax per module name, alive only inside ``nvfp4_sync_family_amax``.
_sync_family_amax: dict[str, torch.Tensor] | None = None


def set_nvfp4_qat_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def _module_name(name: str) -> str:
    return _WEIGHT_SUFFIX.sub("", _WRAPPER_PREFIX.sub("", name))


def is_nvfp4_qat_target(
    name: str,
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = (),
) -> bool:
    """Whether a module name, or the name of one of its weight parameters, is quantized.

    Hook installation passes module names and weight sync passes parameter names; both go
    through this one predicate so the trainer and the rollout engine agree on the layer set.
    """
    module_name = _module_name(name)
    if not any(re.search(pattern, module_name) for pattern in include):
        return False
    return not any(re.search(pattern, module_name) for pattern in exclude)


def _amax(weights: Sequence[torch.Tensor], group: "dist.ProcessGroup | None" = None) -> torch.Tensor:
    """Tensor-scale amax over ``weights``, reduced over ``group`` when the weights are shards.

    MAX is exact, so the reduced value equals the amax of the gathered tensor bit for bit; that
    is what puts every shard on the same NVFP4 grid as the full tensor sent to the rollout engine.
    """
    amax = torch.stack([w.detach().abs().amax().float() for w in weights]).amax()
    if group is not None:
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
    return amax


def megatron_amax_group(name: str, weight: torch.Tensor) -> "dist.ProcessGroup | None":
    """The ``AmaxGroupResolver`` for Megatron models.

    Applies the rule ``all_gather_param`` (update_weight/common.py) uses to decide what weight
    sync gathers, so the hooks reduce the amax of exactly the weights that reach the rollout
    engine as a concatenation of shards.
    """
    from megatron.core import parallel_state as mpu

    if not getattr(weight, "tensor_model_parallel", False) or getattr(weight, "parallel_mode", None) == "duplicated":
        return None
    if ".experts." in name:
        return mpu.get_expert_tensor_parallel_group() if mpu.get_expert_tensor_parallel_world_size() > 1 else None
    return mpu.get_tensor_model_parallel_group() if mpu.get_tensor_model_parallel_world_size() > 1 else None


def _check_block_aligned(name: str, weight: torch.Tensor) -> None:
    # fake_quantize_nvfp4 pads a ragged last block. A padded row-parallel shard would put its
    # block boundaries somewhere else than the gathered tensor the rollout engine receives.
    if weight.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 QAT requires the local in_features of {name} to be a multiple of {NVFP4_BLOCK_SIZE}, got weight shape {tuple(weight.shape)}."
        )


def _is_grouped_linear(module: nn.Module) -> bool:
    return any("GroupedLinear" in cls.__name__ for cls in type(module).__mro__)


def _hook_weight_getter(module: nn.Module, per_tensor_amax: bool, group: "dist.ProcessGroup | None") -> None:
    """Quantize what Transformer Engine's ``_get_weight_tensors()`` hands to the GEMM.

    A TE dense linear is one fused GEMM, so its tensors share one amax; a grouped linear holds
    one independent matrix per expert.
    """
    get_weight_tensors = module._get_weight_tensors

    def get_fake_quantized_weight_tensors():
        weights = get_weight_tensors()
        if not _enabled:
            return weights
        if per_tensor_amax:
            return [fake_quantize_nvfp4(w, tensor_amax=_amax([w], group)) for w in weights]
        amax = _amax(weights, group)
        return [fake_quantize_nvfp4(w, tensor_amax=amax) for w in weights]

    module._get_weight_tensors = get_fake_quantized_weight_tensors


def _hook_plain_linear(module: nn.Linear, family: Sequence[nn.Linear], group: "dist.ProcessGroup | None") -> None:
    def forward(input: torch.Tensor) -> torch.Tensor:
        if not _enabled:
            return F.linear(input, module.weight, module.bias)
        amax = _amax([member.weight for member in family], group)
        return F.linear(input, fake_quantize_nvfp4(module.weight, tensor_amax=amax), module.bias)

    module.forward = forward


def install_nvfp4_qat(
    model: nn.Module,
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = (),
    amax_group: AmaxGroupResolver = lambda name, weight: None,
) -> list[str]:
    """Install fake-quant hooks on the selected linears and return their module names."""
    targets = {name: module for name, module in model.named_modules() if is_nvfp4_qat_target(name, include, exclude)}
    if not targets:
        raise ValueError(f"NVFP4 QAT matched no modules with include={list(include)} exclude={list(exclude)}.")
    plain = {name: module for name, module in targets.items() if isinstance(module, nn.Linear)}

    # Same family definition as the QAD baseline: projections the serving runtime fuses into one
    # GEMM share one tensor scale. Megatron already stores QKV and gate/up fused, so only the
    # Gated DeltaNet input projections form multi-member families here.
    for family_names in build_nvfp4_fused_family_names(sorted(plain)):
        family = [plain[name] for name in family_names]
        for name, module in zip(family_names, family, strict=True):
            if getattr(module, "_nvfp4_qat_installed", False):
                continue
            _check_block_aligned(name, module.weight)
            _hook_plain_linear(module, family, amax_group(name, module.weight))
            module._nvfp4_qat_installed = True

    for name, module in targets.items():
        if name in plain or getattr(module, "_nvfp4_qat_installed", False):
            continue
        if not hasattr(module, "_get_weight_tensors"):
            raise TypeError(f"NVFP4 QAT selected {name} but {type(module).__name__} is not a supported linear.")
        weights = module._get_weight_tensors()
        grouped = _is_grouped_linear(module)
        if not grouped and len(weights) != 1:
            # Weight sync sees one Megatron parameter per dense linear and could not reproduce
            # an amax shared across several.
            raise ValueError(f"NVFP4 QAT does not support {name}: it holds several weight tensors.")
        for weight in weights:
            _check_block_aligned(name, weight)
        _hook_weight_getter(module, per_tensor_amax=grouped, group=amax_group(name, weights[0]))
        module._nvfp4_qat_installed = True

    return sorted(targets)


@contextmanager
def nvfp4_sync_family_amax(
    named_weights: Iterable[tuple[str, torch.Tensor]],
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = (),
) -> Iterator[None]:
    """Provide multi-member family amaxes to ``nvfp4_qdq_for_sync`` for the duration of one sync.

    Weight sync converts one parameter at a time, but a Gated DeltaNet input projection needs
    the amax of its whole family. The table is built from the weights being synced and dropped
    on exit, so a later sync can never quantize against a stale amax.
    """
    global _sync_family_amax
    weights = {
        _module_name(name): weight for name, weight in named_weights if is_nvfp4_qat_target(name, include, exclude)
    }
    table = {}
    for family in build_nvfp4_fused_family_names(sorted(weights)):
        if len(family) == 1:
            continue
        amax = _amax([weights[member] for member in family])
        table.update(dict.fromkeys(family, amax))
    outer_table, _sync_family_amax = _sync_family_amax, table
    try:
        yield
    finally:
        _sync_family_amax = outer_table


@torch.no_grad()
def nvfp4_qdq_for_sync(
    name: str,
    param: torch.Tensor,
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = (),
) -> torch.Tensor:
    """The weight the rollout engine must load for the Megatron parameter ``name``.

    ``param`` is the full, already TP-gathered tensor. Quantized layers get the same ``Q(w)`` the
    hooked forward uses; everything else passes through.
    """
    if not is_nvfp4_qat_target(name, include, exclude):
        return param

    module_name = _module_name(name)
    if len(nvfp4_fused_family_candidates(module_name)) == 1:
        return fake_quantize_nvfp4(param, tensor_amax=_amax([param]))

    if _sync_family_amax is None or module_name not in _sync_family_amax:
        raise RuntimeError(
            f"{name} shares its NVFP4 tensor scale with sibling projections; convert it inside nvfp4_sync_family_amax(...) built from the weights being synced."
        )
    return fake_quantize_nvfp4(param, tensor_amax=_sync_family_amax[module_name].to(param.device))
