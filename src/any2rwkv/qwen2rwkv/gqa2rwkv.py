"""Fail-closed boundary for the unresolved Qwen3.5 GQA -> RWKV conversion.

The bounded-hazard, dual-expert, PISA/PWT sidecar, and Hedgehog/H2O attempts
all failed their strict feasibility or layer-output gates. Their executable
product paths were removed so no rejected approximation can emit a checkpoint
or masquerade as a supported runtime.
"""

GQA_REJECTION = (
    "GQA -> RWKV conversion is not implemented: bounded-hazard, dual-expert, "
    "and Hedgehog/H2O approximations were removed after failing their strict "
    "layer-output NMSE gates"
)


def initialize_gqa_layer(*args, **kwargs):
    """Reject GQA conversion before parameters, training, or artifacts change."""

    raise RuntimeError(GQA_REJECTION)


__all__ = ["GQA_REJECTION", "initialize_gqa_layer"]
