#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Modified for ARIA in 2026.
# Copyright (C) 2026 Yiheng Han (ARIA modifications only).
#

from .loss import (
    DPOLoss,
    GPTLMLoss,
    KDLoss,
    KTOLoss,
    LogExpLoss,
    PairWiseLoss,
    PolicyLoss,
    PRMLoss,
    SFTLoss,
    ValueLoss,
    VanillaKTOLoss,
)


def __getattr__(name):
    """Keep optional FlashAttention/RLHF imports out of the core ARIA path."""
    if name == "Actor":
        from .actor import Actor

        return Actor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Actor",
    "SFTLoss",
    "DPOLoss",
    "GPTLMLoss",
    "KDLoss",
    "KTOLoss",
    "LogExpLoss",
    "PairWiseLoss",
    "PolicyLoss",
    "PRMLoss",
    "ValueLoss",
    "VanillaKTOLoss",
]
