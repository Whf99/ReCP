"""Interfaces connecting ReCP method components to the training engine."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class UGTOutput:
    """Aligned UGT target and the visual quantities used by BFCL."""

    posterior_alpha: torch.Tensor
    posterior_probability: torch.Tensor
    image_probability: torch.Tensor
    visual_uncertainty: torch.Tensor
    reliability: torch.Tensor
    valid_mask: torch.Tensor


class TEER(Protocol):
    """Paper Sec. 3.2: evidence/labels -> supervised evidential loss and uncertainty."""

    def __call__(
        self,
        evidence: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]: ...


class UGT(Protocol):
    """Paper Sec. 3.3: aligned teacher evidence/features plus prototypes -> rectified target."""

    def __call__(
        self,
        teacher_alpha: torch.Tensor,
        teacher_features: torch.Tensor,
        text_prototypes: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        detach_target: bool = True,
    ) -> UGTOutput: ...

    def consistency_loss(
        self,
        student_alpha: torch.Tensor,
        target: UGTOutput,
    ) -> torch.Tensor: ...


class BFCL(Protocol):
    """Paper Sec. 3.4: reliable pseudo-targets/features -> boundary-focused contrastive loss."""

    def __call__(
        self,
        features: torch.Tensor,
        target: UGTOutput,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor: ...
