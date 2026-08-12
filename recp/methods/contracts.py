"""Public contracts for method-specific components withheld during review."""

from dataclasses import dataclass
from typing import Protocol

import torch


class RestrictedComponentError(RuntimeError):
    pass


@dataclass(frozen=True)
class UGTOutput:
    posterior_alpha: torch.Tensor
    visual_uncertainty: torch.Tensor
    reliability: torch.Tensor
    valid_mask: torch.Tensor


class TEER(Protocol):
    """Paper Sec. 3.2: evidence/labels -> supervised evidential loss and uncertainty."""

    def __call__(self, evidence: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]: ...


class UGT(Protocol):
    """Paper Sec. 3.3: aligned teacher evidence/features plus prototypes -> rectified target."""

    def __call__(
        self,
        teacher_evidence: torch.Tensor,
        teacher_features: torch.Tensor,
        text_prototypes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> UGTOutput: ...

    def consistency_loss(
        self,
        student_evidence: torch.Tensor,
        target: UGTOutput,
    ) -> torch.Tensor: ...


class BFCL(Protocol):
    """Paper Sec. 3.4: reliable pseudo-targets/features -> boundary-focused contrastive loss."""

    def __call__(self, features: torch.Tensor, target: UGTOutput) -> torch.Tensor: ...


def restricted_component(name: str) -> None:
    raise RestrictedComponentError(
        f"{name} is intentionally excluded from the review-stage partial release. "
        "See docs/RELEASE_SCOPE.md."
    )
