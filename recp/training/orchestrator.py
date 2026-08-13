"""Loss composition for one ReCP teacher--student optimization step."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    ugt: float
    bfcl: float

    def __post_init__(self) -> None:
        if self.ugt < 0.0 or self.bfcl < 0.0:
            raise ValueError("loss weights must be non-negative")


def recp_step(
    student: nn.Module,
    teacher: nn.Module,
    labeled: dict[str, torch.Tensor],
    unlabeled: dict[str, torch.Tensor] | None,
    teer: Callable[..., dict[str, torch.Tensor]],
    ugt: Callable[..., Any],
    bfcl: Callable[..., torch.Tensor],
    align_teacher_to_student: Callable[..., tuple[dict[str, torch.Tensor], torch.Tensor]],
    text_prototypes: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """Compose TEER, UGT, and BFCL in the order defined by ReCP."""
    labeled_output = student(labeled["image"])
    supervised = teer(
        labeled_output["evidence"],
        labeled["mask"],
        valid_mask=labeled.get("valid_mask"),
    )
    details = {
        f"teer_{name}": supervised[name]
        for name in ("dice", "edl", "nll", "kl")
        if name in supervised and supervised[name].ndim == 0
    }

    zero = supervised["loss"].new_zeros(())
    if weights.ugt == 0.0 and weights.bfcl == 0.0:
        return {
            "loss": supervised["loss"],
            "supervised": supervised["loss"],
            "ugt": zero,
            "bfcl": zero,
            **details,
        }
    if unlabeled is None:
        raise ValueError("an unlabeled batch is required after the supervised warm-up")

    student_output = student(unlabeled["strong_image"])
    with torch.no_grad():
        teacher_output = teacher(unlabeled["weak_image"])
        aligned_teacher, valid_mask = align_teacher_to_student(
            teacher_output,
            unlabeled["alignment"],
            unlabeled["valid_mask"],
        )
    target = ugt(
        aligned_teacher["evidence"].detach() + 1.0,
        aligned_teacher["features"].detach(),
        text_prototypes,
        valid_mask,
        detach_target=True,
    )

    consistency = (
        ugt.consistency_loss(student_output["evidence"] + 1.0, target)
        if weights.ugt > 0.0
        else zero
    )
    contrastive = (
        bfcl(student_output["features"], target, valid_mask)
        if weights.bfcl > 0.0
        else zero
    )
    total = (
        supervised["loss"]
        + weights.ugt * consistency
        + weights.bfcl * contrastive
    )
    return {
        "loss": total,
        "supervised": supervised["loss"],
        "ugt": consistency,
        "bfcl": contrastive,
        **details,
    }
