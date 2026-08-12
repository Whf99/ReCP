"""Public high-level ReCP step showing manuscript-aligned component ordering."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    ugt: float
    bfcl: float


def recp_step(
    student: nn.Module,
    teacher: nn.Module,
    labeled: dict[str, torch.Tensor],
    unlabeled: dict[str, torch.Tensor],
    teer: Callable[..., dict[str, torch.Tensor]],
    ugt: Callable[..., Any],
    bfcl: Callable[..., torch.Tensor],
    align_teacher_to_student: Callable[..., tuple[dict[str, torch.Tensor], torch.Tensor]],
    text_prototypes: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """Compose TEER -> UGT -> BFCL without disclosing their restricted internals."""
    labeled_output = student(labeled["image"])
    supervised = teer(labeled_output["evidence"], labeled["mask"])

    student_output = student(unlabeled["strong_image"])
    with torch.no_grad():
        teacher_output = teacher(unlabeled["weak_image"])
        aligned_teacher, valid_mask = align_teacher_to_student(
            teacher_output,
            unlabeled["alignment"],
            unlabeled["valid_mask"],
        )
        target = ugt(
            aligned_teacher["evidence"],
            aligned_teacher["features"],
            text_prototypes,
            valid_mask,
        )

    # The restricted UGT implementation supplies its reliability-weighted loss.
    consistency = ugt.consistency_loss(student_output["evidence"], target)
    contrastive = bfcl(student_output["features"], target)
    total = supervised["loss"] + weights.ugt * consistency + weights.bfcl * contrastive
    return {"loss": total, "supervised": supervised["loss"], "ugt": consistency, "bfcl": contrastive}
