"""Dataset-independent batch checks; no private preprocessing is included."""

import torch


def validate_labeled_batch(batch: dict[str, torch.Tensor]) -> None:
    if not {"image", "mask"}.issubset(batch):
        raise ValueError("A labeled batch requires image and mask tensors.")
    if batch["image"].ndim != 4 or batch["mask"].ndim not in (3, 4):
        raise ValueError("Expected 2D batched tensors in NCHW/NHW form.")


def validate_unlabeled_batch(batch: dict[str, torch.Tensor]) -> None:
    required = {"weak_image", "strong_image", "alignment", "valid_mask"}
    if not required.issubset(batch):
        raise ValueError(f"An unlabeled batch requires {sorted(required)}.")
