"""Dataset-independent checks for labeled and unlabeled batches."""

from typing import Any

import torch


def _validate_image(image: Any, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"{name} must have shape N x 1 x H x W")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return image


def _validate_mask(mask: Any, image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"{name} must have shape N x H x W or N x 1 x H x W")
        canonical_shape = (mask.shape[0], *mask.shape[2:])
    elif mask.ndim == 3:
        canonical_shape = tuple(mask.shape)
    else:
        raise ValueError(f"{name} must have shape N x H x W or N x 1 x H x W")
    expected = (image.shape[0], *image.shape[2:])
    if canonical_shape != expected:
        raise ValueError(f"{name} batch and spatial dimensions must match the image")
    return mask


def validate_labeled_batch(batch: dict[str, Any]) -> None:
    if not {"image", "mask"}.issubset(batch):
        raise ValueError("a labeled batch requires image and mask tensors")
    image = _validate_image(batch["image"], "image")
    _validate_mask(batch["mask"], image, "mask")
    if "valid_mask" in batch:
        _validate_mask(batch["valid_mask"], image, "valid_mask")


def validate_unlabeled_batch(batch: dict[str, Any]) -> None:
    required = {"weak_image", "strong_image", "alignment", "valid_mask"}
    if not required.issubset(batch):
        raise ValueError(f"an unlabeled batch requires {sorted(required)}")
    weak = _validate_image(batch["weak_image"], "weak_image")
    strong = _validate_image(batch["strong_image"], "strong_image")
    if weak.shape != strong.shape:
        raise ValueError("weak_image and strong_image must have identical shapes")
    _validate_mask(batch["valid_mask"], strong, "valid_mask")
