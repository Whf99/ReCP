"""Public, equation-level implementation of TEER from the ReCP manuscript."""

from typing import Optional

import torch
from torch.nn import functional as F


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim == 3:
        labels = F.one_hot(labels.long(), num_classes=num_classes).movedim(-1, 1)
    if labels.ndim != 4 or labels.shape[1] != num_classes:
        raise ValueError("labels must be NHW, N1HW, or one-hot NKHW tensors")
    return labels.to(dtype=torch.float32)


def _masked_mean(value: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return value.mean()
    mask = valid_mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, value.shape)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def dirichlet_statistics(
    evidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return alpha=e+1, pi=alpha/S, and vacuity u=K/S."""
    if evidence.ndim != 4:
        raise ValueError("evidence must have shape NKHW")
    if torch.any(evidence < 0):
        raise ValueError("Dirichlet evidence must be non-negative")
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    probability = alpha / strength.clamp_min(1e-8)
    uncertainty = float(alpha.shape[1]) / strength.clamp_min(1e-8)
    return alpha, probability, uncertainty


def target_exempted_alpha(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Construct alpha_tilde=1+(1-y)*(alpha-1), as defined in TEER."""
    target = _one_hot(labels, alpha.shape[1]).to(device=alpha.device, dtype=alpha.dtype)
    if target.shape != alpha.shape:
        raise ValueError("labels and alpha must share batch and spatial dimensions")
    return 1.0 + (1.0 - target) * (alpha - 1.0)


def evidential_nll(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-hot Dirichlet NLL: log(S)-log(alpha_y)."""
    target = _one_hot(labels, alpha.shape[1]).to(device=alpha.device, dtype=alpha.dtype)
    strength = alpha.sum(dim=1, keepdim=True)
    per_pixel = torch.log(strength.clamp_min(1e-8)) - (
        target * torch.log(alpha.clamp_min(1e-8))
    ).sum(dim=1, keepdim=True)
    return _masked_mean(per_pixel, valid_mask)


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Pixel-wise KL(Dir(alpha) || Dir(1))."""
    num_classes = alpha.shape[1]
    strength = alpha.sum(dim=1, keepdim=True)
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(alpha.new_tensor(float(num_classes)))
    )
    expected_log = torch.digamma(alpha) - torch.digamma(strength)
    return log_normalizer + ((alpha - 1.0) * expected_log).sum(dim=1, keepdim=True)


def soft_dice_loss(
    probability: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Class-averaged soft Dice loss used in the supervised objective."""
    target = _one_hot(labels, probability.shape[1]).to(
        device=probability.device, dtype=probability.dtype
    )
    dims = (0, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def teer_supervised_loss(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    kl_weight: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute L_sup=L_dice+L_NLL+lambda_KL*KL(Dir(alpha_tilde)||Dir(1))."""
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    alpha, probability, uncertainty = dirichlet_statistics(evidence)
    exempted = target_exempted_alpha(alpha, labels)
    dice = soft_dice_loss(probability, labels)
    nll = evidential_nll(alpha, labels, valid_mask)
    kl = _masked_mean(dirichlet_kl_to_uniform(exempted), valid_mask)
    edl = nll + float(kl_weight) * kl
    return {
        "loss": dice + edl,
        "dice": dice,
        "edl": edl,
        "nll": nll,
        "kl": kl,
        "probability": probability,
        "uncertainty": uncertainty,
    }

