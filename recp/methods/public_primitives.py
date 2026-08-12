"""Limited, equation-level ReCP primitives disclosed during peer review.

Only operations already explicit in the manuscript are included. The complete
loss construction, fusion, sampling strategy, and optimization logic remain in
the restricted research implementation.
"""

import torch
from torch.nn import functional as F


def dirichlet_statistics(
    evidence: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TEER public part: return alpha, class probability, and vacuity K / S."""
    if evidence.ndim != 4 or evidence.shape[1] != num_classes:
        raise ValueError("Expected non-negative evidence with shape N x K x H x W.")
    if torch.any(evidence < 0):
        raise ValueError("Dirichlet evidence must be non-negative.")
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    probability = alpha / strength.clamp_min(1e-8)
    uncertainty = float(num_classes) / strength.clamp_min(1e-8)
    return alpha, probability, uncertainty


def target_exempted_alpha(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TEER public part: alpha_tilde = 1 + (1-y) * (alpha-1)."""
    if labels.ndim == alpha.ndim and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim == alpha.ndim - 1:
        labels = F.one_hot(labels.long(), num_classes=alpha.shape[1]).movedim(-1, 1)
    if labels.shape != alpha.shape:
        raise ValueError("Labels must be class indices or one-hot tensors matching alpha.")
    return 1.0 + (1.0 - labels.to(alpha.dtype)) * (alpha - 1.0)


def jensen_shannon_divergence(
    image_probability: torch.Tensor,
    text_probability: torch.Tensor,
) -> torch.Tensor:
    """UGT public part: pixel-wise JS divergence across the class dimension."""
    eps = torch.finfo(image_probability.dtype).eps
    p = image_probability.clamp_min(eps)
    q = text_probability.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_p = (p * (p.log() - midpoint.log())).sum(dim=1, keepdim=True)
    kl_q = (q * (q.log() - midpoint.log())).sum(dim=1, keepdim=True)
    return 0.5 * (kl_p + kl_q)


def uncertainty_gate(
    visual_uncertainty: torch.Tensor,
    cross_modal_divergence: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """UGT public part: omega = u_img * exp(-gamma * JS)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    return visual_uncertainty.clamp(0.0, 1.0) * torch.exp(-gamma * cross_modal_divergence)


def dynamic_boundary_mask(pseudo_labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """BFCL public part: mark pixels whose neighborhood contains another class."""
    if radius < 1:
        raise ValueError("radius must be at least 1.")
    if pseudo_labels.ndim == 4 and pseudo_labels.shape[1] == 1:
        pseudo_labels = pseudo_labels[:, 0]
    if pseudo_labels.ndim != 3:
        raise ValueError("Expected pseudo labels with shape N x H x W.")
    labels = pseudo_labels.to(torch.float32).unsqueeze(1)
    kernel = 2 * radius + 1
    labels = F.pad(labels, (radius, radius, radius, radius), mode="replicate")
    local_max = F.max_pool2d(labels, kernel, stride=1)
    local_min = -F.max_pool2d(-labels, kernel, stride=1)
    return local_max.ne(local_min)
