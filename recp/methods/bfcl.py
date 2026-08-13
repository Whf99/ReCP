"""Candidate construction for Boundary-Focused Contrastive Learning."""

from typing import Optional

import torch
from torch.nn import functional as F


def dynamic_boundary_mask(pseudo_labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Mark pixels whose local neighborhood contains another pseudo class."""
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if pseudo_labels.ndim == 4 and pseudo_labels.shape[1] == 1:
        pseudo_labels = pseudo_labels[:, 0]
    if pseudo_labels.ndim != 3:
        raise ValueError("pseudo_labels must have shape NHW or N1HW")
    labels = F.pad(
        pseudo_labels.float().unsqueeze(1),
        (radius, radius, radius, radius),
        mode="replicate",
    )
    kernel = 2 * radius + 1
    local_max = F.max_pool2d(labels, kernel, stride=1)
    local_min = -F.max_pool2d(-labels, kernel, stride=1)
    return local_max.ne(local_min)


def reliable_anchor_pools(
    image_probability: torch.Tensor,
    posterior_probability: torch.Tensor,
    confidence_threshold: float = 0.7,
    radius: int = 1,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return reliable boundary and interior candidate masks.

    The returned masks are the two reliability-filtered pools defined by BFCL.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    if image_probability.shape != posterior_probability.shape:
        raise ValueError("image and posterior probabilities must have identical shapes")
    confidence = image_probability.max(dim=1, keepdim=True).values
    reliable = confidence > float(confidence_threshold)
    pseudo_labels = posterior_probability.argmax(dim=1)
    boundary = dynamic_boundary_mask(pseudo_labels, radius=radius)
    if valid_mask is not None:
        valid = valid_mask.to(device=reliable.device, dtype=torch.bool)
        while valid.ndim < reliable.ndim:
            valid = valid.unsqueeze(1)
        valid = torch.broadcast_to(valid, reliable.shape)
        reliable = reliable & valid
        boundary = boundary & valid
    return {
        "pseudo_labels": pseudo_labels,
        "boundary": boundary,
        "reliable": reliable,
        "reliable_boundary": reliable & boundary,
        "reliable_interior": reliable & ~boundary,
    }


def boundary_uncertainty_weights(
    selected_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    visual_uncertainty: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize b(i)+u_img(i) over selected anchors."""
    selected = selected_mask.to(dtype=visual_uncertainty.dtype)
    hardness = boundary_mask.to(dtype=visual_uncertainty.dtype) + visual_uncertainty
    weighted = selected * hardness
    return weighted / (weighted.sum() + eps)
