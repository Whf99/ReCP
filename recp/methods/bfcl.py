"""Public candidate construction for Boundary-Focused Contrastive Learning."""

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
) -> dict[str, torch.Tensor]:
    """Return reliable boundary and interior candidate masks.

    The release intentionally leaves the per-batch sampling quota and ranking
    policy unspecified; these masks are the two pools described in the paper.
    """
    confidence = image_probability.max(dim=1, keepdim=True).values
    reliable = confidence > float(confidence_threshold)
    pseudo_labels = posterior_probability.argmax(dim=1)
    boundary = dynamic_boundary_mask(pseudo_labels, radius=radius)
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
