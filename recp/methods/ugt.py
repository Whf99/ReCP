"""Public mathematical path of Uncertainty-Gated Text Prior Injection."""

from typing import Optional

import torch
from torch.nn import functional as F


def aggregate_text_prototypes(prompt_embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize prompts, average the three anchors per class, then normalize again.

    Expected shape is K x M x D, where K is the class count and M=3 in ReCP.
    The frozen text encoder itself and pretrained weights are not distributed.
    """
    if prompt_embeddings.ndim != 3 or prompt_embeddings.shape[1] != 3:
        raise ValueError("prompt_embeddings must have shape K x 3 x D")
    normalized = F.normalize(prompt_embeddings.float(), dim=-1)
    return F.normalize(normalized.mean(dim=1), dim=-1)


def text_pseudo_alpha(
    projected_features: torch.Tensor,
    text_prototypes: torch.Tensor,
    text_scale: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map pixel/prototype cosine scores to alpha_text=Softplus(tau*s)+1."""
    if projected_features.ndim != 4 or text_prototypes.ndim != 2:
        raise ValueError("expected projected features NDHW and prototypes KD")
    if projected_features.shape[1] != text_prototypes.shape[1]:
        raise ValueError("feature and prototype embedding dimensions must match")
    features = F.normalize(projected_features, dim=1)
    prototypes = F.normalize(text_prototypes, dim=1)
    similarity = torch.einsum("ndhw,kd->nkhw", features, prototypes)
    alpha_text = F.softplus(float(text_scale) * similarity) + 1.0
    return alpha_text, similarity


def jensen_shannon_divergence(
    image_probability: torch.Tensor,
    text_probability: torch.Tensor,
) -> torch.Tensor:
    """Pixel-wise JS divergence over the class dimension."""
    if image_probability.shape != text_probability.shape:
        raise ValueError("image and text probabilities must have identical shapes")
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
    """Compute omega=u_img*exp(-gamma*d)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return visual_uncertainty.clamp(0.0, 1.0) * torch.exp(
        -float(gamma) * cross_modal_divergence
    )


def rectify_evidence(
    alpha_image: torch.Tensor,
    alpha_text: torch.Tensor,
    visual_uncertainty: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Compute the UGT posterior alpha_img+omega*(alpha_text-1)."""
    image_probability = alpha_image / alpha_image.sum(dim=1, keepdim=True).clamp_min(1e-8)
    text_probability = alpha_text / alpha_text.sum(dim=1, keepdim=True).clamp_min(1e-8)
    divergence = jensen_shannon_divergence(image_probability, text_probability)
    gate = uncertainty_gate(visual_uncertainty, divergence, gamma)
    posterior_alpha = alpha_image + gate * (alpha_text - 1.0)
    posterior_probability = posterior_alpha / posterior_alpha.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    return {
        "posterior_alpha": posterior_alpha,
        "posterior_probability": posterior_probability,
        "divergence": divergence,
        "gate": gate,
        "reliability": 1.0 - visual_uncertainty.clamp(0.0, 1.0),
    }


def reliability_weighted_consistency(
    student_alpha: torch.Tensor,
    target_probability: torch.Tensor,
    reliability: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reliability-weighted cross-entropy on geometrically valid pixels."""
    student_probability = student_alpha / student_alpha.sum(
        dim=1, keepdim=True
    ).clamp_min(eps)
    target = target_probability.detach()
    cross_entropy = -(target * student_probability.clamp_min(eps).log()).sum(
        dim=1, keepdim=True
    )
    weight = reliability
    if valid_mask is not None:
        mask = valid_mask.to(device=weight.device, dtype=weight.dtype)
        while mask.ndim < weight.ndim:
            mask = mask.unsqueeze(1)
        weight = weight * torch.broadcast_to(mask, weight.shape)
    return (weight * cross_entropy).sum() / (weight.sum() + eps)
