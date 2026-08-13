"""Mathematical operators for Uncertainty-Gated Text Prior Injection."""

from typing import Optional

import torch
from torch.nn import functional as F

from .contracts import UGTOutput


def aggregate_text_prototypes(prompt_embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize prompts, average the three anchors per class, then normalize again.

    Expected shape is K x M x D, where K is the class count and M=3 in ReCP.
    Text embeddings are computed once with the frozen text encoder.
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
    if text_scale <= 0:
        raise ValueError("text_scale must be positive")
    features = F.normalize(projected_features, dim=1)
    prototypes = F.normalize(
        text_prototypes.to(
            device=projected_features.device,
            dtype=projected_features.dtype,
        ),
        dim=1,
    )
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
        -float(gamma) * cross_modal_divergence.clamp_min(0.0)
    )


def rectify_evidence(
    alpha_image: torch.Tensor,
    alpha_text: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Compute the UGT posterior alpha_img+omega*(alpha_text-1)."""
    if alpha_image.ndim != 4 or alpha_image.shape != alpha_text.shape:
        raise ValueError(
            "alpha_image and alpha_text must have identical N x K x H x W shapes"
        )
    if not torch.isfinite(alpha_image).all() or not torch.isfinite(alpha_text).all():
        raise ValueError("Dirichlet parameters must be finite")
    if torch.any(alpha_image < 1.0) or torch.any(alpha_text < 1.0):
        raise ValueError("Dirichlet parameters must be at least one")
    image_strength = alpha_image.sum(dim=1, keepdim=True).clamp_min(1e-8)
    text_strength = alpha_text.sum(dim=1, keepdim=True).clamp_min(1e-8)
    image_probability = alpha_image / image_strength
    text_probability = alpha_text / text_strength
    visual_uncertainty = float(alpha_image.shape[1]) / image_strength
    divergence = jensen_shannon_divergence(image_probability, text_probability)
    gate = uncertainty_gate(visual_uncertainty, divergence, gamma)
    posterior_alpha = alpha_image + gate * (alpha_text - 1.0)
    posterior_probability = posterior_alpha / posterior_alpha.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    return {
        "posterior_alpha": posterior_alpha,
        "posterior_probability": posterior_probability,
        "image_probability": image_probability,
        "text_probability": text_probability,
        "divergence": divergence,
        "gate": gate,
        "visual_uncertainty": visual_uncertainty,
        "reliability": 1.0 - visual_uncertainty.clamp(0.0, 1.0),
    }


def ugt_target_from_projected_features(
    alpha_image: torch.Tensor,
    projected_features: torch.Tensor,
    text_prototypes: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 0.5,
    text_scale: float = 0.07,
    *,
    detach_target: bool = True,
) -> UGTOutput:
    """Construct the aligned UGT pseudo-target from projected visual features."""
    alpha_text, _ = text_pseudo_alpha(
        projected_features,
        text_prototypes,
        text_scale=text_scale,
    )
    rectified = rectify_evidence(alpha_image, alpha_text, gamma=gamma)
    values = {
        name: rectified[name]
        for name in (
            "posterior_alpha",
            "posterior_probability",
            "image_probability",
            "visual_uncertainty",
            "reliability",
        )
    }
    if detach_target:
        values = {name: value.detach() for name, value in values.items()}
    return UGTOutput(valid_mask=valid_mask.bool(), **values)


def reliability_weighted_consistency(
    student_alpha: torch.Tensor,
    target_probability: torch.Tensor,
    reliability: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reliability-weighted cross-entropy on geometrically valid pixels."""
    if student_alpha.shape != target_probability.shape:
        raise ValueError("student alpha and target probability shapes must match")
    if reliability.ndim != student_alpha.ndim or reliability.shape[1] != 1:
        raise ValueError("reliability must have shape N x 1 x H x W")
    if (
        reliability.shape[0] != student_alpha.shape[0]
        or reliability.shape[2:] != student_alpha.shape[2:]
    ):
        raise ValueError("reliability batch and spatial dimensions must match")
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
