"""Inspect ReCP equation-level operators with generated tensors."""

import torch

from recp.methods import (
    aggregate_text_prototypes,
    dirichlet_statistics,
    dynamic_boundary_mask,
    recp_prompts,
    rectify_evidence,
    text_pseudo_alpha,
)


def main() -> None:
    torch.manual_seed(7)
    evidence = torch.rand(1, 2, 8, 8)
    alpha_image, probability_image, _ = dirichlet_statistics(evidence)

    prompt_embeddings = torch.randn(2, 3, 16)
    text_prototypes = aggregate_text_prototypes(prompt_embeddings)
    projected_features = torch.randn(1, 16, 8, 8)
    alpha_text, _ = text_pseudo_alpha(projected_features, text_prototypes)

    rectified = rectify_evidence(alpha_image, alpha_text, gamma=0.5)
    pseudo_labels = rectified["posterior_probability"].argmax(dim=1)
    boundary = dynamic_boundary_mask(pseudo_labels)

    print(f"prompts: {recp_prompts()}")
    print(f"image probability: {tuple(probability_image.shape)}")
    print(f"rectified alpha: {tuple(rectified['posterior_alpha'].shape)}")
    print(f"boundary pixels: {int(boundary.sum())}")


if __name__ == "__main__":
    main()
