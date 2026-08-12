import unittest

import torch

from recp.methods.prompts import recp_prompts
from recp.methods.ugt import (
    aggregate_text_prototypes,
    jensen_shannon_divergence,
    rectify_evidence,
    reliability_weighted_consistency,
    uncertainty_gate,
)


class UGTTests(unittest.TestCase):
    def test_prompt_order_matches_background_then_tumor(self) -> None:
        self.assertEqual(tuple(recp_prompts()), ("non_tumor", "tumor"))
        self.assertTrue(all(len(prompts) == 3 for prompts in recp_prompts().values()))

    def test_prototype_centroid_shape(self) -> None:
        embeddings = torch.randn(2, 3, 8)
        prototypes = aggregate_text_prototypes(embeddings)
        self.assertEqual(prototypes.shape, (2, 8))
        self.assertTrue(torch.allclose(prototypes.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_gate_decreases_with_cross_modal_divergence(self) -> None:
        p = torch.tensor([[[[0.9]], [[0.1]]]])
        same = jensen_shannon_divergence(p, p)
        opposite = jensen_shannon_divergence(p, p.flip(1))
        uncertainty = torch.tensor([[[[0.5]]]])
        self.assertGreater(
            uncertainty_gate(uncertainty, same, 0.5).item(),
            uncertainty_gate(uncertainty, opposite, 0.5).item(),
        )

    def test_rectification_only_adds_nonnegative_text_evidence(self) -> None:
        alpha_image = torch.full((1, 2, 2, 2), 2.0)
        alpha_text = torch.full((1, 2, 2, 2), 1.5)
        output = rectify_evidence(alpha_image, alpha_text, torch.full((1, 1, 2, 2), 0.5), 0.5)
        self.assertTrue(torch.all(output["posterior_alpha"] >= alpha_image))

    def test_valid_mask_accepts_nhw(self) -> None:
        alpha = torch.full((2, 2, 3, 3), 2.0)
        target = torch.full((2, 2, 3, 3), 0.5)
        reliability = torch.full((2, 1, 3, 3), 0.5)
        mask = torch.ones(2, 3, 3)
        loss = reliability_weighted_consistency(alpha, target, reliability, mask)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

