import unittest

import torch

from recp.methods.public_primitives import (
    dirichlet_statistics,
    dynamic_boundary_mask,
    jensen_shannon_divergence,
    target_exempted_alpha,
    uncertainty_gate,
)


class PublicPrimitiveTests(unittest.TestCase):
    def test_dirichlet_statistics_follow_alpha_equals_e_plus_one(self) -> None:
        evidence = torch.tensor([[[[2.0]], [[0.0]]]])
        alpha, probability, uncertainty = dirichlet_statistics(evidence, num_classes=2)
        self.assertTrue(torch.equal(alpha, torch.tensor([[[[3.0]], [[1.0]]]])))
        self.assertAlmostEqual(probability[0, 0, 0, 0].item(), 0.75)
        self.assertAlmostEqual(uncertainty[0, 0, 0, 0].item(), 0.5)

    def test_teer_exempts_target_evidence(self) -> None:
        alpha = torch.tensor([[[[5.0]], [[3.0]]]])
        exempted = target_exempted_alpha(alpha, torch.tensor([[[[0]]]]))
        self.assertEqual(exempted[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(exempted[0, 1, 0, 0].item(), 3.0)

    def test_ugt_gate_decreases_with_divergence(self) -> None:
        p = torch.tensor([[[[0.9]], [[0.1]]]])
        q_same = p.clone()
        q_other = torch.tensor([[[[0.1]], [[0.9]]]])
        same = uncertainty_gate(torch.tensor([[[[0.5]]]]), jensen_shannon_divergence(p, q_same), 0.5)
        other = uncertainty_gate(torch.tensor([[[[0.5]]]]), jensen_shannon_divergence(p, q_other), 0.5)
        self.assertGreater(same.item(), other.item())

    def test_bfcl_boundary_marks_class_transition(self) -> None:
        labels = torch.tensor([[[0, 0, 1], [0, 0, 1], [0, 0, 1]]])
        boundary = dynamic_boundary_mask(labels, radius=1)
        self.assertFalse(boundary[0, 0, 0, 0].item())
        self.assertTrue(boundary[0, 0, 1, 1].item())


if __name__ == "__main__":
    unittest.main()
