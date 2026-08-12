import unittest

import torch

from recp.methods.teer import (
    dirichlet_kl_to_uniform,
    dirichlet_statistics,
    target_exempted_alpha,
    teer_supervised_loss,
)


class TEERTests(unittest.TestCase):
    def test_dirichlet_statistics_follow_manuscript(self) -> None:
        evidence = torch.tensor([[[[2.0]], [[0.0]]]])
        alpha, probability, uncertainty = dirichlet_statistics(evidence)
        self.assertTrue(torch.equal(alpha, torch.tensor([[[[3.0]], [[1.0]]]])))
        self.assertAlmostEqual(probability[0, 0, 0, 0].item(), 0.75)
        self.assertAlmostEqual(uncertainty[0, 0, 0, 0].item(), 0.5)

    def test_target_class_is_exempted_from_kl(self) -> None:
        alpha = torch.tensor([[[[5.0]], [[3.0]]]])
        exempted = target_exempted_alpha(alpha, torch.tensor([[[[0]]]]))
        self.assertEqual(exempted[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(exempted[0, 1, 0, 0].item(), 3.0)

    def test_uniform_dirichlet_has_zero_kl(self) -> None:
        alpha = torch.ones(2, 2, 3, 3)
        self.assertTrue(torch.allclose(dirichlet_kl_to_uniform(alpha), torch.zeros(2, 1, 3, 3)))

    def test_supervised_loss_is_finite(self) -> None:
        evidence = torch.ones(2, 2, 4, 4)
        labels = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        output = teer_supervised_loss(evidence, labels, kl_weight=0.1)
        self.assertTrue(torch.isfinite(output["loss"]))


if __name__ == "__main__":
    unittest.main()

