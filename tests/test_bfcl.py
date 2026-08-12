import unittest

import torch

from recp.methods.bfcl import (
    boundary_uncertainty_weights,
    dynamic_boundary_mask,
    reliable_anchor_pools,
)


class BFCLTests(unittest.TestCase):
    def test_boundary_marks_class_transition(self) -> None:
        labels = torch.tensor([[[0, 0, 1], [0, 0, 1], [0, 0, 1]]])
        boundary = dynamic_boundary_mask(labels, radius=1)
        self.assertFalse(boundary[0, 0, 0, 0].item())
        self.assertTrue(boundary[0, 0, 1, 1].item())

    def test_anchor_pools_are_reliability_first(self) -> None:
        image_probability = torch.tensor([[[[0.9, 0.55]], [[0.1, 0.45]]]])
        posterior_probability = image_probability.clone()
        pools = reliable_anchor_pools(image_probability, posterior_probability, 0.7)
        self.assertTrue(pools["reliable"][0, 0, 0, 0].item())
        self.assertFalse(pools["reliable"][0, 0, 0, 1].item())

    def test_selected_weights_sum_to_one(self) -> None:
        selected = torch.tensor([[[[True, True]]]])
        boundary = torch.tensor([[[[False, True]]]])
        uncertainty = torch.tensor([[[[0.2, 0.4]]]])
        weights = boundary_uncertainty_weights(selected, boundary, uncertainty)
        self.assertAlmostEqual(weights.sum().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
