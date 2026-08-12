import unittest

import torch

from recp.evaluation import region_metrics
from recp.training import recp_loss_weights


class ScheduleAndMetricTests(unittest.TestCase):
    def test_warmup_disables_unsupervised_losses(self) -> None:
        weights = recp_loss_weights(14, 15, 300, 1.0, 0.3)
        self.assertEqual(weights.ugt, 0.0)
        self.assertEqual(weights.bfcl, 0.0)

    def test_peak_weights_match_manuscript_defaults(self) -> None:
        weights = recp_loss_weights(300, 15, 300, 1.0, 0.3)
        self.assertAlmostEqual(weights.ugt, 1.0)
        self.assertAlmostEqual(weights.bfcl, 0.3)

    def test_perfect_region_metrics_are_one(self) -> None:
        mask = torch.tensor([[[0, 1], [0, 1]]])
        metrics = region_metrics(mask, mask)
        for value in metrics.values():
            self.assertTrue(torch.allclose(value, torch.ones_like(value)))


if __name__ == "__main__":
    unittest.main()
