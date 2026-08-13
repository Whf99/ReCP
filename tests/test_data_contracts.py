import unittest

import torch

from recp.data import validate_labeled_batch, validate_unlabeled_batch


class DataContractTests(unittest.TestCase):
    def test_labeled_contract_accepts_nhw_mask(self) -> None:
        validate_labeled_batch(
            {
                "image": torch.zeros(2, 1, 8, 8),
                "mask": torch.zeros(2, 8, 8, dtype=torch.long),
            }
        )

    def test_labeled_contract_rejects_spatial_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            validate_labeled_batch(
                {
                    "image": torch.zeros(2, 1, 8, 8),
                    "mask": torch.zeros(2, 7, 8, dtype=torch.long),
                }
            )

    def test_unlabeled_contract_checks_both_views(self) -> None:
        image = torch.zeros(2, 1, 8, 8)
        validate_unlabeled_batch(
            {
                "weak_image": image,
                "strong_image": image.clone(),
                "alignment": {"kind": "identity"},
                "valid_mask": torch.ones(2, 1, 8, 8, dtype=torch.bool),
            }
        )


if __name__ == "__main__":
    unittest.main()
