import unittest

import torch
from torch import nn

from recp.methods.contracts import UGTOutput
from recp.training import LossWeights, recp_step


class _SyntheticModel(nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = image.shape
        evidence = torch.full((batch, 2, height, width), 2.0)
        features = torch.ones(batch, 4, height, width)
        return {"evidence": evidence, "features": features}


class _SyntheticUGT:
    def __init__(self) -> None:
        self.teacher_alpha: torch.Tensor | None = None
        self.student_alpha: torch.Tensor | None = None

    def __call__(
        self,
        teacher_alpha: torch.Tensor,
        teacher_features: torch.Tensor,
        text_prototypes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> UGTOutput:
        self.teacher_alpha = teacher_alpha
        reliability = torch.ones_like(valid_mask)
        return UGTOutput(teacher_alpha, teacher_alpha.reciprocal(), reliability, valid_mask)

    def consistency_loss(
        self,
        student_alpha: torch.Tensor,
        target: UGTOutput,
    ) -> torch.Tensor:
        self.student_alpha = student_alpha
        return student_alpha.mean() * 0.0


class OrchestratorTests(unittest.TestCase):
    def test_ugt_receives_alpha_not_raw_evidence(self) -> None:
        model = _SyntheticModel()
        ugt = _SyntheticUGT()
        image = torch.zeros(1, 1, 4, 4)
        labeled = {"image": image, "mask": torch.zeros(1, 1, 4, 4)}
        unlabeled = {
            "weak_image": image,
            "strong_image": image,
            "alignment": torch.eye(3).unsqueeze(0),
            "valid_mask": torch.ones(1, 1, 4, 4),
        }

        def teer(evidence: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"loss": evidence.mean() * 0.0}

        def align(
            output: dict[str, torch.Tensor],
            alignment: torch.Tensor,
            valid_mask: torch.Tensor,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            return output, valid_mask

        def bfcl(features: torch.Tensor, target: UGTOutput) -> torch.Tensor:
            return features.mean() * 0.0

        recp_step(
            model,
            model,
            labeled,
            unlabeled,
            teer,
            ugt,
            bfcl,
            align,
            torch.ones(2, 4),
            LossWeights(ugt=1.0, bfcl=0.3),
        )
        self.assertTrue(torch.equal(ugt.teacher_alpha, torch.full((1, 2, 4, 4), 3.0)))
        self.assertTrue(torch.equal(ugt.student_alpha, torch.full((1, 2, 4, 4), 3.0)))


if __name__ == "__main__":
    unittest.main()
