import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from recp.training import ExperimentBundle, ReCPTrainer, recp_loss_weights


class _TrainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = torch.nn.functional.softplus(self.head(image))
        return {"evidence": evidence, "features": image}


class TrainingEngineTests(unittest.TestCase):
    def test_supervised_warmup_updates_student_and_teacher(self) -> None:
        student = _TrainableModel()
        teacher = _TrainableModel()
        teacher.load_state_dict(student.state_dict())
        image = torch.ones(1, 1, 4, 4)
        labeled_loader = [{"image": image, "mask": torch.zeros(1, 4, 4, dtype=torch.long)}]

        def teer(evidence, labels, valid_mask=None):
            del labels, valid_mask
            return {"loss": evidence.mean()}

        def unused(*args, **kwargs):
            raise AssertionError("unsupervised callbacks must not run during warm-up")

        def validate(model, modules, device):
            del model, modules, device
            return {"dice": 0.25}

        bundle = ExperimentBundle(
            labeled_loader=labeled_loader,
            unlabeled_loader=[],
            steps_per_epoch=1,
            teer=teer,
            ugt=unused,
            bfcl=unused,
            align_teacher_to_student=unused,
            text_prototypes=torch.nn.functional.normalize(torch.ones(2, 4), dim=1),
            validate=validate,
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        initial_student = student.head.weight.detach().clone()
        initial_teacher = teacher.head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            trainer = ReCPTrainer(
                student,
                teacher,
                bundle,
                optimizer,
                torch.device("cpu"),
                Path(directory),
                ema_decay=0.99,
            )
            weights = recp_loss_weights(0, 15, 300, 1.0, 0.3)
            losses = trainer.train_epoch(0, weights)
            self.assertIn("loss", losses)
            self.assertFalse(torch.equal(initial_student, student.head.weight))
            self.assertFalse(torch.equal(initial_teacher, teacher.head.weight))


if __name__ == "__main__":
    unittest.main()
