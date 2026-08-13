import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import evaluate_case
from recp.evaluation import EvaluationCase, patient_metrics, summarize_patient_metrics


class PatientMetricTests(unittest.TestCase):
    def test_perfect_volume_has_ideal_metrics(self) -> None:
        target = np.zeros((5, 8, 8), dtype=np.uint8)
        target[1:4, 2:6, 2:6] = 1
        metrics = patient_metrics(target, target, spacing_mm=(2.5, 1.0, 1.0))
        self.assertEqual(metrics["hd95_mm"], 0.0)
        self.assertEqual(metrics["assd_mm"], 0.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["jaccard"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["surface_dice"], 1.0)

    def test_one_voxel_translation_uses_physical_spacing(self) -> None:
        target = np.zeros((7, 7, 7), dtype=np.uint8)
        prediction = np.zeros_like(target)
        target[2:5, 2:5, 2:5] = 1
        prediction[3:6, 2:5, 2:5] = 1
        metrics = patient_metrics(
            prediction,
            target,
            spacing_mm=(2.0, 1.0, 1.0),
            tolerance_mm=1.0,
        )
        self.assertAlmostEqual(metrics["hd95_mm"], 2.0)
        self.assertGreater(metrics["assd_mm"], 0.0)
        self.assertLess(metrics["surface_dice"], 1.0)

    def test_case_evaluation_loads_probability_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = np.zeros((4, 6, 6), dtype=np.uint8)
            target[:, 2:4, 2:4] = 1
            prediction = target.astype(np.float32) * 0.8
            prediction_path = root / "prediction.npy"
            target_path = root / "target.npy"
            np.save(prediction_path, prediction)
            np.save(target_path, target)
            case = EvaluationCase(
                case_id="patient-a",
                prediction=prediction_path,
                target=target_path,
                spacing_mm=(3.0, 1.0, 1.0),
            )
            result = evaluate_case(case, 0.5, 1.0, None, None)
            self.assertEqual(result["case_id"], "patient-a")
            self.assertEqual(result["dice"], 1.0)

    def test_summary_uses_sample_standard_deviation(self) -> None:
        case_a = {name: 0.0 for name in (
            "dice", "jaccard", "precision", "recall", "hd95_mm", "assd_mm", "surface_dice"
        )}
        case_b = {name: 1.0 for name in case_a}
        summary = summarize_patient_metrics([case_a, case_b])
        self.assertEqual(summary["dice"]["mean"], 0.5)
        self.assertAlmostEqual(summary["dice"]["std"], 2.0 ** -0.5)

    def test_single_empty_mask_is_not_a_perfect_detection(self) -> None:
        prediction = np.zeros((3, 4, 4), dtype=np.uint8)
        target = np.zeros_like(prediction)
        target[1, 1:3, 1:3] = 1
        metrics = patient_metrics(prediction, target, spacing_mm=(2.0, 1.0, 1.0))
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
