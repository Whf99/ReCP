"""Compute patient-level overlap and surface metrics for ReCP predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recp.evaluation import (
    load_manifest,
    load_volume,
    patient_metrics,
    summarize_patient_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary NPC segmentations at patient level."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to floating-point prediction volumes.",
    )
    parser.add_argument(
        "--surface-tolerance-mm",
        type=float,
        default=1.0,
        help="Surface Dice tolerance in millimeters (default: 1.0).",
    )
    parser.add_argument(
        "--prediction-key",
        help="Array name when prediction files are multi-array NPZ archives.",
    )
    parser.add_argument(
        "--target-key",
        help="Array name when target files are multi-array NPZ archives.",
    )
    return parser.parse_args()


def binarize(array: np.ndarray, threshold: float, *, is_prediction: bool) -> np.ndarray:
    array = np.asarray(array)
    if not np.all(np.isfinite(array)):
        raise ValueError("volume contains NaN or infinite values")
    if is_prediction and np.issubdtype(array.dtype, np.floating):
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(
                "floating-point predictions must contain probabilities in [0, 1]"
            )
        return array >= threshold
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(
            f"binary label volume contains values outside {{0, 1}}: {unique[:8]}"
        )
    return array.astype(bool, copy=False)


def evaluate_case(
    case: Any,
    threshold: float,
    tolerance_mm: float,
    prediction_key: str | None,
    target_key: str | None,
) -> dict[str, Any]:
    prediction_array, prediction_spacing = load_volume(case.prediction, prediction_key)
    target_array, target_spacing = load_volume(case.target, target_key)
    prediction = binarize(prediction_array, threshold, is_prediction=True)
    target = binarize(target_array, threshold, is_prediction=False)

    spacing = case.spacing_mm or target_spacing or prediction_spacing
    if spacing is None:
        raise ValueError(
            f"case {case.case_id!r} has no physical spacing; add spacing_mm to the manifest"
        )
    if target_spacing is not None and prediction_spacing is not None:
        if not np.allclose(target_spacing, prediction_spacing, rtol=1e-5, atol=1e-6):
            raise ValueError(f"NIfTI spacing differs for case {case.case_id!r}")
    if case.spacing_mm is not None:
        for source, header_spacing in (
            ("prediction", prediction_spacing),
            ("target", target_spacing),
        ):
            if header_spacing is not None and not np.allclose(
                case.spacing_mm,
                header_spacing,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"manifest and {source} spacing differ for case {case.case_id!r}"
                )

    metrics = patient_metrics(prediction, target, spacing, tolerance_mm)
    return {
        "case_id": case.case_id,
        "shape": list(prediction.shape),
        "spacing_mm": [float(value) for value in spacing],
        **metrics,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.surface_tolerance_mm < 0:
        raise ValueError("surface-tolerance-mm must be non-negative")

    cases = load_manifest(args.manifest)
    patient_results = [
        evaluate_case(
            case,
            threshold=args.threshold,
            tolerance_mm=args.surface_tolerance_mm,
            prediction_key=args.prediction_key,
            target_key=args.target_key,
        )
        for case in cases
    ]
    summary = summarize_patient_metrics(patient_results)
    report = {
        "number_of_patients": len(patient_results),
        "prediction_threshold": float(args.threshold),
        "surface_tolerance_mm": float(args.surface_tolerance_mm),
        "summary": summary,
        "patients": patient_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for name, statistics in summary.items():
        print(f"{name}: {statistics['mean']:.4f} +/- {statistics['std']:.4f}")
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
