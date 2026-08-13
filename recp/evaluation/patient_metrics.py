"""Patient-level overlap and surface metrics for binary segmentation volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


METRIC_NAMES = (
    "dice",
    "jaccard",
    "precision",
    "recall",
    "hd95_mm",
    "assd_mm",
    "surface_dice",
)


def _as_binary_pair(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim not in (2, 3):
        raise ValueError("patient masks must be 2D or 3D arrays")
    return prediction, target


def _validate_spacing(spacing_mm: Sequence[float], ndim: int) -> tuple[float, ...]:
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != ndim:
        raise ValueError(f"spacing_mm must contain {ndim} values")
    if not np.all(np.isfinite(spacing)) or any(value <= 0 for value in spacing):
        raise ValueError("spacing_mm values must be finite and positive")
    return spacing


def overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return Dice, Jaccard, precision, and recall for one patient."""
    prediction, target = _as_binary_pair(prediction, target)
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))

    pred_count = true_positive + false_positive
    target_count = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    both_empty = pred_count == 0 and target_count == 0
    return {
        "dice": 1.0 if both_empty else 2.0 * true_positive / (pred_count + target_count),
        "jaccard": 1.0 if union == 0 else true_positive / union,
        "precision": (
            1.0 if both_empty else 0.0 if pred_count == 0 else true_positive / pred_count
        ),
        "recall": (
            1.0 if both_empty else 0.0 if target_count == 0 else true_positive / target_count
        ),
    }


def surface_mask(mask: np.ndarray) -> np.ndarray:
    """Extract the one-voxel inner surface using face connectivity."""
    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def directed_surface_distances(
    source_surface: np.ndarray,
    target_surface: np.ndarray,
    spacing_mm: Sequence[float],
) -> np.ndarray:
    """Distance in millimeters from each source-surface point to the target surface."""
    spacing = _validate_spacing(spacing_mm, source_surface.ndim)
    if not np.any(source_surface) or not np.any(target_surface):
        raise ValueError("directed surface distance requires two non-empty surfaces")
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return np.asarray(distance_map[source_surface], dtype=np.float64)


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: Sequence[float],
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    """Return HD95, ASSD, and area-weighted Surface Dice for one patient.

    HD95 and ASSD use bidirectional surface-voxel distances. Surface Dice uses
    physical surfel areas, matching the SegRap2023 GTV evaluation definition.
    """
    prediction, target = _as_binary_pair(prediction, target)
    spacing = _validate_spacing(spacing_mm, prediction.ndim)
    if tolerance_mm < 0 or not np.isfinite(tolerance_mm):
        raise ValueError("tolerance_mm must be finite and non-negative")

    prediction_surface = surface_mask(prediction)
    target_surface = surface_mask(target)
    prediction_empty = not np.any(prediction_surface)
    target_empty = not np.any(target_surface)
    if prediction_empty and target_empty:
        return {"hd95_mm": 0.0, "assd_mm": 0.0, "surface_dice": 1.0}
    if prediction_empty or target_empty:
        physical_extent = (
            np.maximum(np.asarray(prediction.shape, dtype=np.float64) - 1.0, 0.0)
            * np.asarray(spacing)
        )
        diagonal = float(np.linalg.norm(physical_extent))
        return {"hd95_mm": diagonal, "assd_mm": diagonal, "surface_dice": 0.0}

    prediction_to_target = directed_surface_distances(
        prediction_surface, target_surface, spacing
    )
    target_to_prediction = directed_surface_distances(
        target_surface, prediction_surface, spacing
    )
    hd95 = float(
        np.percentile(np.concatenate((prediction_to_target, target_to_prediction)), 95)
    )
    assd = 0.5 * (
        float(prediction_to_target.mean()) + float(target_to_prediction.mean())
    )
    try:
        from surface_distance import metrics as surface_distance_metrics
    except ImportError as error:
        raise ImportError("Surface Dice evaluation requires surface-distance") from error
    distances = surface_distance_metrics.compute_surface_distances(
        target,
        prediction,
        spacing,
    )
    surface_dice = float(
        surface_distance_metrics.compute_surface_dice_at_tolerance(
            distances,
            tolerance_mm,
        )
    )
    return {"hd95_mm": hd95, "assd_mm": assd, "surface_dice": surface_dice}


def patient_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: Sequence[float],
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    """Compute all seven ReCP evaluation metrics for one patient volume."""
    metrics = overlap_metrics(prediction, target)
    metrics.update(surface_metrics(prediction, target, spacing_mm, tolerance_mm))
    return metrics


def summarize_patient_metrics(
    cases: Sequence[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Return mean and sample standard deviation over a fixed patient cohort."""
    if not cases:
        raise ValueError("at least one patient is required")
    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray([case[name] for case in cases], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"metric contains non-finite values: {name}")
        summary[name] = {
            "mean": float(values.mean()),
            "std": 0.0 if len(values) == 1 else float(values.std(ddof=1)),
        }
    return summary
