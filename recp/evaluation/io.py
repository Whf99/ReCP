"""Input helpers for prediction manifests and medical image volumes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prediction: Path
    target: Path
    spacing_mm: tuple[float, ...] | None = None


def load_manifest(path: Path) -> list[EvaluationCase]:
    """Read a JSON manifest and resolve case paths relative to the manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"evaluation manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a non-empty 'cases' list")

    cases: list[EvaluationCase] = []
    identifiers: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"case {index} must be a mapping")
        case_id = str(entry.get("case_id", "")).strip()
        if not case_id or case_id in identifiers:
            raise ValueError(f"case_id is empty or duplicated: {case_id!r}")
        identifiers.add(case_id)
        prediction = _resolve_path(path.parent, entry.get("prediction"), "prediction")
        target = _resolve_path(path.parent, entry.get("target"), "target")
        raw_spacing = entry.get("spacing_mm")
        if raw_spacing is not None and (
            not isinstance(raw_spacing, Sequence)
            or isinstance(raw_spacing, (str, bytes))
        ):
            raise TypeError("spacing_mm must be a sequence of physical voxel sizes")
        spacing = (
            None
            if raw_spacing is None
            else tuple(float(value) for value in raw_spacing)
        )
        cases.append(EvaluationCase(case_id, prediction, target, spacing))
    return cases


def _resolve_path(parent: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"case field '{field}' must be a path string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} volume not found: {path}")
    return path


def load_volume(
    path: Path,
    array_key: str | None = None,
) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """Load a NIfTI, NPY, or NPZ volume and optional header spacing."""
    lower_name = path.name.lower()
    if lower_name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as error:
            raise ImportError("reading NIfTI files requires nibabel") from error
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        spacing = tuple(float(value) for value in image.header.get_zooms()[: array.ndim])
        return array, spacing
    if lower_name.endswith(".npy"):
        return np.load(path, allow_pickle=False), None
    if lower_name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            if array_key is not None:
                if array_key not in archive:
                    raise KeyError(f"array '{array_key}' not found in {path}")
                return np.asarray(archive[array_key]), None
            if len(archive.files) != 1:
                raise ValueError(f"NPZ file requires an array key: {path}")
            return np.asarray(archive[archive.files[0]]), None
    raise ValueError(f"unsupported volume format: {path}")
