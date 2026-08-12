"""Dataset-independent binary region metrics for patient-level aggregation."""

import torch


def region_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return per-item Dice, Jaccard, precision, and recall."""
    pred = prediction.bool().reshape(prediction.shape[0], -1)
    truth = target.bool().reshape(target.shape[0], -1)
    true_positive = (pred & truth).sum(dim=1).float()
    false_positive = (pred & ~truth).sum(dim=1).float()
    false_negative = (~pred & truth).sum(dim=1).float()
    return {
        "dice": (2.0 * true_positive + eps)
        / (2.0 * true_positive + false_positive + false_negative + eps),
        "jaccard": (true_positive + eps)
        / (true_positive + false_positive + false_negative + eps),
        "precision": (true_positive + eps) / (true_positive + false_positive + eps),
        "recall": (true_positive + eps) / (true_positive + false_negative + eps),
    }

