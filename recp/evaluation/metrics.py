"""Dataset-independent binary region metrics for patient-level aggregation."""

import torch


def region_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-item Dice, Jaccard, precision, and recall."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include a batch dimension")
    pred = prediction.bool().reshape(prediction.shape[0], -1)
    truth = target.bool().reshape(target.shape[0], -1)
    true_positive = (pred & truth).sum(dim=1).float()
    false_positive = (pred & ~truth).sum(dim=1).float()
    false_negative = (~pred & truth).sum(dim=1).float()
    prediction_count = true_positive + false_positive
    target_count = true_positive + false_negative
    overlap_denominator = prediction_count + target_count
    union = true_positive + false_positive + false_negative
    both_empty = overlap_denominator == 0
    one = torch.ones_like(true_positive)
    zero = torch.zeros_like(true_positive)
    return {
        "dice": torch.where(
            both_empty,
            one,
            2.0 * true_positive / overlap_denominator.clamp_min(1.0),
        ),
        "jaccard": torch.where(
            both_empty,
            one,
            true_positive / union.clamp_min(1.0),
        ),
        "precision": torch.where(
            both_empty,
            one,
            torch.where(
                prediction_count == 0,
                zero,
                true_positive / prediction_count.clamp_min(1.0),
            ),
        ),
        "recall": torch.where(
            both_empty,
            one,
            torch.where(
                target_count == 0,
                zero,
                true_positive / target_count.clamp_min(1.0),
            ),
        ),
    }

