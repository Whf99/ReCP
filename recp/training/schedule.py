"""Schedules used by ReCP losses."""

import math

from .orchestrator import LossWeights


def gaussian_rampup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must lie in [0, total_epochs)")
    if epoch < warmup_epochs:
        return 0.0
    length = max(total_epochs - warmup_epochs - 1, 1)
    phase = min(max((epoch - warmup_epochs) / length, 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - phase) ** 2)


def recp_loss_weights(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    lambda_ugt_max: float,
    lambda_bfcl_max: float,
) -> LossWeights:
    """Return UGT and BFCL weights under a shared Gaussian ramp-up."""
    if lambda_ugt_max < 0.0 or lambda_bfcl_max < 0.0:
        raise ValueError("maximum loss weights must be non-negative")
    ramp = gaussian_rampup(epoch, warmup_epochs, total_epochs)
    return LossWeights(
        ugt=float(lambda_ugt_max) * ramp,
        bfcl=float(lambda_bfcl_max) * ramp,
    )

