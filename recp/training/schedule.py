"""Public scheduling utility described in the manuscript."""

import math

from .orchestrator import LossWeights


def gaussian_rampup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    length = max(total_epochs - warmup_epochs, 1)
    phase = min(max((epoch - warmup_epochs) / length, 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - phase) ** 2)


def recp_loss_weights(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    lambda_ugt_max: float,
    lambda_bfcl_max: float,
) -> LossWeights:
    """Return the two manuscript loss weights under a shared Gaussian ramp-up."""
    ramp = gaussian_rampup(epoch, warmup_epochs, total_epochs)
    return LossWeights(
        ugt=float(lambda_ugt_max) * ramp,
        bfcl=float(lambda_bfcl_max) * ramp,
    )

