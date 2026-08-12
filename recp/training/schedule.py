"""Public scheduling utility described in the manuscript."""

import math


def gaussian_rampup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    length = max(total_epochs - warmup_epochs, 1)
    phase = min(max((epoch - warmup_epochs) / length, 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - phase) ** 2)

