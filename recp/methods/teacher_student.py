"""Non-sensitive EMA teacher update shared by mean-teacher methods."""

import torch
from torch import nn


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1).")
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)

