"""EMA teacher update for teacher--student optimization."""

import torch
from torch import nn


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1).")
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        raise ValueError("Student and Teacher parameter structures do not match.")
    for name, teacher_parameter in teacher_parameters.items():
        teacher_parameter.mul_(decay).add_(
            student_parameters[name], alpha=1.0 - decay
        )

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    if student_buffers.keys() != teacher_buffers.keys():
        raise ValueError("Student and Teacher buffer structures do not match.")
    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name])

