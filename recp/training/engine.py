"""Training engine for the ReCP teacher--student optimization loop."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from recp.data import validate_labeled_batch, validate_unlabeled_batch
from recp.methods import update_ema

from .orchestrator import LossWeights, recp_step


Batch = dict[str, Any]
ValidationFunction = Callable[
    [nn.Module, Mapping[str, nn.Module], torch.device],
    Mapping[str, float],
]


@dataclass
class ExperimentBundle:
    """Objects supplied by a dataset/experiment factory.

    Keeping dataset construction behind a factory makes the training loop usable
    with different de-identified cohorts without embedding site-specific paths or
    preprocessing rules in the entry point.
    """

    labeled_loader: Iterable[Batch]
    unlabeled_loader: Iterable[Batch]
    steps_per_epoch: int
    teer: Callable[..., dict[str, torch.Tensor]]
    ugt: Callable[..., Any]
    bfcl: Callable[..., torch.Tensor]
    align_teacher_to_student: Callable[..., tuple[dict[str, torch.Tensor], torch.Tensor]]
    text_prototypes: torch.Tensor
    validate: ValidationFunction | None = None
    validation_loader: Iterable[Batch] | None = None
    auxiliary_modules: dict[str, nn.Module] = field(default_factory=dict)
    prepare_epoch: Callable[[int], None] | None = None

    def check(self) -> None:
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        if self.text_prototypes.ndim != 2 or self.text_prototypes.shape[0] != 2:
            raise ValueError("text_prototypes must have shape 2 x D")
        if not torch.isfinite(self.text_prototypes).all():
            raise ValueError("text_prototypes must be finite")
        prototype_norms = torch.linalg.vector_norm(self.text_prototypes.float(), dim=1)
        if not torch.allclose(
            prototype_norms,
            torch.ones_like(prototype_norms),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("text_prototypes must be L2-normalized")
        if self.text_prototypes.requires_grad:
            raise ValueError("text_prototypes must be frozen")
        if not all(
            isinstance(module, nn.Module)
            for module in self.auxiliary_modules.values()
        ):
            raise TypeError("auxiliary_modules values must be torch modules")
        if self.validate is None and self.validation_loader is None:
            raise ValueError("provide a validation callback or validation_loader")
        for name in ("teer", "ugt", "bfcl", "align_teacher_to_student"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors while preserving batch metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class MetricAccumulator:
    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._count = 0

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        self._count += 1
        for name, value in values.items():
            if value.ndim == 0:
                self._totals[name] = self._totals.get(name, 0.0) + float(value.detach())

    def mean(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {name: total / self._count for name, total in self._totals.items()}


def _next_or_restart(iterator: Any, loader: Iterable[Batch], name: str) -> tuple[Batch, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as error:
            raise ValueError(f"{name} loader is empty") from error


class ReCPTrainer:
    """Run optimization, EMA updates, validation, logging, and checkpointing."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        experiment: ExperimentBundle,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        ema_decay: float,
        validation_model: str = "teacher",
        gradient_clip_norm: float | None = None,
    ) -> None:
        experiment.check()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when specified")
        if validation_model not in {"student", "teacher"}:
            raise ValueError("validation_model must be 'student' or 'teacher'")

        self.student = student.to(device)
        self.teacher = teacher.to(device).requires_grad_(False)
        self.experiment = experiment
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.ema_decay = float(ema_decay)
        self.validation_model = validation_model
        self.gradient_clip_norm = gradient_clip_norm
        self.best_dice = float("-inf")
        self.start_epoch = 0

        for module in self.experiment.auxiliary_modules.values():
            module.to(device)
        self.experiment.text_prototypes = self.experiment.text_prototypes.to(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _trainable_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.student.parameters())
        for module in self.experiment.auxiliary_modules.values():
            parameters.extend(module.parameters())
        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
        return unique

    def train_epoch(self, epoch: int, weights: LossWeights) -> dict[str, float]:
        self.student.train()
        self.teacher.eval()
        for module in self.experiment.auxiliary_modules.values():
            module.train()
        if self.experiment.prepare_epoch is not None:
            self.experiment.prepare_epoch(epoch)

        labeled_batches = iter(self.experiment.labeled_loader)
        unlabeled_batches = iter(self.experiment.unlabeled_loader)
        accumulator = MetricAccumulator()

        for _ in range(self.experiment.steps_per_epoch):
            labeled, labeled_batches = _next_or_restart(
                labeled_batches,
                self.experiment.labeled_loader,
                "labeled",
            )
            labeled = move_to_device(labeled, self.device)
            validate_labeled_batch(labeled)

            use_unlabeled = weights.ugt > 0.0 or weights.bfcl > 0.0
            unlabeled = None
            if use_unlabeled:
                unlabeled, unlabeled_batches = _next_or_restart(
                    unlabeled_batches,
                    self.experiment.unlabeled_loader,
                    "unlabeled",
                )
                unlabeled = move_to_device(unlabeled, self.device)
                validate_unlabeled_batch(unlabeled)

            self.optimizer.zero_grad(set_to_none=True)
            losses = recp_step(
                student=self.student,
                teacher=self.teacher,
                labeled=labeled,
                unlabeled=unlabeled,
                teer=self.experiment.teer,
                ugt=self.experiment.ugt,
                bfcl=self.experiment.bfcl,
                align_teacher_to_student=self.experiment.align_teacher_to_student,
                text_prototypes=self.experiment.text_prototypes,
                weights=weights,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            losses["loss"].backward()
            if self.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(self._trainable_parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            update_ema(self.student, self.teacher, self.ema_decay)
            accumulator.update(losses)

        return accumulator.mean()

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.student.eval()
        self.teacher.eval()
        for module in self.experiment.auxiliary_modules.values():
            module.eval()
        if self.experiment.validate is None:
            raise RuntimeError("validation callback was not configured")
        model = self.teacher if self.validation_model == "teacher" else self.student
        metrics = dict(
            self.experiment.validate(
                model,
                self.experiment.auxiliary_modules,
                self.device,
            )
        )
        if "dice" not in metrics:
            raise KeyError("validation callback must return a 'dice' value")
        metrics = {name: float(value) for name, value in metrics.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("validation returned a non-finite metric")
        return metrics

    def checkpoint_state(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "best_dice": self.best_dice,
            "validation_model": self.validation_model,
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "auxiliary_modules": {
                name: module.state_dict()
                for name, module in self.experiment.auxiliary_modules.items()
            },
        }

    def save_checkpoint(self, epoch: int, filename: str) -> Path:
        destination = self.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_state(epoch), temporary)
        temporary.replace(destination)
        return destination

    def load_checkpoint(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location=self.device)
        checkpoint_validation_model = state.get("validation_model")
        if (
            checkpoint_validation_model is not None
            and checkpoint_validation_model != self.validation_model
        ):
            raise ValueError(
                "checkpoint validation model does not match the current configuration"
            )
        auxiliary_state = state.get("auxiliary_modules", {})
        checkpoint_modules = set(auxiliary_state)
        current_modules = set(self.experiment.auxiliary_modules)
        if checkpoint_modules != current_modules:
            raise ValueError(
                "checkpoint auxiliary modules do not match the current experiment"
            )
        self.student.load_state_dict(state["student"])
        self.teacher.load_state_dict(state["teacher"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name, module_state in auxiliary_state.items():
            self.experiment.auxiliary_modules[name].load_state_dict(module_state)
        self.start_epoch = int(state["epoch"]) + 1
        self.best_dice = float(state.get("best_dice", float("-inf")))

    def _append_log(self, record: Mapping[str, Any]) -> None:
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def fit(
        self,
        epochs: int,
        weight_schedule: Callable[[int], LossWeights],
        checkpoint_interval: int = 1,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")

        for epoch in range(self.start_epoch, epochs):
            weights = weight_schedule(epoch)
            training = self.train_epoch(epoch, weights)
            validation = self.validate()
            record = {
                "epoch": epoch + 1,
                "lambda_ugt": weights.ugt,
                "lambda_bfcl": weights.bfcl,
                "validation_model": self.validation_model,
                "train": training,
                "validation": validation,
            }
            self._append_log(record)
            print(json.dumps(record, sort_keys=True))

            improved = validation["dice"] > self.best_dice
            if improved:
                self.best_dice = validation["dice"]
                self.save_checkpoint(epoch, "best.pt")
            if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
                self.save_checkpoint(epoch, "last.pt")
