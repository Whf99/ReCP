"""Command-line training entry point for ReCP."""

from __future__ import annotations

import argparse
import importlib
import json
import random
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from recp.data import validate_labeled_batch
from recp.methods import dirichlet_statistics, recp_prompts
from recp.models import UNet2D
from recp.training import ExperimentBundle, ReCPTrainer, recp_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ReCP with an experiment factory that supplies data and method modules."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/recp.yaml"))
    parser.add_argument(
        "--factory",
        help="Experiment factory in 'package.module:function' form.",
    )
    parser.add_argument(
        "--model-factory",
        help="Optional model factory in 'package.module:function' form.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name accepted by PyTorch, or 'auto' (default).",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and tensor contracts, then exit.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError("configuration root must be a mapping")
    validate_config(content)
    return content


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = {"experiment", "data", "model", "training", "method"}
    missing = required_sections.difference(config)
    if missing:
        raise KeyError(f"missing configuration sections: {sorted(missing)}")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    method = config["method"]
    for name, section in (
        ("experiment", config["experiment"]),
        ("data", data),
        ("model", model),
        ("training", training),
        ("method", method),
    ):
        if not isinstance(section, Mapping):
            raise TypeError(f"configuration section '{name}' must be a mapping")

    if list(data["image_size"]) != [256, 256]:
        raise ValueError("ReCP expects 256 x 256 input slices")
    if int(data["in_channels"]) != 1 or int(data["num_classes"]) != 2:
        raise ValueError("ReCP expects one CT channel and two segmentation classes")
    if list(data["class_names"]) != ["non_tumor", "tumor"]:
        raise ValueError("class_names must be ordered as non_tumor, tumor")
    if not 0.0 < float(data["labeled_ratio"]) <= 1.0:
        raise ValueError("labeled_ratio must lie in (0, 1]")
    if model["backbone"] != "unet2d":
        raise ValueError("the included model factory supports backbone='unet2d'")
    if int(model["base_channels"]) < 1:
        raise ValueError("base_channels must be positive")

    epochs = int(training["epochs"])
    warmup_epochs = int(training["warmup_epochs"])
    if epochs < 1 or not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must lie in [0, epochs)")
    if int(training["batch_size"]) < 1:
        raise ValueError("batch_size must be positive")
    if training["optimizer"].lower() != "adamw":
        raise ValueError("ReCP is configured with the AdamW optimizer")
    if float(training["learning_rate"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("optimizer learning rate/weight decay are invalid")
    if not 0.0 <= float(training["ema_decay"]) < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if training.get("validation_model", "teacher") not in {"student", "teacher"}:
        raise ValueError("validation_model must be 'student' or 'teacher'")
    gradient_clip = training.get("gradient_clip_norm")
    if gradient_clip is not None and float(gradient_clip) <= 0:
        raise ValueError("gradient_clip_norm must be positive when specified")
    if int(training.get("checkpoint_interval", 1)) < 1:
        raise ValueError("checkpoint_interval must be positive")

    for name in ("lambda_ugt_max", "lambda_bfcl_max", "gate_gamma"):
        if float(method[name]) < 0:
            raise ValueError(f"method.{name} must be non-negative")
    if float(method["text_temperature"]) <= 0:
        raise ValueError("method.text_temperature must be positive")
    if not 0.0 <= float(method["confidence_threshold"]) <= 1.0:
        raise ValueError("method.confidence_threshold must lie in [0, 1]")
    if int(method["prompt_pairs"]) != 3:
        raise ValueError("ReCP uses three CT prompt pairs")
    if method["text_encoder"] != "clip_vit_b32":
        raise ValueError("ReCP uses the CLIP ViT-B/32 text encoder")
    if method["freeze_text_encoder"] is not True:
        raise ValueError("the text encoder must remain frozen")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def build_models(
    config: Mapping[str, Any],
    model_factory_specification: str | None = None,
) -> tuple[nn.Module, nn.Module]:
    data = config["data"]
    model = config["model"]
    if model_factory_specification:
        factory = import_factory(model_factory_specification)
        student = factory(config=config)
        if not isinstance(student, nn.Module):
            raise TypeError("model factory must return one torch.nn.Module")
    else:
        student = UNet2D(
            in_channels=int(data["in_channels"]),
            num_classes=int(data["num_classes"]),
            base=int(model["base_channels"]),
        )
    teacher = deepcopy(student).requires_grad_(False)
    return student, teacher


def import_factory(specification: str) -> Callable[..., ExperimentBundle]:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("factory must use 'package.module:function' syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"factory is not callable: {specification}")
    return function


def unique_trainable_parameters(
    student: nn.Module,
    auxiliary_modules: Mapping[str, nn.Module],
) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in (student, *auxiliary_modules.values()):
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    if not parameters:
        raise ValueError("no trainable parameters were supplied")
    return parameters


@torch.no_grad()
def validate_labeled_loader(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> dict[str, float]:
    """Compute mean patient Dice when each batch contains one complete volume."""
    patient_dice: list[float] = []
    for batch in loader:
        validate_labeled_batch(batch)
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        output = model(image)
        _, probability, _ = dirichlet_statistics(output["evidence"])
        prediction = probability.argmax(dim=1).eq(1)
        target = target.eq(1)
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid = valid_mask.to(device, non_blocking=True).bool()
            if valid.ndim == 4 and valid.shape[1] == 1:
                valid = valid[:, 0]
            prediction = prediction & valid
            target = target & valid
        intersection = float((prediction & target).sum())
        denominator = float(prediction.sum()) + float(target.sum())
        patient_dice.append(
            1.0 if denominator == 0.0 else 2.0 * intersection / denominator
        )
    if not patient_dice:
        raise ValueError("validation loader is empty")
    return {"dice": float(np.mean(patient_dice))}


def finalize_experiment_bundle(
    bundle: ExperimentBundle,
) -> ExperimentBundle:
    """Attach labeled-volume validation when a callback is not supplied."""
    validation = bundle.validate
    if validation is None:
        if bundle.validation_loader is None:
            raise ValueError("provide either validate callback or validation_loader")

        def validation(model, modules, device):
            del modules
            return validate_labeled_loader(model, bundle.validation_loader, device)

    callbacks = [
        callback
        for callback in (
            bundle.prepare_epoch,
            getattr(bundle.teer, "set_epoch", None),
        )
        if callback
    ]

    def prepare_epoch(epoch: int) -> None:
        for callback in callbacks:
            callback(epoch)

    bundle.validate = validation
    bundle.prepare_epoch = prepare_epoch
    return bundle


@torch.no_grad()
def check_tensor_contracts(config: Mapping[str, Any], student: nn.Module) -> dict[str, Any]:
    data = config["data"]
    height, width = map(int, data["image_size"])
    image = torch.zeros(1, int(data["in_channels"]), height, width)
    was_training = student.training
    student.eval()
    try:
        output = student(image)
    finally:
        student.train(was_training)
    required = {"evidence", "features"}
    if not required.issubset(output):
        raise KeyError(f"model output is missing keys: {sorted(required.difference(output))}")
    _, probability, uncertainty = dirichlet_statistics(output["evidence"])
    expected = (1, int(data["num_classes"]), height, width)
    if tuple(probability.shape) != expected:
        raise ValueError(f"unexpected probability shape: {tuple(probability.shape)}")
    return {
        "configuration": str(config["experiment"]["name"]),
        "parameters": sum(parameter.numel() for parameter in student.parameters()),
        "evidence_shape": list(output["evidence"].shape),
        "feature_shape": list(output["features"].shape),
        "uncertainty_shape": list(uncertainty.shape),
        "prompt_count": sum(len(items) for items in recp_prompts().values()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_config = config["experiment"]
    training_config = config["training"]
    method_config = config["method"]

    seed_everything(
        int(experiment_config["seed"]),
        bool(experiment_config.get("deterministic", True)),
    )
    student, teacher = build_models(config, args.model_factory)
    contract_summary = check_tensor_contracts(config, student)
    if args.check_config:
        print(json.dumps(contract_summary, indent=2, sort_keys=True))
        return

    factory_specification = args.factory or experiment_config.get("factory")
    if not factory_specification:
        raise ValueError(
            "training requires --factory package.module:function; the factory must "
            "return recp.training.ExperimentBundle"
        )
    factory = import_factory(str(factory_specification))
    bundle = factory(config=config, student=student, teacher=teacher)
    if not isinstance(bundle, ExperimentBundle):
        raise TypeError("experiment factory must return ExperimentBundle")
    bundle = finalize_experiment_bundle(bundle)
    bundle.check()

    device = select_device(args.device)
    student.to(device)
    teacher.to(device)
    for module in bundle.auxiliary_modules.values():
        module.to(device)
    optimizer = torch.optim.AdamW(
        unique_trainable_parameters(student, bundle.auxiliary_modules),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    output_dir = args.output_dir or Path(training_config.get("output_dir", "runs/recp"))
    trainer = ReCPTrainer(
        student=student,
        teacher=teacher,
        experiment=bundle,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        ema_decay=float(training_config["ema_decay"]),
        validation_model=str(training_config.get("validation_model", "teacher")),
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    epochs = int(training_config["epochs"])
    warmup_epochs = int(training_config["warmup_epochs"])

    def weight_schedule(epoch: int):
        return recp_loss_weights(
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            lambda_ugt_max=float(method_config["lambda_ugt_max"]),
            lambda_bfcl_max=float(method_config["lambda_bfcl_max"]),
        )

    trainer.fit(
        epochs=epochs,
        weight_schedule=weight_schedule,
        checkpoint_interval=int(training_config.get("checkpoint_interval", 1)),
    )


if __name__ == "__main__":
    main()
