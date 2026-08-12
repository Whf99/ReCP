"""Review-stage entry point: validates the public orchestration contract."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from recp.methods.contracts import restricted_component
from recp.methods import dirichlet_statistics, dynamic_boundary_mask, recp_prompts
from recp.models import UNet2D
from recp.training import gaussian_rampup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg, train_cfg = config["model"], config["data"], config["training"]
    student = UNet2D(data_cfg["in_channels"], data_cfg["num_classes"], model_cfg["base_channels"])
    teacher = deepcopy(student).requires_grad_(False)
    sample = torch.zeros(1, data_cfg["in_channels"], *data_cfg["image_size"])
    output = student(sample)
    _, probability, uncertainty = dirichlet_statistics(output["evidence"])
    boundary = dynamic_boundary_mask(probability.argmax(dim=1))
    ramp = gaussian_rampup(train_cfg["warmup_epochs"], train_cfg["warmup_epochs"], train_cfg["epochs"])
    print(
        "student/teacher ready; "
        f"evidence={tuple(output['evidence'].shape)}; "
        f"uncertainty={tuple(uncertainty.shape)}; "
        f"boundary={tuple(boundary.shape)}; "
        f"prompts={sum(map(len, recp_prompts().values()))}; "
        f"warmup_ramp={ramp:.1f}"
    )
    if not args.dry_run:
        restricted_component("TEER/UGT/BFCL training")


if __name__ == "__main__":
    main()
