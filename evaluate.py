"""Review-stage evaluation contract; clinical evaluation data are not distributed."""

import argparse

import torch

from recp.evaluation import region_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        example = torch.tensor([[[0, 1], [0, 1]]])
        metrics = region_metrics(example, example)
        names = ", ".join(metrics)
        print(f"evaluation contract ready: {names}; boundary metrics require physical spacing")
        return
    raise RuntimeError("Provide an approved dataset adapter and model checkpoint in the complete release.")


if __name__ == "__main__":
    main()

