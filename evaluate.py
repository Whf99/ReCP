"""Review-stage evaluation contract; clinical evaluation data are not distributed."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("evaluation contract ready: patient-level Dice/Jaccard/precision/recall/HD95/ASSD/Surface Dice")
        return
    raise RuntimeError("Provide an approved dataset adapter and model checkpoint in the complete release.")


if __name__ == "__main__":
    main()

