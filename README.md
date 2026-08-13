# ReCP: Review-Stage Partial Code Release

This repository is a **review-stage partial release** accompanying the manuscript
"Reliability-Calibrated Pseudo-Supervision for Nasopharyngeal Carcinoma Segmentation."

ReCP formulates semi-supervised NPC segmentation as reliability-calibrated
pseudo-supervision in an EMA teacher--student framework. It contains three
components:

1. Target-Exempted Evidential Regularization (TEER)
2. Uncertainty-Gated Text Prior Injection (UGT)
3. Boundary-Focused Contrastive Learning (BFCL)

## Scope of this release

The review package exposes a compact 2D U-Net API model, EMA update, training
orchestration, data contracts, manuscript-aligned configuration, and inspectable
implementations of the principal equations:

- **TEER:** Dirichlet evidence statistics, one-hot evidential NLL,
  target-exempted KL-to-uniform, soft Dice, and their supervised composition.
- **UGT:** the six fixed CT prompts, prototype centroid construction, text
  pseudo-evidence, Jensen--Shannon gating, evidence rectification, and
  reliability-weighted valid-region consistency.
- **BFCL:** dynamic pseudo-boundaries, reliability-first boundary/interior
  candidate pools, and boundary--uncertainty hardness weights.

The visual-to-text projection design, weak/strong augmentation and alignment
implementation, BFCL sampling quota, local positive/negative pairing, hard-negative
ranking, and final weighted InfoNCE implementation remain withheld during review.
The package therefore supports method inspection but **does not claim full result
reproduction**.

The compact U-Net validates tensor and integration contracts. It is not the
parameter-matched experimental model used for the parameter count, FLOPs, latency,
or accuracy reported in the manuscript.

No clinical data, patient identifiers, private paths, pretrained weights,
checkpoints, split manifests, predictions, or experiment logs are included. A
complete implementation is planned for release after acceptance/publication.

## Repository structure

```text
configs/review.yaml              manuscript-aligned review configuration
recp/models/unet2d.py            compact evidential U-Net API model
recp/methods/teer.py             public TEER equations
recp/methods/ugt.py              public UGT equations
recp/methods/prompts.py          fixed minimalist CT anchors
recp/methods/bfcl.py             public BFCL candidate construction
recp/methods/teacher_student.py  EMA teacher update
recp/training/                   ramp-up and high-level ReCP orchestration
recp/evaluation/metrics.py       dataset-independent region metrics
examples/inspect_components.py   synthetic equation-level example
tests/                           behavioral tests for disclosed components
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Structural checks

```bash
python train.py --config configs/review.yaml --dry-run
python evaluate.py --dry-run
python -m examples.inspect_components
python -m unittest discover -s tests
```

These commands use synthetic tensors and do not access clinical data. Starting a
full training run intentionally raises a restricted-component notice.

## Expected data contract

Each training batch is a mapping. Labeled batches contain `image`, `mask`, and
optionally `valid_mask`. Unlabeled batches contain `weak_image`, `strong_image`,
`alignment`, and `valid_mask`. Teacher outputs from the weak view must be aligned
to the Student strong view before UGT. Images are single-channel 2D CT slices
resized to 256 x 256. The class order is `non_tumor`, then `tumor`.

Dataset-specific preprocessing, augmentation parameters, patient-level split
files, and cross-view alignment code are not part of this review release.

The public/restricted boundary is stated above and enforced by explicit restricted
interfaces in `recp/methods/contracts.py`.
