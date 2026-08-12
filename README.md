# ReCP: Review-Stage Partial Code Release

This repository is a **review-stage partial release** accompanying the manuscript
"Reliability-Calibrated Pseudo-Supervision for Nasopharyngeal Carcinoma Segmentation."

ReCP formulates semi-supervised NPC segmentation as reliability-calibrated
pseudo-supervision in an EMA teacher--student framework. The paper contains three
components:

1. Target-Exempted Evidential Regularization (TEER)
2. Uncertainty-Gated Text Prior Injection (UGT)
3. Boundary-Focused Contrastive Learning (BFCL)

## Scope of this release

The public review package exposes the experiment configuration, a compact 2D
U-Net API model, EMA update, training orchestration, data contracts, and the exact
input/output contracts of the three method components. It additionally includes a
limited set of equation-level primitives: TEER Dirichlet statistics and target
exemption, the UGT Jensen--Shannon uncertainty gate, and the BFCL dynamic boundary
mask. Complete objectives, evidence fusion, anchor mining, and hard-pair selection
remain withheld. This package therefore supports structural inspection but **does
not claim full result reproduction**.

The compact U-Net included here validates tensor and integration contracts. It is
not the parameter-matched experimental model used for the parameter count, FLOPs,
latency, or accuracy reported in the manuscript.

No clinical data, patient identifiers, private paths, pretrained weights,
checkpoints, or experiment logs are included. A complete implementation is
planned for release after acceptance/publication.

## Quick structural check

```bash
python train.py --config configs/review.yaml --dry-run
python evaluate.py --dry-run
python -m unittest discover -s tests
```

These commands validate configuration and model/data contracts only. Training the
full ReCP method requires the restricted components and an institution-approved,
de-identified dataset adapter.

## Expected data contract

Each training batch is a mapping. Labeled batches contain `image`, `mask`, and
optionally `valid_mask`; unlabeled batches contain weak/strong views, alignment
metadata, and their geometric validity mask. The public orchestration explicitly
aligns Teacher outputs into the Student view before UGT. Images are single-channel 2D CT slices resized to
256 x 256. Dataset-specific preprocessing and patient split files are not part of
this review release.

See `recp/methods/public_primitives.py`, `docs/PAPER_CODE_MAPPING.md`, and
`docs/PAPER_CONSISTENCY_CHECKLIST.md` before use. The detailed inclusion and
exclusion policy is in `docs/RELEASE_SCOPE.md`.

## Acknowledgement

The research code was developed with reference to common semi-supervised medical
segmentation practice and the SSL4MIS ecosystem. No SSL4MIS source file is copied
into this partial release. Users of the complete research pipeline should cite the
relevant original methods and datasets described in the manuscript.
