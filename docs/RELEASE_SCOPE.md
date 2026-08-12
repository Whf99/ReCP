# Review release scope

## Included

- Method-level training flow and manuscript-aligned hyperparameters
- Compact 2D U-Net API model and feature/output contract (not the reported parameter-matched experimental network)
- EMA teacher update and Gaussian ramp-up utility
- Labeled/unlabeled batch contracts
- TEER, UGT, and BFCL typed input/output interfaces
- TEER Dirichlet/vacuity and target-exemption primitives
- UGT pixel-wise JS divergence and uncertainty gate
- BFCL dynamic pseudo-boundary construction
- Small behavioral tests for the disclosed primitives
- Dry-run entry points and disclosure of reproducibility limits

## Intentionally excluded

- Complete TEER NLL/KL loss construction and annealing integration
- Complete UGT text-prototype projection, pseudo-evidence fusion, and consistency loss
- Geometric alignment implementation and augmentation parameters (the alignment call and validity-mask contract are public)
- Complete BFCL reliability-first anchor mining, hard-pair sampling, and weighted InfoNCE
- Text prototype files and pretrained CLIP weights
- Private dataset preprocessing, split lists, or patient-level metadata
- Model checkpoints, training logs, prediction outputs, and unpublished results
- Institution-specific paths or computing-environment details
- Legacy experimental scripts and third-party source files

## Publication-stage additions

Subject to author and institutional approval, the full release should add the three
method implementations, deterministic split manifests, preprocessing documentation,
training/evaluation tests, and final checkpoints where licensing and data governance
allow. This document and the README must be updated when that happens.
