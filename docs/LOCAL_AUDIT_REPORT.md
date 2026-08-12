# Local audit report

Audit date: 2026-08-12

## Source integrity

- Both research source directories were inspected read-only.
- No source file under either original directory was modified.
- The candidate package was created independently under `outputs`.
- No Git repository was initialized and no remote operation was performed.

## Correspondence findings

- The historical code contains a mean-teacher foundation and experimental text
  feature fusion.
- It does not implement manuscript-equivalent TEER, UGT gating with JS divergence,
  or BFCL dynamic boundary sampling.
- Historical defaults also differ from the manuscript optimizer, duration, warm-up,
  output parameterization, and loss composition.
- Consequently, no historical training script is presented as ReCP in this package.
- The public orchestrator was corrected to show Teacher-to-Student cross-view
  alignment before UGT, as required by the manuscript algorithm.
- The compact public U-Net is explicitly identified as an API model rather than the
  parameter-matched experimental network reported in the manuscript.
- TEER label conversion accepts both NHW class-index masks and the N1HW masks
  allowed by the public labeled-batch contract.

## Disclosure and privacy checks

- No absolute research-machine paths detected in the candidate files.
- No credential/key literals detected.
- No checkpoints, pretrained weights, clinical images, HDF5/NIfTI/DICOM files,
  patient split lists, predictions, or logs included.
- No third-party SSL4MIS source file copied into the candidate package.

## Validation

- Python syntax compilation: passed.
- Evaluation contract dry-run: passed.
- Training tensor dry-run: not executed because PyTorch is not installed in either
  available local Python runtime. This is an environment limitation, not reported
  as a passed test.
- Primitive behavioral tests: authored but not executed for the same missing
  PyTorch dependency; their import failure is not reported as a test pass.
- Full ReCP training: intentionally unavailable in this partial release.
- A limited equation-level implementation of all three components was added after
  the initial audit; it does not expose the complete training objectives.

## Required decisions before publication

1. Confirm ownership and final license (the current license is review-only).
2. Confirm that revealing typed contracts and method orchestration is acceptable.
3. Decide whether reviewer access should be public, private, or anonymous.
4. Approve the exact file list before any Git commit or upload.

Only paths in `docs/UPLOAD_ALLOWLIST.txt` should be staged. Generated caches and
all non-allowlisted files must remain excluded.

The final static paper/code review is recorded in
`docs/PAPER_CONSISTENCY_CHECKLIST.md`.
