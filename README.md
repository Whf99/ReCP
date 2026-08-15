# ReCP

ReCP is the PyTorch implementation accompanying
"Reliability-Calibrated Pseudo-Supervision for Nasopharyngeal Carcinoma
Segmentation." It addresses semi-supervised NPC segmentation with an EMA
teacher--student framework and three reliability-aware components:

1. Target-Exempted Evidential Regularization (TEER)
2. Uncertainty-Gated Text Prior Injection (UGT)
3. Boundary-Focused Contrastive Learning (BFCL)

## Method overview

The segmentation network predicts non-negative evidence for the `non_tumor` and
`tumor` classes. TEER converts this evidence into Dirichlet parameters and
regularizes non-target evidence without suppressing the target-class parameter.
UGT combines the aligned EMA-Teacher posterior with three fixed CT text anchors
per class; its contribution is controlled by visual vacuity and Jensen--Shannon
cross-modal divergence. BFCL constructs dynamic pseudo-boundaries and emphasizes
reliable, difficult boundary and interior features.

The implementation organizes the TEER and UGT equations, BFCL boundary and
candidate construction, EMA update, loss scheduling, training engine,
patient-level evaluation, and a compact 2D U-Net behind explicit tensor
contracts. Data loading, split handling, and preprocessing are decoupled through
an experiment factory so that no clinical paths, split files, or institutional
metadata are embedded in the training entry point.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Configuration and model checks

```bash
python train.py --config configs/recp.yaml --check-config
python -m examples.inspect_components
python -m unittest discover -s tests
```

`--check-config` validates the 256 x 256 single-channel input contract, the
two-class evidential output, decoder feature output, prompt count, and all
training hyperparameters without loading a dataset.

## Training

The training entry point performs configuration validation, deterministic
seeding, Student/Teacher construction, AdamW optimization, supervised warm-up,
Gaussian ramp-up of UGT and BFCL, EMA updates after each optimizer step,
validation, JSONL logging, and `best.pt`/`last.pt` checkpointing.

Data and experiment-specific modules are provided by a Python factory:

```bash
python train.py \
  --config configs/recp.yaml \
  --factory my_experiment.factory:build_experiment \
  --output-dir runs/recp
```

An alternative U-Net implementation with the same `evidence`/`features` output
contract can be selected independently:

```bash
python train.py \
  --config configs/recp.yaml \
  --model-factory my_experiment.models:build_model \
  --factory my_experiment.factory:build_experiment \
  --output-dir runs/recp
```

The model factory receives `config` and returns one `torch.nn.Module`; the EMA
Teacher is initialized as an exact copy by the training entry point.

The factory receives `config`, `student`, and `teacher`, and returns
`recp.training.ExperimentBundle`. The bundle contains labeled and unlabeled data
loaders, TEER/UGT/BFCL callables, the weak-to-strong alignment function, frozen
text prototypes, optional trainable auxiliary modules, and the validation
callback. The training loop itself is implemented in
`recp/training/engine.py`.

When `validation_loader` is used instead of a custom callback, each yielded
batch represents one complete patient volume with slices placed on the batch
dimension. The default validator selects the checkpoint with the highest mean
patient-level foreground Dice.
The supplied configuration validates the EMA Teacher. Set
`training.validation_model` to `student` if the Student is the intended
selection model; a custom callback can instead evaluate the fused UGT output.

Resume a run with:

```bash
python train.py \
  --config configs/recp.yaml \
  --factory my_experiment.factory:build_experiment \
  --resume runs/recp/last.pt
```

## Batch contract

A labeled batch contains:

```python
{
    "image": Tensor[N, 1, H, W],
    "mask": Tensor[N, H, W],
    "valid_mask": Tensor[N, 1, H, W],  # optional
}
```

An unlabeled batch contains:

```python
{
    "weak_image": Tensor[N, 1, H, W],
    "strong_image": Tensor[N, 1, H, W],
    "alignment": ...,                  # interpreted by the alignment callback
    "valid_mask": Tensor[N, 1, H, W],
}
```

The Teacher output from the weak view must be transformed into the Student strong
view before UGT is evaluated. Images are single-channel 2D CT slices resized to
256 x 256, and class indices are ordered as `non_tumor`, then `tumor`.

## Patient-level evaluation

`evaluate.py` reads complete patient volumes and reports Dice, Jaccard, precision,
recall, HD95, ASSD, and Surface Dice with a 1 mm tolerance. Distances use the
physical voxel spacing, and the output contains each patient's metrics plus the
mean and standard deviation over the cohort.

If both foreground masks are empty, overlap and Surface Dice are reported as
one and surface distances as zero. If exactly one is empty, overlap and Surface
Dice are zero and HD95/ASSD use the physical image diagonal as a finite penalty.

```bash
python evaluate.py \
  --manifest examples/evaluation_manifest.json \
  --output results/metrics.json
```

The manifest is a JSON list (or a mapping with a `cases` list). Paths are resolved
relative to the manifest:

```json
{
  "cases": [
    {
      "case_id": "case_001",
      "prediction": "predictions/case_001.nii.gz",
      "target": "targets/case_001.nii.gz"
    },
    {
      "case_id": "case_002",
      "prediction": "predictions/case_002.npy",
      "target": "targets/case_002.npy",
      "spacing_mm": [3.0, 1.0, 1.0]
    }
  ]
}
```

NIfTI spacing is read from the header. NPY/NPZ cases must provide `spacing_mm`.
Floating-point foreground-probability volumes are thresholded at 0.5 by default;
binary targets must contain only 0 and 1. Prediction and target volumes must
already share the same voxel grid. Spacing values follow the corresponding NumPy
array axis order.

## Data and model assets

Clinical images, patient identifiers, split files, trained weights, checkpoints,
predictions, and experiment logs are not distributed in this repository. Use only
institution-approved, de-identified data and preserve patient-level train,
validation, and test separation.
