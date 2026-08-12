# Paper-to-code mapping and source audit

| Manuscript item | Existing research-code finding | Review package |
|---|---|---|
| EMA teacher--student | Present in `TI-MT_v2` and NPC SSL baselines | `recp/methods/teacher_student.py` |
| 2D U-Net, 256 x 256 | A historical U-Net exists, but its output is binary and text concatenation is asymmetric | `recp/models/unet2d.py` exposes non-negative two-class evidence; it is not parameter-matched to the reported experimental model |
| Teacher weak view aligned to Student strong view, with validity mask V | Not explicit in the historical training path | Explicit aligner call precedes UGT in `recp/training/orchestrator.py` |
| TEER, target-exempted alpha/NLL/KL | No equivalent implementation located | Dirichlet statistics and target exemption public; complete loss restricted |
| UGT prototypes and Softplus pseudo-evidence | Existing code uses one/two ad-hoc prompts and feature concatenation | Restricted `UGT` contract |
| UGT gate using vacuity and JS divergence | No equivalent implementation located | JS divergence and uncertainty gate public; fusion restricted |
| Reliability-weighted valid-region consistency | Existing code uses unmasked MSE consistency | Public orchestration; restricted loss implementation |
| BFCL dynamic boundary and reliable hard anchors | Existing code has global image--text contrast, not BFCL | Dynamic boundary mask public; anchor mining and loss restricted |
| Dice + EDL + ramped UGT + ramped BFCL | Existing objective differs materially | Configuration and schedule exposed; core losses restricted |
| AdamW, 300 epochs, 15-epoch warm-up | Existing `TI-MT_v2` uses Adam, 200 epochs and a different warm-up | `configs/review.yaml` follows the manuscript |

## Audit conclusion

The inspected sources are useful historical baselines but are not an implementation
of the submitted ReCP formulation. Renaming them would misrepresent correspondence
with the manuscript. The review package therefore exposes a clean architecture and
explicitly marks unavailable method internals instead of presenting legacy logic as
ReCP.

The high-level order of the three components and total objective is visible in
`recp/training/orchestrator.py`; that file deliberately delegates all restricted
mathematics to the typed component interfaces.
