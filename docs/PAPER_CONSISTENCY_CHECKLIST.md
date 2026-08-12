# Paper consistency checklist

This checklist records what can be verified in the review-stage partial release.
"Consistent" means the disclosed code does not contradict the manuscript; it does
not mean the withheld method can be reproduced from this repository.

| Manuscript definition | Public-code evidence | Status |
|---|---|---|
| Two-class, non-negative evidence; alpha = evidence + 1 | `dirichlet_statistics`, `UNet2D.forward` | Consistent |
| Vacuity uncertainty u = K / sum(alpha) | `dirichlet_statistics` | Consistent |
| Target-exempted alpha = 1 + (1-y)(alpha-1) | `target_exempted_alpha` | Consistent |
| Pixel-wise JS divergence between image/text probabilities | `jensen_shannon_divergence` | Consistent |
| UGT gate omega = u * exp(-gamma d) | `uncertainty_gate` | Consistent |
| Pseudo-boundary from a class transition in local neighborhood | `dynamic_boundary_mask` | Consistent |
| Teacher weak view aligned into Student strong view with validity mask | `recp_step` aligner contract | Consistent |
| Total objective: supervised + lambda_ugt consistency + lambda_bfcl contrast | `recp_step` | Consistent |
| 256 x 256, K=2, 10% default labeled ratio | `configs/review.yaml` | Consistent |
| AdamW, lr 2e-4, weight decay 1e-4, batch 32 | `configs/review.yaml` | Consistent |
| 300 epochs, 15-epoch warm-up, EMA decay 0.99 | config and schedule/EMA utilities | Consistent |
| lambda_ugt max 1.0, lambda_bfcl max 0.3, gamma 0.5 | `configs/review.yaml` | Consistent |
| CLIP ViT-B/32 and three prompt pairs | Documented only; embeddings withheld | Not implemented by design |
| Complete TEER/UGT/BFCL objectives and sampling | Restricted interfaces | Not implemented by design |
| Reported 5.72M/7.05M parameter models and measured results | Compact API model only | Not reproduced by design |

