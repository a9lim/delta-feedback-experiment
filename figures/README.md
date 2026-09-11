# Figures

There are no current trained-run figures. New analyses require a v28
checkpoint and a matching GPT-NeoX/ChatML token store.

Analysis scripts write their figures and a JSON record beside them under
`figures/<kind>-<tag>/`. Those directories are ignored by git and regenerate
from the snapshots and logs; the commands are in
[operations.md](../docs/operations.md#inspect-a-checkpoint).

| Directory | Written by | Contents |
|---|---|---|
| `route-<tag>/` | `scripts/route_report.py` | Per-site/group source mass, entropy, query geometry |
| `fused-<tag>/` | `scripts/fused_diagnostics.py`, `entry_sweeps.py`, `feedback_followups.py`, `payload_swap.py`, `dense_feedback_continue.py` | Fused-pass structure, entry interventions, impulse response, ablations, payload swaps |
| `compare-<A>-vs-<B>/` | `scripts/compare_conditions.py`, `analysis_figures.py` | Paired per-token loss structure, predictor divergence, CKA, payload redundancy, and the composed panels |
| `weights-<A>-vs-<B>/` | `scripts/weight_divergence.py` | Angular movement of shared parameters |
| `curves-<A>-vs-<B>/` | `scripts/training_curves.py` | Paired validation and training curves, matched compute, routing and contraction monitors |
| `downstream-<tag>/` | `scripts/downstream_eval.py` | Zero-shot task records per mode, with paired comparisons beside them |
| `depth-<tag>/` | `scripts/depth_trace.py` | Loss and core update size by iteration count, plain and fused, and core router mass by iteration |

Anything meant to survive a regeneration, such as an architecture diagram,
goes under `figures/architectures/`, which is tracked.
