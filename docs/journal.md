# Working journal

Use this page for disposable hypotheses, measurements, and unresolved
interpretations. Identify diagnostic checkpoints by their original contracts;
do not relabel them as current registered runs. Promote a result only with its
controls and artifacts under [the evidence
rules](interpretability.md#deliverables-and-acceptance). Git history retains
superseded working notes.

## 2026-09-08 — `mhdb` against full-feedback `df`: what the feedback package changed inside

Two completed diagnostic runs on Jobe, paired by construction (seed 1, data
seed 0, identical rows and schedule, NorMuonH `6e-3`, 10,745 steps):

| run | arm | passes | pass-tokens | s/step | val (32 rows) | val_fused |
|---|---|---|---:|---:|---:|---:|
| `screen-df-mhdb-s1-lowLR` | `mhdb` | 1 throughout | 3.52B | 4.9 | 3.069 | — |
| `screen-df-full-s1-lowLR` | `df` | 3 from step 0 | 10.56B | 14.2 | 3.076 | 3.077 |

The `df` run is the fresh-run form of the question left open on 2026-09-02:
does the FBT benefit form when the fused mode is trained densely from the
start? It is a recipe variant (`--feedback-start 0 --three-pass 1`), not a
registered arm, and equal steps here are 3x the compute. Nothing below is a
training-effect claim; it is one paired observation plus same-checkpoint
interventions on the final snapshots. Scripts: `scripts/compare_arms.py`,
`weight_divergence.py`, `fused_diagnostics.py`, `entry_sweeps.py`,
`feedback_followups.py`, `payload_swap.py`, `route_report.py`,
`training_curves.py`, `analysis_figures.py`; records and figures under
`figures/curves-mhdb-vs-df-full-lowLR/`,
`figures/compare-screen-df-mhdb-s1-lowLR-vs-screen-df-full-s1-lowLR/`,
`figures/weights-…/`, `figures/fused-screen-df-full-s1-lowLR/`, and
`figures/route-…/` (ignored); run logs copied to `logs/`.

### The gap is small and mostly a small-sample draw

On 256 held-out rows (262k tokens; the first 32 reproduce the logged numbers
to the third decimal):

| predictor | CE | difference | s.e. |
|---|---:|---:|---:|
| `mhdb` pass 1 | 2.8862 | | |
| `df` pass 1 | 2.8887 | +0.0025 vs `mhdb` | 0.0013 |
| `df` fused (prefix 1) | 2.8899 | +0.0012 vs `df` pass 1 | 0.0006 |
| `df` fused, second iteration | 2.8915 | +0.0016 vs fused | |

Top-1 accuracy is 42.92% for both pass-1 predictors. The per-step pass-1
training loss on identical rows (exact pairing, EMA 300) was +0.0054 for the
first 37% of the schedule, −0.0007 through the second half of the heat, and
+0.0013 in the cooldown: the plain-mode tax of serving two input regimes is
real, small, and largest while the fused mode is forming.

### The two arms differ like two seeds

KL(`mhdb` ‖ `df` pass 1) is 0.215 nats, argmax agreement 77.6%, per-token CE
correlation 0.972; a split-validated 50/50 probability mixture of the two
gains 0.049 nats. Within `df`, KL(pass 1 ‖ fused) is 0.039, agreement 89.4%,
mixture gain 0.011. In weight space every shared trunk matrix sits 85–87°
from the paired initialization and 87° from its twin in the other run
(cosine 0.06–0.08; total-update cosine about 0.5), while norms, router
vectors, and PKDA vectors stay within a few degrees and the embedding at
cosine 0.77. Under fixed-radius normalized updates the two trajectories
separate like different seeds, so the +0.0025 is a mean shift riding on a
±0.24 interquartile per-token spread and no per-token difference is
attributable to the package. Conditioning on the reference's own loss shows
the expected regression toward the mean (`df` looks better exactly where
`mhdb` is worst, and worse where `mhdb` is confident); that panel is an
artifact of the conditioning, so the tables condition on the reference's
entropy and on token frequency instead.

Where the +0.0025 concentrates: positions 1–15 carry +0.02 to +0.06 (few
tokens, 1–2 s.e.), positions beyond 64 carry +0.0015 to +0.002; rare target
tokens +0.012, frequent ones −0.001 to +0.002; input-frequency and surprise
bins are flat within noise.

### Dense exposure closed the fused deficit and formed no benefit

Against `screen-df-s1` (75/22/3 recipe, +0.024 fused gap):

| diagnostic | `screen-df-s1` | `screen-df-full-s1-lowLR` |
|---|---:|---:|
| fused − pass 1 | +0.024 | +0.0012 |
| KL(pass 1 ‖ fused) / argmax agreement | 0.057 / 87.5% | 0.039 / 89.4% |
| fused wins on | 47.6% of tokens | 50.4% |
| split-validated mixture gain | 0.006 | 0.011 |
| rarest 10% input tokens / most frequent 10% | +0.047 / +0.011 | +0.015 / −0.004 |
| fused-in token in the top 1% of surprise | +0.085 | +0.024 |
| positions 1–3 / 4–15 / 64+ | +0.065 / +0.058 / +0.024 | +0.025 / +0.014 / +0.001 |
| second fused iteration | +0.007 worse | +0.0016 worse |
| gradient cosine, pass 1 vs fused (trunk matrices / all shared) | 0.75 / 0.86 | 0.82 / 0.90 |

The plain-prefix-512 and random-prefix fused passes agree with prefix 1 to
±0.001, jitter changes nothing, and the 30-iteration self-composition
contracts cleanly (relative top-state update 0.16 → 0.037 → 0.013 → 0.009 and
flat; loss within 0.001 of pass 1 on 8 rows). The residual cost keeps its
decode signature (rare and surprising fused-in tokens, rarely fused early
positions) at three to ten times smaller amplitude. The benefit side did not
appear: fused wins half the tokens by tiny margins, iterating hurts, and the
mixture gain of 0.011 is the whole complementary information.

### Mechanism: the first cell cancels the fused seed, the rest runs plain

On the same tokens, `df`'s fused pass differs from its plain pass almost only
in cell 0. The fused seed enters at 23x the plain seed's RMS (0.85 against
0.04); the first completed block delta then changes by 3.7x its own RMS and is
orthogonal to its plain-pass self (cosine −0.02, CKA 0.57, RMS 0.76 ≈ the
seed's), while block 1 changes by 0.24x (cosine 0.97, CKA 0.974), block 2 by
0.15x (0.988, 0.992), and `h_top` by 0.15x (0.987, 0.992). Cell 0 is a decoder
that maps the fused seed back onto the plain representation; cells 1–2 then
compute what they compute on a plain pass. Between the two arms the same
sources have CKA 0.92 (block 0), 0.81 (block 1), 0.95 (block 2 and `h_top`)
on plain passes: the middle cell is where the two trainings diverged most.

Routing is consistent with that: `df`'s early MLP sites read the null on
both passes (L1–L3.mlp 0.78–0.89, L4.attn 0.84–0.90) where `mhdb`'s read the
partial and seed (0.44–0.75); on the fused pass `df`'s L0.mlp reads the fused
seed at 0.86, and the top attention sites switch from block 0 to the seed
(L9/L10/L11.attn seed mass 0.25/0.23/0.00 → 0.39/0.38/0.53), a direct read of
the previous column's state just before the readout. The payload router's
head 0 reads block 0 on pass 1 and the seed on pass 2 at weight 1.0, one head
sits on its null (a learned constant of RMS 0.33), and per-head ablation costs
at most +0.006; forcing every head to the null costs +0.023, removing the
routed term costs +3.3. The payload itself is linearly the readout state:
ridge R² 0.978 from `final_norm(h_top)`, 0.96 from raw `h_top`, 0.69 from the
other arm's readout state. The entry is fully co-adapted: a constant gate
costs +8.4, gate temperature 2 costs +0.09, seed scale 0.5 or 2 costs +7.3 or
+2.5, a unit embedding bypass +0.023, a payload from another row +0.46, a zero
payload +32. The token is recoverable from the fused seed (ridge probe R²
0.78, nearest-embedding hit 98%); the gate's token-dependent share of the
pre-norm input is 15% at the median (36% at p90, up from 8% / 15%).

### Reading

1. With every step three-pass from step 0, the feedback package costs the
   plain mode +0.0025 ± 0.0013 nats at 25 tokens per parameter and 3x the
   pass-tokens; the logged +0.007 was a 32-row draw. The paired training
   trace puts the cost early and in the cooldown, not in a competition
   signature.
2. Dense exposure answered the 2026-09-02 question: the fused deficit was a
   recipe artifact (0.024 → 0.001), and the FBT benefit still does not form
   on this trunk at this scale. The model neutralizes the fused seed in cell
   0 and reproduces the plain computation; the previous column's state is
   load-bearing only as the carrier through which the current token is
   decoded.
3. For the organism the channel is stable, harmless, and token-legible: the
   payload is the readout basis and the seed decodes the token at 98%. That is
   the opposite of an opaque cross-column channel. The 2026-09-02 decision
   tree's "redesign the entry" branch is the one this selects; whether to
   take it, and whether the `mhdb` run's coincidence with the current
   defaults makes it a registered specimen, are decisions for a9.

Repo notes: checkpoint-analysis helpers now live in
`delta_feedback_experiment.analysis` (loader, trainer numerics, per-token
losses, fused inputs; `tests/test_analysis.py`), the September-2 scratch
scripts were promoted to `scripts/` under the names above with a shared
`scripts/figstyle.py`, and their raw outputs moved to
`figures/fused-screen-df-s1/raw/`.
