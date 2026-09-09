# What we have seen so far

This page is the distilled current picture of the specimens. It is rewritten
as the picture changes; the dated detail behind it is in
[journal.md](journal.md), and the numbers below come from the 2026-09-08
analysis of the paired `ar` and `arf` runs unless stated otherwise.

## Specimens

| Run | Condition | Recipe | NorMuonH lr | val | val_fused | Where |
|---|---|---|---:|---:|---:|---|
| `screen-delta-ar-s1` | `ar` | default (one pass throughout) | 6e-3 | 3.069 | | Jobe |
| `screen-delta-arf-s1` | `arf` | `--feedback-start 0 --three-pass 1` | 6e-3 | 3.076 | 3.077 | Jobe |
| `screen-delta-arf-s1-highLR` | `arf` | `--feedback-start 0 --three-pass 1` | 2e-2 | 3.113 | 3.113 | Jobe |

All are seed 1, data seed 0, 10,745 steps, 320 rows of 1,024 predictions per
step, on the same rows. The three-pass-always `arf` runs spend 10.56B
pass-tokens against `ar`'s 3.52B; steps are matched data, not matched
compute. Only the `ar` run coincides with the current defaults.

## The channel as trained

- **Fused equals plain.** On 256 held-out rows the fused pass (plain prefix
  length 1) costs +0.0012 ± 0.0006 nats over pass 1; a second fused iteration
  costs another +0.0016. Fused wins 50.4% of tokens by tiny margins. The
  30-iteration self-composition contracts cleanly (relative top-state update
  0.16, 0.037, 0.013, 0.009, then flat).
- **Cell 0 cancels the fused seed.** The fused seed enters at 23x the plain
  seed's RMS. Block 0's delta changes by 3.7x its own RMS and is orthogonal
  to its plain-pass self (cosine −0.02); blocks 1 and 2 and `h_top` change by
  0.15–0.24x with cosine 0.97–0.99. The first cell maps the fused seed back
  onto the plain representation and the rest of the column runs plain.
- **The payload is the readout basis.** Ridge R² from `final_norm(h_top)` to
  the payload is 0.978. The token is recoverable from the fused seed (ridge R²
  0.78, nearest-embedding hit 98%). The gate's token-dependent share of the
  pre-norm entry input is 15% at the median, 36% at p90.
- **Routing.** Early MLP sites read the null (0.78–0.89) where `ar`'s read
  the partial and seed; on the fused pass L0.mlp reads the fused seed at 0.86
  and the top attention sites shift mass from block 0 to the seed. The payload
  router's head 0 reads block 0 on pass 1 and the seed on pass 2; one head
  sits on its null. Per-head ablation costs at most +0.006; forcing every head
  to the null +0.023; removing the routed term +3.3.
- **The entry is fully co-adapted.** A constant gate costs +8.4, gate
  temperature 2 costs +0.09, seed scale 0.5 or 2 costs +7.3 or +2.5, a unit
  embedding bypass +0.023, a payload from another row +0.46, a zero payload
  +32.

Reading: the channel is live, stable, and harmless, and its content is the
current token and the readout state. The previous column's state is
load-bearing only as the carrier through which the current token is decoded.
That is the opposite of an opaque cross-column channel.

## Recipe observations

- **Dense exposure closed the fused deficit.** The 75/22/3 recipe left a
  +0.024 fused gap with a decode-cost signature on rare and surprising fused-in
  tokens; training three passes from step 0 shrank it to +0.0012 with the same
  signature at three to ten times smaller amplitude. No fused benefit formed
  on FineWeb tokens.
- **The plain-mode tax is small.** `arf` pass 1 is +0.0025 ± 0.0013 nats worse
  than `ar` on the same rows at 3x the pass-tokens. On the paired training
  trace the tax is +0.0054 over the first 37% of the schedule, about zero
  through the second half of the heat, and +0.0013 in the cooldown.
- **Two paired runs differ like two seeds.** KL(`ar` ‖ `arf` pass 1) is
  0.215 nats, argmax agreement 77.6%; every shared trunk matrix sits 85–87°
  from its paired initialization and 87° from its twin. Per-token differences
  between two runs are not attributable to a package.
- **NorMuonH 2e-2 is too hot.** The same recipe at 2e-2 finished 0.037 nats
  worse than at 6e-3, with its paired pass-1 training loss running 0.22 above
  `ar` through the second half of the heat. 6e-3 is now the default.
- **The specimens' training curves carry the source's crawl order.** Every
  run's detrended pass-1 training loss swells and dips with a 742-step period
  and a 0.1-nat peak-to-trough, identical across runs (residual correlation
  0.99 at lag zero between any two). The source parquet files are single-crawl
  runs of about 60M tokens read in order, so each stretch of training was one
  CommonCrawl dump, and the held-out slice was the head of one 2013 crawl.
  The stream is now a keyed document shuffle of `sample-350BT` with the base
  tokenizer's EOS, and the parametrization is muP-pinned to the flagship
  width; the three specimens above trained on the old order under the plain
  screen parametrization, and the current code does not load their snapshots.

## Downstream

Both 6e-3 specimens beat the 300B-token `pythia-160m` on the educational and
science-flavored zero-shot tasks (ARC-Easy by 8–9 points, HellaSwag by 4,
OpenBookQA by 5) and trail it badly on LAMBADA (30 against 35 accuracy), which
is fiction and long-range. `ar` against `arf` Standard is indistinguishable
apart from a yes-bias on BoolQ.

Within `arf`, Fused against Standard on identical documents: LAMBADA +1.47 ±
0.36 accuracy, HellaSwag margin +0.16 ± 0.02 nats, and 0.05–0.3 nats less
probability on every short answer after a QA prompt. The gain is front-loaded:
a second fused prefill pass gives nothing more. Soft mode (feedback only along
the scored continuation) equals Standard on LAMBADA to four decimals, so the
LAMBADA gain is context refinement, not the transition into the scored token.
On HellaSwag about 70% of the margin gain survives with a plain context.

## What this says about growing

The recipe change (dense feedback from step 0) was the right fix for the fused
deficit, and it exposed that the channel on this trunk and data has nothing
opaque to carry: FineWeb next-token prediction is served well enough by the
current token plus the mixer caches, so the model learns to route the token
through the payload and cancel the rest. Candidate moves, none chosen:

- Land the previous column's state somewhere the current column cannot cancel
  in one cell, for example at a core entry with the FBT gate kept at the pass
  entry.
- Build the `l` letter ([architecture.md](architecture.md#letter-l-the-tied-depth-loop)) so
  within-column refinement exists and persists across columns through the
  payload and a shared core cache.
- Split a persistent state stream from a read-only prediction stream
  ([literature](literature.md#free-pause-tokens-and-the-loop)).
- Add a task that needs cross-column latent state.
