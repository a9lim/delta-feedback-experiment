# Findings

**No scientific result yet.** The design and training harness exist, but no
screen arm has completed enough training to support an architecture claim.

## Jobe execution evidence

The authoritative 220M DF path was exercised on Jobe's RTX 4090 at B=4,
T=1025, including four shared-pool training/evaluation graphs, every reachable
k/z-loss mode, optimizer updates, exact CCE-native z-loss parity, bespoke-route
forward/backward parity, cached decoding, and production-scale pinned-host
checkpoint staging. The retained modes are raw k=1 without z-loss, raw k=2
with z-loss, and raw k=3 with z-loss; DF-soft still checkpoints k>=2 because
its standing payload/null sources enlarge the route bank.

The warm-cache CUDA gate measured 66.2 ms at k=1, 176.4 ms at k=2, and 244.5
ms at k=3 per B=4 microbatch, with persistent optimizer state and evaluation
graph resident. Raw k=3 reduced replay from 284.3 ms checkpointed to 244.5 ms,
while peak allocation rose only from 9.63 to 9.87 GiB. Preparation took 44.9 s
on the retained raw candidate. At 73 microbatches per 292-row step, the default
75/22/3 schedule projects to about 13 GPU-hours of forward/backward compute
before optimizer, evaluation, diagnostic, and snapshot overhead. This is
systems evidence only, not evidence for the DF hypothesis.

The live CUDA gate also passed every arm at screen geometry. BF16/TF32, CCE's
gradient filter and fused exact z-loss backward, the bespoke router's FP32
reduction order, packed NorMuon matrices, whole-block compilation, and
FlashAttention deliberately change the numerical training stream. The semantic
gates remain uniform zero-init routing, telescoping residuals, causal multi-pass
feedback, live feedback gradients, finite updates, and the contraction monitor.

Scientific claims will appear here with limitations attached once trial runs
produce audited numbers; `data/summary/` carries compact machine-readable
results and `figures/README.md` the figure map.
