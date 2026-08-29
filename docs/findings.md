# Findings

**No scientific result yet.** The design and training harness exist, but no
screen arm has completed enough training to support an architecture claim.

## Jobe execution evidence

The authoritative 220M DF path was exercised on Jobe's RTX 4090 at B=4,
T=1025, including graph preparation, all reachable k/z-loss modes, optimizer
updates, a complete 292-row real-data step, evaluation, route reporting, the
contraction monitor, and snapshot writing. The three retained modes are k=1
without z-loss, k=2 with z-loss, and k=3 with z-loss. k=3 is automatically
checkpointed so its retained graph and persistent optimizer state coexist;
DF-soft also checkpoints k=2 because its standing payload/null sources enlarge
the route bank.

Steady full-step compute on the DF arm measured 5.87 s at k=1, 13.27 s at k=2,
and 23.81 s at k=3; NorMuon plus Adam took about 47 ms. Fixed-graph peak
allocation with optimizer state resident was 15.5 GiB. The default exact pass
schedule therefore projects to roughly 15 GPU-hours before evaluation and
checkpoint overhead. This is systems evidence only, not evidence for the DF
hypothesis; cold graph preparation was about 11-15 seconds with a warm compiler
cache and is outside step timing.

The live CUDA gate also passed every arm at screen geometry. BF16/TF32, CCE's
gradient filter, algebraic router reduction order, packed NorMuon matrices, and
FlashAttention deliberately change the numerical training stream. The semantic
gates remain uniform zero-init routing, telescoping residuals, causal multi-pass
feedback, live feedback gradients, finite updates, and the contraction monitor.

Scientific claims will appear here with limitations attached once trial runs
produce audited numbers; `data/summary/` carries compact machine-readable
results and `figures/README.md` the figure map.
