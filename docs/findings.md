# Findings

**No scientific result yet.** No MHDAR x FBT screen arm has run far enough to
support an architecture claim.

## Implementation evidence

The portable MHDAR implementation passes the offline invariant suite on the
Mac. The checked surface includes independent head choices, full-width
RMSNorm semantics, uniform zero-init routing in every head, telescoping
residuals, causal multi-pass feedback, feedback gradients, cached decoding,
and exact resume behavior under checkpoint contract v4.

The authoritative Jobe gate passes on the RTX 4090. Direct bespoke-router
checks cover H=4 with prefix masking and H=8 at the flagship width (D=1536)
with the DF-soft null-source pointer path; values, per-head weights, query/key
gradients, and every source gradient match the portable implementation within
the registered BF16 bounds. The full 220M DF path then passes exact CCE-native
z-loss parity, cached FlashAttention decoding, captured/eager evaluation
parity, all four reachable shared-pool train/eval CUDA graphs, finite optimizer
updates, and production-scale contract-v4 checkpoint staging.

On the final warm-cache gate, preparation took 45.1 s, peak allocation was
9.93 GiB, and cached-decoding relative drift was 0.0077. B=4, T=1025 replay
times were 64.9 ms for k=1, 174.0 ms for k=2, and 240.7 ms for k=3. These are
systems measurements only, not evidence for the MHDAR x FBT hypothesis. The
earlier single-head contract-v3 measurements have been retired.
