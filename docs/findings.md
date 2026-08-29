# Findings

**No scientific result yet.** No MHDAR x FBT screen arm has run far enough to
support an architecture claim.

## Implementation evidence

The portable MHDAR implementation passes the offline invariant suite on the
Mac. The checked surface includes independent head choices, full-width
RMSNorm semantics, uniform zero-init routing in every head, telescoping
residuals, causal multi-pass feedback, feedback gradients, cached decoding,
and exact resume behavior under checkpoint contract v4.

The earlier single-head contract-v3 CUDA measurements are not evidence for
this architecture and have been retired. Current CUDA execution evidence will
be recorded here only after the MHDAR bespoke forward/backward and the full
220M graph-capture gate pass on Jobe.
