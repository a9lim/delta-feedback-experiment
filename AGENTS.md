# AGENTS.md

Research repo: pretraining factorial combining Delta Attention Residuals
(depth-axis routing over sublayer deltas) with full-bandwidth latent feedback
(previous column's top state fed back to layer 0). Independent of every
sibling experiment. Currently in design phase — the ledger is settled, the
harness is not yet written.

## Read first

- [`docs/design.md`](docs/design.md): the decision ledger (D1-D18). Settled
  decisions are not relitigated silently — reopen them as new numbered
  entries with a reason.
- [`README.md`](README.md): status, plan shape, install.

## Operating notes

- **Greenfield contract:** predecessors (stateful-thought-experiment,
  chain-of-dots-experiment) were deliberately cleared by a9. Do not dig up or
  inherit their design decisions.
- **One recipe everywhere** (D10): FBT's training recipe is binding for every
  arm including vanilla; DAR module conventions nest inside it. Divergences
  from the papers must be recorded in the ledger.
- **Paired comparisons are load-bearing** (D13): arms share data, batch
  order, and seeds; effect sizes (~few % PPL) are near noise otherwise.
  Report matched-tokens and matched token-equivalent compute (an n-pass
  batch costs n).
- **Contraction diagnostic** (D8): iterated fused prefill passes with
  ||h(k) - h(k-1)|| decay is a standing monitor — run it before trusting any
  feedback-arm result, and as the stability gate before the flagship.
- **Machines:** trials and debug on jobe (`ssh jobe`, 1x4090, torch pinned
  2.8.0+cu128 — respect the flash-attn wheel constraint); analysis and
  probing on the Mac (MPS). Flagship on Prime Intellect marketplace pods
  (D14) — checkpoint-tolerant discipline if on spot instances.
- **Promotion gate** (D16) is pre-registered; the flagship spend (~$6-7k
  H100-rate) is not authorized by default — confirm with a9 at promotion
  time with trial evidence in hand.
- Smoke, then pilot, then scale. A null on the FBT side at trial scale is a
  formation-conditions finding, not a failure (the DAR arm is the positive
  control).
