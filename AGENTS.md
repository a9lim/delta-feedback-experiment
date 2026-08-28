# AGENTS.md

Research repo: pretraining factorial combining Delta Attention Residuals
(depth-axis routing over sublayer deltas) with full-bandwidth latent feedback
(previous column's top state fed back to layer 0). Independent of every
sibling experiment. Currently in design phase — the design is settled, the
harness is not yet written.

## Documents

- [`docs/design.md`](docs/design.md) is the authoritative current
  architecture, training, evaluation, scale plan, and gates. It defines the
  current experiment only — keep it current-only, no decision history.
- [`docs/findings.md`](docs/findings.md) contains only evidence valid for
  the current design.
- [`docs/journal.md`](docs/journal.md) is disposable working space,
  periodically cleared; durable content graduates to design or findings.
- [`references/refs.yaml`](references/refs.yaml) carries source roles.

Update code, docs, and tests together when the design changes; do not keep
alternate schemas or stale guidance around.

## Operating notes

- **Greenfield contract:** predecessors (stateful-thought-experiment,
  chain-of-dots-experiment) were deliberately cleared by a9. Do not dig up
  or inherit their design decisions.
- **One recipe everywhere** (design: Training): FBT's training recipe is
  binding for every arm including vanilla; DAR module conventions nest
  inside it. Record any divergence from the parent papers in design.md when
  it is made.
- **Paired comparisons are load-bearing** (design: Arms): arms share data,
  batch order, and seeds; effect sizes (~few % PPL) are near noise
  otherwise. Report matched tokens and matched token-equivalent compute (an
  n-pass batch costs n).
- **Contraction diagnostic** (design: Training / Evaluation): iterated
  fused prefill passes with ||h(k) − h(k−1)|| decay is a standing monitor —
  run it before trusting any feedback-arm result, and as the stability gate
  before the flagship.
- **Machines:** trials and debug on jobe (`ssh jobe`, 1×4090, torch pinned
  2.8.0+cu128 — respect the flash-attn wheel constraint); analysis and
  probing on the Mac (MPS). Flagship on Prime Intellect marketplace pods —
  checkpoint-tolerant discipline if on spot instances.
- **Promotion gate** (design: Gates) is pre-registered; the flagship spend
  (~$6–7k H100-rate) is not authorized by default — confirm with a9 at
  promotion time with trial evidence in hand.
- Smoke, then pilot, then scale. A null on the FBT side at trial scale is a
  formation-conditions finding, not a failure (the DAR arm is the positive
  control).
