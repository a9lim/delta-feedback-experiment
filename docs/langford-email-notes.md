# Notes for an email to John Langford on the FBT fusion entry

Starting point only. Written 2026-09-14 while `screen-delta-f-s1` was still
in its heat phase; refresh the numbers before sending. Tags: [standard]
is in the cited paper, [synthesis] is our reading, [speculation] is a
guess neither we nor they have tested.

## The one-paragraph version

- Section 3.1 of the Full-bandwidth Transformer (arXiv 2608.08888) chooses
  an asymmetric GLU entry, hidden state on the value path and token
  embedding only as a sigmoid gate, over any symmetric fusion. The stated
  reason is a shortcut: a symmetric entry lets the model suppress the state
  path, recover the plain token input, and sit at ordinary pretraining loss
  with the wide channel unused. [standard]
- DeepSeek-V3's MTP module (arXiv 2412.19437, Eq. 21) already contains that
  symmetric fusion: RMSNorm'd hidden state concatenated with the RMSNorm'd
  next-token embedding, one linear projection. Trained as MTP, that
  projection cannot drop the hidden-state block without losing most of the
  MTP objective. [standard for the form, synthesis for the identification]
- We share one projection between the MTP entry and the feedback entry.
  The MTP loss then pins the state block of the feedback fusion in every
  pass and every condition, including single-pass batches, so the
  Section 3.1 shortcut is closed by an objective rather than by the
  parameterization. The token keeps an additive path into the residual,
  which the GLU denies it. [synthesis]

## What our entry looks like

- Seed at a feedback position: `seed_t = W_e e_t + W_p p_(t-1)`, one
  bias-free `2D -> D` matrix, no gate, no activation, no second norm.
  `e_t` is the raw token embedding; `p_(t-1)` is the writer's payload,
  a learned RMSNorm of the top state plus a routed cell-delta read, with
  fixed output scale `0.02` matching embedding initialization.
- MTP input at the same position, same matrix: `u_t = W_e e_(t+1) +
  W_p (p_t + jitter_t)`, target `x_(t+2)`, through one independent
  PKDA/MoE block with its own sequence mixer, tied readout, loss weight
  `0.3`. Feedback shifts `u_t` right by one to seed the next pass.
- Differences from DeepSeek Eq. 21 worth stating: the token side is raw,
  the payload is normalized once at the writer, and the fused tensor is
  reused as the next pass's seed rather than discarded after MTP.
- Reference: `docs/architecture.md`, sections "Payload and letter f" and
  "Two-token prediction".

## Why the pin works, stated carefully

- With `W_p = 0` the MTP head is a one-layer language model over shifted
  raw embeddings, since the auxiliary block has its own mixer. That is a
  multi-nat penalty, not a loss-free shortcut. Do not describe it as "one
  token of context". [synthesis]
- The pin holds the fusion matrix, not trunk use. The trunk can still
  un-mix the seed one block later. The precise claim is that the shortcut
  named in Section 3.1, suppressing the state path at the entry, is
  closed. Whether the channel survives past the entry is a separate
  measurement (below). [synthesis]
- Trained numbers say the pinned block carries near-full state: the MTP
  head sits about a quarter nat above the trunk's own next-token loss.

## Why the paper likely did not see it

- FBT's footnote 1 lists MTP, JTP, and next-latent prediction as extra
  supervision "compatible with this scheme" and cites Gloeckle et al. 2024
  for MTP. Gloeckle heads read the hidden state alone, with no next-token
  input, so there is no fusion in that variant to pin. The DeepSeek variant
  is the one whose entry coincides with the feedback entry. [synthesis]
- NextLat (arXiv 2511.05963, Teoh et al., same group) fuses the current
  hidden state with the next token to predict the next hidden state, but
  in a separate lightweight dynamics model, not the model's own entry.
  HiLP (arXiv 2608.05806) does not touch fusion. The pieces sit in three
  papers without the identification. [synthesis]
- Optional, and mark it as a guess if used: the GLU does not make the
  plain-token input unrepresentable either. If the value path carries a
  constant direction, the gate alone reproduces a bounded function of the
  token embedding. Both designs rest on training dynamics, which favors a
  pin by objective. [speculation]
- The paper offers no ablation of gate versus add versus concat; the
  choice is argued, not measured. [standard]

## What we have measured

Under the previous gated entry (`screen-delta-arf-s1`, 2026-09-08 paired
comparison against `screen-delta-ar-s1`; the gate has since been replaced
by the concat projection):

| quantity | value |
|---|---|
| val gap, feedback arm minus plain arm, pass 1 | +0.0025 ± 0.0013 |
| zero payload at the entry | +32 nats |
| shuffled payload | +0.46 nats |
| constant gate | +8.4 nats |
| token recoverable from seed | 98% |
| payload vs readout state, ridge R² | 0.978 |
| cell 0 delta vs its plain self, CKA | 0.57 |
| cells 1–2 and top state vs plain, CKA | 0.97–0.99 |
| Fused vs Standard, LAMBADA acc | +1.47 ± 0.36 |
| Fused vs Standard, HellaSwag margin | +0.16 ± 0.02 |

- Reading: entry fully load-bearing, trunk mostly reverted to the plain
  computation after cell 0, payload token-legible. This is the "survives
  past the entry" caveat, and it is honest to include it.
- FBT Table 2 (1B, 200B tokens, 0-shot, 1 pass) shows about +0.9 average
  on five common tasks; our five-task pooled shift on the same tasks was
  about zero. Different scale and recipe; state it only as context.

Under the concat projection (`screen-delta-f-s1`, in flight on jobe,
snapshot 2026-09-14 evening, heat phase, step 3,659 of 9,975, feedback
from step 0, two or three passes per step):

| quantity | value |
|---|---|
| pass-1 next-token loss, train | 3.25 |
| MTP loss per pass, train | 3.48 |
| val, step 3,500 | 3.222 |
| val_fused, step 3,500 | 3.213 |
| val_mtp minus val, step 3,500 | +0.247 |

Fused-minus-pass-1 validation gap by step, paired on the same rows and
the same model, so between-run seed noise does not apply:

| step | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|---|---|
| val_fused − val | +0.017 | +0.007 | +0.003 | −0.001 | −0.003 | −0.004 | −0.007 | −0.008 | −0.009 |

- The gap crossed zero near step 1,500 and has widened monotonically
  since, with `val_mtp_fused` tracking it (−0.005 at step 3,500).
- For contrast, the two earlier runs had the fused pass worse than pass
  1 at the end: +0.024 (`screen-df-s1`, feedback only in the final 25%)
  and +0.0012 ± 0.0006 (`screen-delta-arf-s1`, gated entry, feedback
  from step 0). This is the first sign flip, and it is the "survives
  past the entry" evidence the gated run lacked.
- The entry changed in several ways at once between the gated run and
  this one: gate removed, payload normalized and scaled at the writer,
  raw token embeddings, plus the common recurrence schedule. The sign
  flip is not attributable to the shared projection alone. Say so.

## Before sending

- Let `screen-delta-f-s1` finish (about two days from the snapshot).
- Run the zero-payload and shuffled-payload sweeps on the finished
  checkpoint, `scripts/payload_swap.py` and `scripts/route_report.py`,
  plus the Fused-versus-Standard comparison. The zero-payload number under
  the symmetric entry is the single figure the email turns on.
- Replace the in-flight table above with the finished numbers.
- Decide whether to include the constant-direction loophole; it is a
  guess about their design, so either omit it or label it.

## Suggested shape

- One line on who we are and that this is a small note, not a request.
- The identification: DeepSeek Eq. 21 is the symmetric entry Section 3.1
  argues against, and MTP training pins its state block.
- The construction: one shared projection for MTP and feedback; why the
  Section 3.1 shortcut is closed; what stays open.
- The numbers: finished zero-payload sweep, MTP-minus-NTP gap, and the
  past-the-entry caveat from the gated run.
- A question worth their answer: did they try a symmetric entry with any
  MTP-style supervision, and if so what happened at the warm start.
- Offer the repo and the checkpoint if useful.

## References

- Wang et al. 2026, Full-bandwidth Transformer, arXiv 2608.08888,
  Section 3.1 (entry, Eq. 4), Section 3.3 (schedule, footnote 1).
- DeepSeek-AI 2025, DeepSeek-V3 Technical Report, arXiv 2412.19437,
  Section 2.2 (MTP, Eq. 21).
- Teoh et al. 2025, Next-Latent Prediction Transformers Learn Compact
  World Models, arXiv 2511.05963.
- Ahn, Lamb, Langford 2025, Efficient Joint Prediction of Multiple Future
  Tokens, arXiv 2503.21801.
- Shi et al. 2026, Hierarchical Latent Prediction for Language Models,
  arXiv 2608.05806.
- Gloeckle et al. 2024, Better & Faster Large Language Models via
  Multi-token Prediction, arXiv 2404.19737.
