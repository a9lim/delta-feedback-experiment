# Optional scale and training-duration references

The small Jobe model is the primary research organism. These specifications
preserve exact budgets for a possible longer-trained or larger specimen; they
are not a capabilities roadmap, scheduled jobs, or prerequisites for starting
interpretability work. The distributed paths remain unimplemented and
unqualified. Nothing here authorizes provisioning or a training launch.

Use more training, width, depth, or context only to answer a named question
that the smaller specimen cannot resolve. First show what is missing: for
example, a behavioral task never acquired, a recurrent channel that remains
unused, or a mechanism whose dependence on training duration needs a paired
test. A lower language-model loss alone is not the reason to scale. A useful
small organism is a valid endpoint for this project.

The terms “screen” and “25x/400x” name controlled training recipes. The ratio
counts predicted tokens per active non-embedding parameter, not reasoning
ability or interpretability. Every feedback pass contributes to compute.

## Longer training at the same model size

If a documented interpretability question requires longer training after an
admissible Jobe result, only `{base, df}` is pretrained from fresh paired
initialization on one 8xH100-80GB Prime node. Both arms use seed 1, data seed
0, global row zero, 171,909 steps, and 56,331,141,120 predicted tokens. Neither
loads a Jobe model or optimizer checkpoint.

| Arm | Exact predicted tokens | Active-token ratio |
|---|---:|---:|
| `base` | 56,331,141,120 | 403.551759 |
| `df` | 56,331,141,120 | 399.999741 |

The fresh WSD schedule is:

| Phase | Steps | Pass behavior |
|---|---:|---|
| Warmup | 1–3,438 | one pass |
| Stable heat | 3,439–137,527 | one pass through 128,932; then feedback arms draw 2 or 3 passes |
| Cooldown | 137,528–171,909 | feedback arms draw 2 or 3 passes |

The pair consumes 112.662B predicted tokens and approximately 128.435B expected
pass-tokens. It tests only the complete DF package against its shared hybrid
baseline; component attribution remains a Jobe factorial claim.

Prime data staging occurs only after selecting an 8xH100 provider and location,
but before provisioning the H100 node. Create a provider-local persistent disk
sized for the compiled stream and run artifacts, attach it to an inexpensive
compatible staging instance, install the exact `data-build` stack, and run `df
tokenize` directly from the pinned Hugging Face dataset and tokenizer into the
mounted data path. The resulting metadata and byte-checksum manifest must match
the canonical Jobe prefix before the disk is detached and attached to the H100
node. The registered path requires neither live training reads from Hugging
Face nor a separately hosted copy of the compiled store.

The schedule is expressible by the current single-process trainer, but Prime
distributed execution is not runnable. Its gate must add exact row sharding,
DDP accumulation and synchronized global clipping, rank-safe telemetry and
snapshots, persistent artifact staging, cross-rank optimizer parity, restart
tests, persistent-disk read qualification, and H100 memory/throughput
qualification. The qualified configuration and projected rental cost are
reviewed before provisioning.

## Larger reference organism

The larger reference model is the exact hard-DF architecture in
[architecture.md](architecture.md): 1,335,420,192 total parameters and
1,102,046,496 active non-embedding parameters. Its budget is 400 predicted
tokens per active non-embedding parameter before batch alignment.

| Quantity | Specified value |
|---|---:|
| Sequence length | 8,192 predictions |
| Global batch | 40 sequences = 327,680 predictions |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 5 |
| Unrounded target | 440,818,598,400 predicted tokens |
| Optimizer steps | 1,345,272 |
| Exact aligned budget | 440,818,728,960 predicted tokens |
| Warmup | steps 1–26,905 |
| Stable heat | steps 26,906–1,076,218 |
| Cooldown | steps 1,076,219–1,345,272 |
| Feedback boundary | after step 1,008,954 |
| Whole-run pass mixture | expected 75% / 22% / 3% |
| Expected compute | approximately 564.248B pass-tokens |

The aligned schedule exceeds the mathematical 400x target by 130,560 tokens,
less than one optimizer batch, and realizes 400.000118 predicted tokens per
active parameter.

The specified target is replicated DDP over eight H100 80GB GPUs with BF16
autocast, FP32 parameters/optimizer state, and no tensor, pipeline, context, or
parameter sharding. Every block is activation-checkpointed on every pass;
payload and source-bank graphs remain differentiable. The exact implementation
must establish parameter and compute accounting, portable/chunk/recurrent PKDA
parity, cache continuation, gated-GQA parity, MHDB source identities, optimizer
partition and radius invariants, distributed row assignment and clipping,
checkpoint portability, restart behavior, memory, and throughput.

## Larger loop reference

Beyond the screen, one loop run is outlined at larger reference width with the
screen's naive depth: the same three cells, one each for prelude, core, and
coda, with the layer geometry from [architecture.md](architecture.md) and a
deeper draw. It is an optional counterpart to the larger reference above, with
the same research and execution gates.

| Field | Value |
|---|---:|
| Residual width / SwiGLU intermediate | 1,536 / 6,656 |
| PKDA heads x width, projection width | 20 x 128, 2,560 |
| Global query / KV heads, head width | 16 / 8, 96 |
| Unique layers / cells | 12 / 3 |
| Context | 8,192 |
| `r_mean` / `r_max` | 16 / 32 |
| Compute depth per pass | `4 + 4r + 4` layers, mean 70.2, cap 136 |

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 233,373,696 |
| 12 SwiGLU channel mixers | 368,050,176 |
| 9 PKDA mixers | 152,148,816 |
| 3 gated global GQA mixers, including Q/K norms | 28,312,128 |
| Hard-DF fusion | 4,718,592 |
| Trunk, entry, and payload norms | 43,008 |
| 24 within-column routers and one payload router | 115,200 |
| **Total** | **786,761,616** |
| **Active non-embedding** | **553,387,920** |

Under the cap the draw gives `E[r] = 15.56`, median 14, `P(r = 1) = 0.1%`, `P(r
= 32) = 5.8%`, and 17.56 expected cells per pass. Metrics use fixed `r = 16`;
the sweep runs to 32.

The run inherits the larger reference recipe: 400 predicted tokens per active
non-embedding parameter aligned to the larger reference batch of 40 sequences
by 8,192 predictions, the same optimizer, pass mixture, and schedule shape, and
the same 8xH100 DDP target with every block checkpointed on every pass.

| Quantity | Value |
|---|---:|
| Global batch | 327,680 predictions |
| Optimizer steps | 675,523 |
| Exact aligned budget | 221,355,376,640 predicted tokens (400.000377 per active parameter) |
| Warmup / stable heat / cooldown | steps 1–13,510 / 13,511–540,418 / 540,419–675,523 |
| Feedback boundary | after step 506,642 |
| Expected pass-tokens | approximately 283.3B |
| Expected cell-tokens | approximately 4.97T |

The larger reference spends about 3.39T cell-tokens, so this run uses about
1.5x as many cell-tokens with 50% of its active parameters. This is an
accounting estimate, not measured device time. Per sequence at 8,192 context
one cell's mixer cache is 27.9 MiB: Standard decoding at `r_max` holds 34
cells, 949 MiB, and Soft or Fused decoding holds 3 cells, 83.7 MiB.

Entry requires an admissible `df-loop - df` screen result, the Prime gates
below, the [loop adoption gates](depth-architecture.md#adoption), and explicit
spend confirmation.

## Entry gates

Before either distributed reference is considered:

1. Complete the admissible two-seed Jobe factorial and state the mechanistic
   question, observed limitation, and smallest comparison that resolves it.
2. Reproduce the relevant checkpoint intervention and recurrent dynamics;
   specify what evidence the extra training or size must add. Include null
   and negative outcomes in the decision rule.
3. Materialize pinned data directly from Hugging Face on provider-local
   persistent storage and establish byte-checksum parity with Jobe's prefix.
4. Qualify distributed row assignment, accumulation, global clipping,
   optimizer parity, checkpoint portability, durable artifacts, restart,
   memory, and throughput on the actual hardware.
5. Review the projected total cost and obtain a9's explicit spend confirmation.

The larger reference additionally requires the fresh Prime pair, stable
long-horizon feedback over at least 30 fused self-compositions, and the exact
architecture gate in [architecture.md](architecture.md). Any larger loop
requires the small `df-loop` comparison and all loop-specific adoption gates.
An absent capability advantage need not invalidate a useful organism; absent
causal use of the mechanism requires explanation before expanding it.

A fresh Prime `{base, df}` pair tests the whole package. It does not replace
the Jobe factorial for component attribution, and none of these comparisons
establishes that mechanisms or monitors transfer to frontier reasoning models.
