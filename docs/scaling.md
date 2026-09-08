# Bigger recipes

The small Jobe model is the organism. These are worked-out recipes for a
longer-trained or larger specimen, kept so the arithmetic is ready if a
question needs them. The distributed paths are not implemented. Prime time
is real money; a9 decides when to spend it.

"Screen", "25x", and "400x" name recipes by predicted tokens per active
non-embedding parameter. Every feedback pass adds to compute.

## Longer training at the same size

A fresh `{a, arf}` pair on one 8xH100-80GB Prime node, both conditions seed 1,
data seed 0, global row zero, 171,909 steps, 56,331,141,120 predicted tokens
each, nothing loaded from Jobe.

| Condition | Predicted tokens | Active-token ratio |
|---|---:|---:|
| `a` | 56,331,141,120 | 403.55 |
| `arf` | 56,331,141,120 | 400.00 |

| Phase | Steps | Passes, `arf` |
|---|---:|---|
| Warmup | 1–3,438 | one |
| Stable heat | 3,439–137,527 | one through 128,932, then two or three |
| Cooldown | 137,528–171,909 | two or three |

The pair is 112.66B predicted tokens and about 128.44B expected pass-tokens.
It compares the complete DF package against its shared hybrid baseline.

Data staging on Prime: after choosing a provider and location and before
provisioning the H100 node, create a provider-local persistent disk sized for
the compiled stream and run artifacts, attach it to a cheap compatible
staging instance, install the `data-build` extra, and run `delta tokenize` from
the pinned Hugging Face dataset and tokenizer into the mounted path. The
resulting `meta.json` and byte checksums match Jobe's prefix; then detach the
disk and attach it to the H100 node. No live Hugging Face reads during
training and no re-hosted copy of the store.

The single-process trainer expresses this schedule already. Distributed
execution needs exact row sharding, DDP accumulation with synchronized global
clipping, rank-safe telemetry and snapshots, persistent artifact staging,
cross-rank optimizer parity, restart tests, persistent-disk read checks, and
H100 memory and throughput measurement. Hopper also needs its own `delta probe`
run and FLA kernel constants swept; see
[runtime-qualification.md](runtime-qualification.md).

## Larger geometry

The larger geometry is `arf` as specified in
[architecture.md](architecture.md) at six cells and width 1,536:
1,335,420,192 parameters, 1,102,046,496 active non-embedding. Its budget is
400 predicted tokens per active non-embedding parameter.

| Quantity | Value |
|---|---:|
| Sequence length | 8,192 predictions |
| Global batch | 40 sequences = 327,680 predictions |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 5 |
| Unrounded target | 440,818,598,400 predicted tokens |
| Optimizer steps | 1,345,272 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup | steps 1–26,905 |
| Stable heat | steps 26,906–1,076,218 |
| Cooldown | steps 1,076,219–1,345,272 |
| Feedback boundary | after step 1,008,954 |
| Whole-run pass mixture | expected 75% / 22% / 3% |
| Expected compute | about 564.25B pass-tokens |

The aligned schedule overshoots the 400x target by 130,560 tokens, less than
one batch, at 400.000118 predicted tokens per active parameter.

The execution target is replicated DDP over eight H100 80GB GPUs with BF16
autocast, FP32 parameters and optimizer state, no tensor, pipeline, context,
or parameter sharding, every block activation-checkpointed on every pass, and
payload and source-bank graphs kept differentiable. Building it means
reproducing the parameter and compute accounting, PKDA parity across the
portable, chunk, and recurrent paths, cache continuation, gated-GQA parity,
MHDB source identities, optimizer partition and radius invariants,
distributed row assignment and clipping, checkpoint portability, restart
behavior, memory, and throughput at that geometry.

## Larger loop

One `arfl` run at the larger width with the screen's depth: three cells,
one each for prelude, core, and coda, and a deeper draw.

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
| FBT fusion, `f` | 4,718,592 |
| Trunk, entry, and payload norms | 43,008 |
| 24 within-column routers and one payload router | 115,200 |
| **Total** | **786,761,616** |
| **Active non-embedding** | **553,387,920** |

Under the cap the draw gives `E[r] = 15.56`, median 14, `P(r = 1) = 0.1%`,
`P(r = 32) = 5.8%`, and 17.56 expected cells per pass. Metrics use fixed
`r = 16`; the sweep runs to 32.

| Quantity | Value |
|---|---:|
| Global batch | 327,680 predictions |
| Optimizer steps | 675,523 |
| Aligned budget | 221,355,376,640 predicted tokens (400.000377 per active parameter) |
| Warmup / stable heat / cooldown | steps 1–13,510 / 13,511–540,418 / 540,419–675,523 |
| Feedback boundary | after step 506,642 |
| Expected pass-tokens | about 283.3B |
| Expected cell-tokens | about 4.97T |

The larger geometry spends about 3.39T cell-tokens, so this run uses about
1.5x as many cell-tokens with half the active parameters; that is accounting,
not measured device time. Per sequence at 8,192 context one cell's mixer
cache is 27.9 MiB: Standard decoding at `r_max` holds 34 cells, 949 MiB, and
Soft or Fused decoding holds 3 cells, 83.7 MiB.

## When one of these would be worth it

When a question about the organism cannot be answered at the small size: a
behavior that never appears at 25x, a channel that stays unused, or a
mechanism whose dependence on training duration is the question. Before
renting, write down what the extra training or size should show, including
what a null would mean, and have the distributed path and the target hardware
qualified.
