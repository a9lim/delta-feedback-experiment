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
It compares the complete `arf` package against its shared hybrid baseline.

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

One `arfl` run at the larger geometry: the six cells of the larger `arf`
above with the middle four tied as the core, so the loop and the flat specimen
have the same 1,335,420,192 parameters, the same 1,102,046,496 active
non-embedding, and the same 400x schedule, and at `r = 1` the loop is the flat
specimen. The draw is the screen's.

| Field | Value |
|---|---:|
| Residual width / SwiGLU intermediate | 1,536 / 6,656 |
| PKDA heads x width, projection width | 20 x 128, 2,560 |
| Global query / KV heads, head width | 16 / 8, 96 |
| Unique layers / cells | 24 / 6: prelude, four core cells, coda |
| Context | 8,192 |
| `r_mean` / `r_max` | 4 / 8 |
| Compute depth per pass | `4 + 16r + 4` layers, mean 70.1, cap 136 |

The parameter table is the larger geometry's; the loop's forty-eight
within-column routers are the flat column's. Under the cap the draw gives
`E[r] = 3.88`, median 4, `P(r = 1) = 10.3%`, `P(r = 8) = 8.0%`, and `2 + 4r`
cells per pass, 17.5 expected. Metrics use fixed `r = 4`; the sweep runs to 8.

| Quantity | Value |
|---|---:|
| Global batch | 327,680 predictions |
| Optimizer steps | 1,345,272 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup / stable heat / cooldown | as the larger geometry |
| Feedback boundary | after step 1,008,954 |
| Expected pass-tokens | about 564.25B |
| Expected cell-tokens | about 9.88T |

The larger geometry spends about 3.39T cell-tokens, so the loop spends about
2.9x the cell-tokens and, with the head executed once per pass, about 2.6x the
arithmetic; that is accounting, not measured device time. The two runs pair as
the screen's do: byte-identical initialization, the same stream, row order,
schedule, and pass-count, prefix, and jitter draws, with the `r` draw the
loop's only extra randomness. Per sequence at 8,192 context one cell's mixer
cache is 27.9 MiB; every decode mode holds `2 + 4r` cells at the request's
`r`, 502 MiB at `r = 4` and 949 MiB at the cap. The shared fused decode cache
belongs to `L`.

## When one of these would be worth it

When a question about the organism cannot be answered at the small size: a
behavior that never appears at 25x, a channel that stays unused, or a
mechanism whose dependence on training duration is the question. Before
renting, write down what the extra training or size should show, including
what a null would mean, and have the distributed path and the target hardware
qualified.
