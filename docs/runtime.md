# GH200 and H100 node runtime estimates

Planning estimates for the canonical [geometry](scaling.md), using a global
batch of 128 rows × 4,096 predictions = 524,288 tokens per update. Tokens per
parameter use training-active non-embedding counts, including MTP. Both
conditions receive the same token budget.

The default schedule uses one column for the first 75% of updates. In the
remaining 25%, the recurrence roll averages 4.48 columns for `fl` and 2.12
for `f`; whole-run averages are **1.87** and **1.28** respectively. MTP and
the joint vocabulary loss run once per executed column.

Tables include a **10% allowance for evaluation and checkpointing**. Add
roughly 0.5–2 hours for cold compilation/setup, potentially longer for
extension. Data preparation, queueing, failures, and long offline downstream
evaluation are excluded. Durations use hours (h) and days (d).

The node is **8×80 GB H100 SXM with NVSwitch**, running the existing
optimizer-sharded `--ranks 8` path with the same global batch. Assumed speedup
over one GH200 is 6.5× for screen/bridge/flagship and 7× for extension, where
optimizer sharding also relieves memory pressure. These speedups are modeled;
no H100 node throughput has been measured.

## Single GH200

| Scale | Condition |25 tok/param|100 tok/param|400 tok/param|
|---|---|---:|---:|---:|
|screen|fl|1.2 d|4.9 d|19.6 d|
|screen|f|19.8 h|3.3 d|13.2 d|
|bridge|fl|4.4 d|17.6 d|70.5 d|
|bridge|f|3.0 d|12.0 d|48.1 d|
|flagship|fl|12.4 d|49.5 d|197.8 d|
|flagship|f|8.4 d|33.5 d|134.1 d|
|extension|fl|56.4 d|225.4 d|901.7 d|
|extension|f|37.9 d|151.6 d|606.6 d|

## 8×H100 node

| Scale | Condition |25 tok/param|100 tok/param|400 tok/param|
|---|---|---:|---:|---:|
|screen|fl|4.5 h|18.0 h|3.0 d|
|screen|f|3.1 h|12.2 h|2.0 d|
|bridge|fl|16.3 h|2.7 d|10.8 d|
|bridge|f|11.1 h|1.8 d|7.4 d|
|flagship|fl|1.9 d|7.6 d|30.4 d|
|flagship|f|1.3 d|5.2 d|20.6 d|
|extension|fl|8.1 d|32.2 d|128.8 d|
|extension|f|5.4 d|21.7 d|86.7 d|

## Method and uncertainty

```text
steps = ceil(tokens_per_param × active_nonembedding_parameters / 524288)
GH200 seconds = steps × modeled_mean_update_seconds × 1.10
H100 node seconds = GH200 seconds / assumed_node_speedup
```

| Scale | Active non-embedding parameters | Mean fl update, seconds | Mean f update, seconds |
|---|---:|---:|---:|
| Screen | 191,702,304 | 10.5 | 7.1 |
| Bridge | 426,644,080 | 17.0 | 11.6 |
| Flagship | 754,314,176 | 27.0 | 18.3 |
| Extension | 1,687,839,328 | 55.0 | 37.0 |

These modeled update times precede the 10% periodic overhead allowance.
The screen anchor is a measured four-column graph: 707.97 ms at four replay
rows, 22.655 seconds for 128 rows, and four actual optimizer updates taking
22.74–22.78 seconds. See [Hopper execution](hopper.md#full-model-screen) for
current measurements and retained raw evidence.

Larger scales extrapolate component costs with allowances for replay width
and checkpointing. They are not full-model measurements. The first 75% of
training benefits from wider single-column replay; extension includes a
checkpointing allowance. The budget and step rounding follow
`reference_active` and `parse_run_args` in
[train.py](../delta_feedback_experiment/train.py).

Allow roughly ±20% for screen, ±25% for bridge, and ±30% for flagship on
GH200. Extension is roughly 0.6–1.6× the point estimate, conditional on fit.
H100 figures have roughly 0.7–1.5× uncertainty, wider for extension. These
are judgment ranges, not statistical confidence intervals.

## Extension memory boundary

Extension's GH200 estimate is conditional on initialization, calibration,
and capture fitting. Estimated persistent state is **70.12 GiB**, and the
initialization floor is **89.33 GiB before temporary allocations**, against
**94.50 GiB** CUDA-visible memory.

The current eight-rank path reduces those estimates to **44.92 GiB
persistent** and **64.13 GiB at initialization** per GPU. Full capture remains
untested. Fully replicated DDP would exceed an 80 GB H100 during the current
extension initialization path; the node table assumes the repository's
optimizer ownership sharding.

Hardware definition: [NVIDIA HGX reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html).
