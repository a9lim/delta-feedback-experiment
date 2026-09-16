#!/bin/bash
# The first session on a rented single GPU, ordered by information per
# GPU-minute: inventory, the CUDA probe, a short run that captures every
# graph, a resume, a cooperative stop, then a summary (memory plan, peak,
# seconds per step by graph) next to the logs. Nothing here is a benchmark;
# the lever measurements in docs/hopper.md come after this passes.
#
#   scripts/first_hour.sh --data-root /data/delta [--tag first-hour] [--steps 12]
#       [--source dclm-100b] [--scale screen] [--condition fl] [--precision fp8]
#       [--seed 1] [--data-seed 0]
#
# Run from the experiment directory. Logs and summary.json land in
# logs/first-hour/<tag>/; snapshots in runs/ like any run's. The run starts
# its recurrence roll at step 0 so the deepest graphs are captured and hit
# within the short window.

set -euo pipefail

DATA_ROOT="" SOURCE=dclm-100b TAG=first-hour STEPS=12
SCALE=screen CONDITION=fl PRECISION=fp8 SEED=1 DATA_SEED=0

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}
while (( $# )); do
  case $1 in
    --data-root) DATA_ROOT=$2; shift ;;
    --source) SOURCE=$2; shift ;;
    --tag) TAG=$2; shift ;;
    --steps) STEPS=$2; shift ;;
    --scale) SCALE=$2; shift ;;
    --condition) CONDITION=$2; shift ;;
    --precision) PRECISION=$2; shift ;;
    --seed) SEED=$2; shift ;;
    --data-seed) DATA_SEED=$2; shift ;;
    -h|--help) usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
  shift
done
[[ -n $DATA_ROOT ]] || usage
command -v delta >/dev/null || { echo "delta is not on PATH" >&2; exit 1; }
[[ -f pyproject.toml && -d delta_feedback_experiment ]] || { echo "run from the experiment directory" >&2; exit 1; }
[[ -f $DATA_ROOT/$SOURCE/meta.json ]] || { echo "no token store at $DATA_ROOT/$SOURCE" >&2; exit 1; }
if ls runs/"$TAG".pt.* >/dev/null 2>&1; then
  echo "runs/$TAG.pt.* exist: pick a fresh --tag, or delta clear $TAG" >&2
  exit 1
fi
(( STEPS >= 6 )) || { echo "--steps must be at least 6" >&2; exit 1; }
HALF=$(( STEPS / 2 ))
OUT="logs/first-hour/$TAG"
mkdir -p "$OUT"
say() { printf '\n== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }
TRAIN=(delta train "$TAG" --condition "$CONDITION" --scale "$SCALE" --seed "$SEED"
       --data-seed "$DATA_SEED" --data-root "$DATA_ROOT" --source "$SOURCE"
       --precision "$PRECISION")

say "inventory -> $OUT/inventory.txt"
{
  date -Is
  printf '%s %s\n' "$(hostname)" "$(uname -mr)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit \
    --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
  python - <<'PY'
import importlib.metadata as m
import torch
free, total = torch.cuda.mem_get_info()
cap = ".".join(map(str, torch.cuda.get_device_capability()))
print(f"{torch.cuda.get_device_name()} sm {cap}, {total / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free before anything")
for dist in ("torch", "triton", "flash-linear-attention", "cut-cross-entropy", "transformers", "numpy"):
    try:
        print(f"{dist} {m.version(dist)}")
    except m.PackageNotFoundError:
        print(f"{dist} absent")
print(f"cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}, nccl {'.'.join(map(str, torch.cuda.nccl.version()))}")
PY
  for d in . .. ../vendor/flash-linear-attention ../vendor/ml-cross-entropy; do
    printf '%s %s\n' "$(git -C "$d" rev-parse --short HEAD)" "$(git -C "$d" rev-parse --show-toplevel)"
  done
} | tee "$OUT/inventory.txt"

say "delta probe"
( time delta probe ) 2>&1 | tee "$OUT/probe.log"

say "train $STEPS steps ($SCALE $CONDITION, $PRECISION, recurrence from step 0, eval and snapshot every $HALF)"
( time "${TRAIN[@]}" --recurrence-start 0 --max-steps "$STEPS" \
    --eval-every "$HALF" --snapshot-every "$HALF" ) 2>&1 | tee "$OUT/train.log"

say "resume from the step-$STEPS snapshot for 3 steps"
( time delta train "$TAG" --resume --max-steps 3 ) 2>&1 | tee "$OUT/resume.log"

say "cooperative stop: SIGINT after two steps of a resumed run"
delta train "$TAG" --resume --max-steps 50 > "$OUT/stop.log" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 900); do
  if (( $(grep -c '^step ' "$OUT/stop.log" 2>/dev/null || true) >= 2 )); then break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 2
done
kill -INT "$pid" 2>/dev/null || true
if wait "$pid"; then echo "stopped run exited 0"; else echo "stopped run exited $?"; fi
grep -E '^(interrupt|checkpoint|yield|done) ' "$OUT/stop.log" || echo "no interrupt or checkpoint record in stop.log"
ls -la runs/"$TAG".pt.* 2>/dev/null || true

say "summary -> $OUT/summary.json"
python - "$OUT" "$TAG" <<'PY'
import json
import statistics
import sys
from pathlib import Path

from transformer_experiments.telemetry import parse_record

out, tag = Path(sys.argv[1]), sys.argv[2]


def records(name):
    path = out / name
    if not path.exists():
        return []
    return [r for r in map(parse_record, path.read_text().splitlines()) if r]


def fields(record, *names):
    return {k: record[k] for k in names if k in record}


train = records("train.log")
summary = {"tag": tag}
for r in train:
    if r["event"] == "memory_plan":
        summary["memory_plan"] = {k: v for k, v in r.items() if k != "event"}
    elif r["event"] in ("run", "execution"):
        summary.setdefault("run", {}).update(fields(
            r, "precision", "static_gib", "activation_budget_gib", "inputs_gib", "cuda_graphs",
            "peak_allocated_gib", "reserved_gib", "free_gib",
        ))
summary["plans"] = [{k: v for k, v in r.items() if k != "event"} for r in train if r["event"] == "plan"]
steps = [r for r in train if r["event"] == "step"]
# Seconds per step = the difference of consecutive elapsed fields, credited
# to the later step's graph; skipped for the first steps (autotune, warm
# caches) and for any step whose gap holds an eval or a snapshot record.
per, prev, clean, n_steps = {}, None, True, 0
for r in train:
    if r["event"] == "step":
        elapsed = float(r["elapsed"])
        n_steps += 1
        if prev is not None and clean and n_steps > 3:
            per.setdefault((int(r["k"]), int(r.get("r", 1))), []).append(elapsed - prev)
        prev, clean = elapsed, True
    elif r["event"] in ("eval", "checkpoint", "route_summary", "expert_summary", "depth_trace"):
        clean = False
summary["step_seconds"] = {
    f"k={k} r={r}": {"n": len(v), "mean": round(statistics.mean(v), 2), "median": round(statistics.median(v), 2)}
    for (k, r), v in sorted(per.items())
}
summary["tok_s_last"] = steps[-1].get("tok_s") if steps else None
summary["evals"] = [fields(r, "step", "val", "val_fused", "val_one", "val_mtp") for r in train if r["event"] == "eval"]
summary["resume_steps"] = [fields(r, "step", "loss") for r in records("resume.log") if r["event"] == "step"]
stop = records("stop.log")
summary["stop"] = {
    "interrupt_at": [r["step"] for r in stop if r["event"] == "interrupt"],
    "checkpoints": [r["path"] for r in stop if r["event"] == "checkpoint"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(f"memory_plan: {summary.get('memory_plan')}")
print(f"run: {summary.get('run')}")
for p in summary["plans"]:
    print(f"plan k={p['k']} r={p['r']}: {p['rows']} rows, saved={p['saved']}, "
          f"recompute {p['checkpoint_blocks']}/{p['eligible_blocks']}, est {p['estimated_gib']} GiB")
for name, s in summary["step_seconds"].items():
    print(f"step {name}: n={s['n']} mean {s['mean']} s median {s['median']} s")
print(f"evals: {summary['evals']}")
print(f"resume: {summary['resume_steps']}")
print(f"stop: {summary['stop']}")
PY
say "done: $OUT"
