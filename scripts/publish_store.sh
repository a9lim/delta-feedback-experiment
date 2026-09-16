#!/bin/bash
# Build the dclm-100b token stream on a machine with fast cores and a fast
# uplink (the rental), check it against a reference manifest, and publish it
# as a public dataset so every later box pulls it at datacenter speed.
#
#   scripts/publish_store.sh --disk /data [--target 100e9] [--workers 16]
#       [--repo a9lim/dclm-100b-neox] [--reference /data/delta/dclm-100b/dclm-100b.sha256]
#       [--skip-upload]
#
# Writes DISK/build/dclm-100b (scratch beside it, removed on success), runs
# delta verify, compares every shard but the reference's last (a partial
# shard fills in the longer build) plus the held-out slice against the
# reference manifest, writes the dataset card, then uploads with hf. Needs
# about 2.5x the final store free on DISK while downloads, parts, and shards
# coexist: a 100e9 target is a 400 GB store. Runs from the experiment
# directory; the tokenizer needs the data-build extra and hf auth.

set -euo pipefail

DISK="" TARGET=100e9 WORKERS=16 READERS=8 REPO=a9lim/dclm-100b-neox REFERENCE="" UPLOAD=1
TOKENS_PER_DOC=1150   # jobe's 15B build measured 1,308 per document; 1,150 keeps the selection inside the 89.27M-document universe with margin

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
while (( $# )); do
  case $1 in
    --disk) DISK=$2; shift ;;
    --target) TARGET=$2; shift ;;
    --workers) WORKERS=$2; shift ;;
    --readers) READERS=$2; shift ;;
    --repo) REPO=$2; shift ;;
    --reference) REFERENCE=$2; shift ;;
    --skip-upload) UPLOAD=0 ;;
    -h|--help) usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
  shift
done
[[ -n $DISK ]] || usage
command -v delta >/dev/null || { echo "delta is not on PATH" >&2; exit 1; }
[[ -f pyproject.toml && -d delta_feedback_experiment ]] || { echo "run from the experiment directory" >&2; exit 1; }
if [[ -z $REFERENCE && -f $DISK/delta/dclm-100b/dclm-100b.sha256 ]]; then
  REFERENCE="$DISK/delta/dclm-100b/dclm-100b.sha256"
fi
OUT="$DISK/build/dclm-100b"
mkdir -p "$DISK/build"
say() { printf '\n== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }

# A worker is one parquet file; the tokenizer's own threads (RAYON) are the
# other axis. The builder's default is cpus / (2 x workers), which left a
# 64-core box a third busy; fill the cores unless the caller pins it.
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(( $(nproc) / WORKERS > 0 ? $(nproc) / WORKERS : 1 ))}
say "build $OUT: target $TARGET tokens, $WORKERS workers x $RAYON_NUM_THREADS threads ($(nproc) cpus, $(df -BG --output=avail "$DISK" | tail -1 | tr -dc 0-9) GB free)"
( time delta tokenize --source dclm-100b --out "$OUT" --target "$TARGET" \
    --tokens-per-doc "$TOKENS_PER_DOC" --workers "$WORKERS" --readers "$READERS" \
    --scratch "$DISK/build/scratch" ) 2>&1 | tail -40

say "verify"
delta verify "$OUT"

say "manifest"
( cd "$OUT" && sha256sum meta.json source.json val.bin val.docs.npy train.docs.npy train.*.bin > dclm-100b.sha256 )
if [[ -n $REFERENCE && -f $REFERENCE ]]; then
  last=$(grep -o 'train\.[0-9]*\.bin' "$REFERENCE" | sort | tail -1)
  ( cd "$OUT" && grep -E ' (val\.bin|val\.docs\.npy|train\.[0-9]+\.bin)$' "$REFERENCE" | grep -v " $last\$" \
      | sha256sum -c --quiet ) && echo "prefix matches $REFERENCE (all but $last, meta.json, train.docs.npy)"
else
  echo "no reference manifest: prefix identity unchecked"
fi

say "dataset card"
python - "$OUT" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

out, repo = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((out / "meta.json").read_text())
shards = sorted(out.glob("train.*.bin"))
gib = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 2**30
pins = ", ".join(f"{k} {v}" for k, v in meta["packages"].items())
card = f"""---
license: odc-by
pretty_name: dclm-100b as one GPT-NeoX token stream
size_categories:
- 100B<n<1T
source_datasets:
- {meta['dataset']}
---

# dclm-100b as one GPT-NeoX token stream

`{meta['dataset']}` (revision `{meta['revision'][:12]}`, published order, no
shuffle) tokenized with `{meta['tokenizer']}` (revision
`{meta['tokenizer_revision'][:12]}`, {meta['tokenizer_vocab_size']:,} tokens;
`{meta['chatml_start_id']}`/`{meta['chatml_end_id']}` are the ChatML markers,
`{meta['eos_id']}` ends every document) into one contiguous uint32 stream:
{meta['train_tokens']:,} training tokens in {len(shards)} shards of 2^28
tokens, after a held-out slice of {meta['val_tokens']:,} tokens taken from the
stream's first documents. {gib:,.0f} GiB in all. The upstream data is
ODC-By 1.0 (attribution: the DCLM authors and Hugging Face's shuffled
100BT sample); this derivative carries the same license.

Layout (the delta-feedback-experiment `document-stream-v1` store):

| File | Contents |
|---|---|
| `meta.json` | source, tokenizer, build packages ({pins}), counts |
| `source.json` | the document universe: every parquet file's row groups |
| `val.bin`, `val.docs.npy` | the held-out slice and one (start, source) record per document |
| `train.NNNN.bin` | uint32 shards, one contiguous stream, 2^28 tokens each |
| `train.docs.npy` | one (start, source) record per training document |
| `dclm-100b.sha256` | checksums of every file above |

Row r of the training stream is tokens [r·(L+1), (r+1)·(L+1)) for a model
of sequence length L, so any prefix of the shards is a training set by
itself: `hf download {repo} --type dataset --include 'train.00[0-1]*.bin'
--include 'val.*' --include '*.json' --include '*.npy'` gives the first 20.
Every store built by the same tool from the same pins is a prefix of this
stream, so shorter builds elsewhere are byte-identical prefixes.
"""
(out / "README.md").write_text(card)
print(card.splitlines()[7])
PY

if (( UPLOAD )); then
  say "upload -> $REPO (public dataset)"
  hf auth whoami >/dev/null || { echo "hf auth login first" >&2; exit 1; }
  hf repos create "$REPO" --type dataset --public --exist-ok
  ( time hf upload "$REPO" "$OUT" . --type dataset --exclude 'scratch/*' --exclude 'parts/*' ) 2>&1 | tail -5
  hf datasets list "$REPO" --tree --human-readable 2>/dev/null | head -12 || true
else
  say "upload skipped; the store is at $OUT"
fi
say "done"
