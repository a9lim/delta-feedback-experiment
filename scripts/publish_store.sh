#!/bin/bash
# Build every document in dclm-100b, hold out the first 100M tokens at a
# document boundary, verify, and publish the store as a public HF dataset.
#
#   scripts/publish_store.sh --disk /data [--target 100e9] [--val 1e8]
#       [--out /data/build/dclm-100b-val100m] [--workers 16] [--readers 8]
#       [--repo a9lim/dclm-100b-neox] [--reference /path/dclm-100b.sha256]
#       [--skip-upload]
#
# Defaults to the entire source, writing DISK/build/dclm-100b-val100m;
# --target requests a finite stored-token prefix instead. Existing stores
# resume/extend only with matching source, tokenizer, ordering and holdout.
# Prior builds and DISK/delta/dclm-100b remain untouched. Allow about 2.5x
# the final store size while downloads, parts and shards coexist. Run from
# the experiment directory with the data-build extra and hf authentication.

set -euo pipefail

DISK="" TARGET="" VAL=1e8 OUT="" WORKERS=16 READERS=8 REPO=a9lim/dclm-100b-neox REFERENCE="" UPLOAD=1
TOKENS_PER_DOC=1150   # Conservative selection estimate for finite --target builds only.

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
while (( $# )); do
  case $1 in
    --disk) DISK=$2; shift ;;
    --target) TARGET=$2; shift ;;
    --val) VAL=$2; shift ;;
    --out) OUT=$2; shift ;;
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
OUT=${OUT:-"$DISK/build/dclm-100b-val100m"}
mkdir -p "$DISK/build"
say() { printf '\n== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }

# A worker is one parquet file; the tokenizer's own threads (RAYON) are the
# other axis. The builder's default is cpus / (2 x workers), which left a
# 64-core box a third busy; fill the cores unless the caller pins it.
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(( $(nproc) / WORKERS > 0 ? $(nproc) / WORKERS : 1 ))}
selection=(--all)
[[ -z $TARGET ]] || selection=(--target "$TARGET" --tokens-per-doc "$TOKENS_PER_DOC")
continuation=()
[[ ! -f $OUT/meta.json ]] || continuation=(--continue)
say "build $OUT: ${TARGET:-entire source}, holdout target $VAL, $WORKERS workers x $RAYON_NUM_THREADS threads ($(nproc) cpus, $(df -BG --output=avail "$DISK" | tail -1 | tr -dc 0-9) GB free)"
time delta tokenize --source dclm-100b --out "$OUT" "${selection[@]}" --val "$VAL" \
    "${continuation[@]}" --workers "$WORKERS" --readers "$READERS" \
    --scratch "$OUT.scratch"

say "verify"
delta verify "$OUT"

say "manifest"
( cd "$OUT" && sha256sum meta.json source.json val.bin val.docs.npy train.docs.npy train.*.bin > dclm-100b.sha256 )
if [[ -n $REFERENCE ]]; then
  python - "$OUT" "$REFERENCE" <<'PYREF'
import json
import subprocess
import sys
from pathlib import Path

out, reference = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
lines = reference.read_text().splitlines()
reference_meta = reference.parent / "meta.json"
if not reference_meta.is_file():
    print(f"reference metadata missing at {reference_meta}: prefix identity unchecked")
else:
    current = json.loads((out / "meta.json").read_text())
    previous = json.loads(reference_meta.read_text())
    identity = (
        "format", "dataset", "revision", "source_sha256", "tokenizer_id",
        "tokenizer_revision", "packages", "shuffle", "shuffle_seed", "val_target",
    )
    mismatches = [key for key in identity if previous.get(key) != current.get(key)]
    if mismatches:
        print(f"reference differs in {', '.join(mismatches)}; split-prefix checks skipped")
    else:
        entries = [line.split(maxsplit=1) for line in lines if line.strip()]
        shards = sorted(name for _, name in entries if name.startswith("train.") and name.endswith(".bin"))
        last = shards[-1] if shards else None
        selected = [
            f"{digest}  {name}"
            for digest, name in entries
            if name in {"val.bin", "val.docs.npy"} or name in shards[:-1]
        ]
        if not selected:
            raise SystemExit("reference has no validation files or complete prefix shards")
        subprocess.run(
            ["sha256sum", "-c", "--quiet"], cwd=out,
            input="\n".join(selected) + "\n", text=True, check=True,
        )
        print(f"prefix matches {reference} (excluding final shard {last}, metadata and training sidecar)")
PYREF
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
stored_docs = meta["train_docs"] + meta["val_docs"]
full_source = meta["target_tokens"] is None
if full_source:
    assert meta["selected_docs"] == meta["universe_docs"] and meta["unused_selected"] == 0
    coverage = f"All {meta['universe_docs']:,} source document positions were processed."
else:
    coverage = f"This is a finite prefix selected from {meta['universe_docs']:,} source document positions."
coverage += f" The store contains {stored_docs:,} nonempty documents; empty source texts emit no tokens."
split_note = (
    f"The {meta['val_target']:,}-token holdout target changes the training offset "
    "relative to stores built with the former 30M-token target. Training shards "
    "from those stores are not byte-identical prefixes of this split."
    if meta["val_target"] != 30_000_000 else ""
)

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
tokens (the last shard may be shorter), after a held-out slice of
{meta['val_tokens']:,} tokens in {meta['val_docs']:,} documents. The holdout
requests {meta['val_target']:,} tokens from the stream's first documents and
ends before the document that would cross that limit. Training begins with
that next whole document; no document is split between validation and training.
The total stored stream has {meta['train_tokens'] + meta['val_tokens']:,} tokens
and occupies {gib:,.0f} GiB. {coverage}

{split_note}

The upstream data is
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
Shorter builds using the same source and tokenizer pins, build packages,
ordering, seed, and held-out target have byte-identical training prefixes.
"""
(out / "README.md").write_text(card)
print(f"wrote {out / 'README.md'} ({stored_docs:,} documents)")
PY

if (( UPLOAD )); then
  say "upload -> $REPO (public dataset)"
  hf auth whoami >/dev/null || { echo "hf auth login first" >&2; exit 1; }
  hf repos create "$REPO" --type dataset --public --exist-ok
  # Xet's default upload concurrency can overwhelm the endpoint and hit 429s.
  # Keep its adaptive controller, with a lower ceiling and fewer ingested files.
  export HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY=${HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY:-8}
  export HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=${HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION:-2}
  export HF_HUB_VERBOSITY=${HF_HUB_VERBOSITY:-info}
  python -u - "$OUT" "$REPO" <<'PYUPLOAD'
import hashlib
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, RepoFile

out, repo = Path(sys.argv[1]), sys.argv[2]
api = HfApi()
hashes = dict(
    (name, digest)
    for digest, name in (
        line.split(maxsplit=1) for line in (out / "dclm-100b.sha256").read_text().splitlines()
    )
)
for name in ("README.md", "dclm-100b.sha256"):
    hashes[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
git_hashes = {}


def remote_state():
    revision = api.repo_info(repo, repo_type="dataset").sha
    entries = {
        item.path: item
        for item in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
        if isinstance(item, RepoFile)
    }
    return revision, entries


def matches(name, entries):
    item = entries.get(name)
    path = out / name
    if item is None or item.size != path.stat().st_size:
        return False
    if item.lfs is not None:
        return item.lfs.sha256 == hashes[name]
    if name not in git_hashes:
        content = path.read_bytes()
        git_hashes[name] = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    return item.blob_id == git_hashes[name]


revision, entries = remote_state()


def upload(names, label):
    pending = [name for name in names if not matches(name, entries)]
    if not pending:
        print(f"{label}: all {len(names)} files already match committed Hub hashes")
        return
    command = ["hf", "upload", repo, str(out), ".", "--type", "dataset"]
    for name in pending:
        command.extend(("--include", name))
    for attempt in range(1, 4):
        print(f"{label}: {len(pending)} files, attempt {attempt}/3", flush=True)
        status = subprocess.run(command).returncode
        if status == 0:
            return
        if attempt == 3:
            print("upload failed; rerun this command to reuse the store and skip matching committed files", file=sys.stderr)
            raise SystemExit(status)
        delay = attempt * 30
        print(f"upload failed (exit {status}); retrying in {delay}s", flush=True)
        time.sleep(delay)


shards = sorted(name for name in hashes if name.startswith("train.") and name.endswith(".bin"))
for offset in range(0, len(shards), 8):
    upload(shards[offset:offset + 8], f"training shards {offset + 1}-{min(offset + 8, len(shards))}/{len(shards)}")
upload(["val.bin", "val.docs.npy", "train.docs.npy", "source.json"], "validation and document index")
# Publish the descriptor and checksum manifest only after every payload is committed.
upload(["meta.json", "README.md", "dclm-100b.sha256"], "store metadata and card")
revision, entries = remote_state()
mismatches = [name for name in hashes if not matches(name, entries)]
if mismatches:
    raise SystemExit(f"Hub verification failed: {mismatches}")
print(f"Verified {len(hashes)} published files by size and content hash at {repo}@{revision}")
PYUPLOAD
else
  say "upload skipped; the store is at $OUT"
fi
say "done"
