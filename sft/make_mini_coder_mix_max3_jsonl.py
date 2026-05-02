"""Build mixed mini-coder JSONL: verified (max 3/instance) + unverified
trajectories from instance_ids NOT covered by the verified subsample,
length-targeted at +40% raw bytes for ~1 epoch fit at 2016 steps.

Inputs:
  /tmp/mini-coder-trajs (HF on-disk source)
  /fast/rolmedo/swesmith/datasets/mini-coder-trajs-verified-max3.jsonl

Output:
  /tmp/mini-coder-trajs-mix-max3.jsonl  (then dd to /fast manually)

Schema per line: instance_id, messages, patch, model, verified.
"""
import json
import os
import random
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/tmp")

import datasets

VERIFIED_JSONL = "/fast/rolmedo/swesmith/datasets/mini-coder-trajs-verified-max3.jsonl"
SRC = "/tmp/mini-coder-trajs"
DST = "/tmp/mini-coder-trajs-mix-max3.jsonl"
EXTRA_FRAC = 0.4
MAX_PER_INSTANCE = 3
MAX_LINE_BYTES = 50 * 1024 * 1024  # defensive: skip pathologically large rows


def main():
    covered = set()
    verified_bytes = 0
    verified_rows = 0
    with open(VERIFIED_JSONL) as f:
        for line in f:
            verified_bytes += len(line)
            verified_rows += 1
            covered.add(json.loads(line)["instance_id"])
    print(
        f"verified subsample: {verified_rows:,} rows, "
        f"{verified_bytes:,} bytes, {len(covered):,} instance_ids",
        flush=True,
    )

    target_extra = int(verified_bytes * EXTRA_FRAC)
    print(f"target extra bytes (+{int(EXTRA_FRAC*100)}%): {target_extra:,}", flush=True)

    print("loading source ...", flush=True)
    ds = datasets.load_from_disk(SRC)
    ds = ds["train"] if hasattr(ds, "keys") else ds
    print(f"  {len(ds):,} rows", flush=True)

    print("filtering: verified=False AND instance_id NOT in covered ...", flush=True)
    unv = ds.filter(
        lambda x: (not x["verified"]) and (x["instance_id"] not in covered),
        num_proc=4,
    )
    print(
        f"  {len(unv):,} candidate rows over "
        f"{len(set(unv['instance_id'])):,} non-covered instance_ids",
        flush=True,
    )

    unv = unv.shuffle(seed=42)

    seen = defaultdict(int)
    extra_bytes = 0
    n_extra = 0
    n_capped = 0
    n_oversize = 0
    with open(VERIFIED_JSONL) as fin, open(DST, "w") as fout:
        for line in fin:
            fout.write(line)
        for row in unv:
            if extra_bytes >= target_extra:
                break
            iid = row["instance_id"]
            if seen[iid] >= MAX_PER_INSTANCE:
                n_capped += 1
                continue
            out = {
                "instance_id": iid,
                "messages": row["messages"],
                "patch": row["patch"],
                "model": row["model"],
                "verified": row["verified"],
            }
            line = json.dumps(out, ensure_ascii=False) + "\n"
            if len(line) > MAX_LINE_BYTES:
                n_oversize += 1
                continue
            seen[iid] += 1
            fout.write(line)
            extra_bytes += len(line)
            n_extra += 1
            if n_extra % 5000 == 0:
                print(
                    f"  extra rows={n_extra:,} bytes={extra_bytes:,} "
                    f"({extra_bytes/target_extra*100:.1f}% of target)",
                    flush=True,
                )

    final_bytes = verified_bytes + extra_bytes
    print(
        f"\ndone -> {DST}\n"
        f"  verified rows : {verified_rows:,} ({verified_bytes:,} B)\n"
        f"  extra rows    : {n_extra:,} ({extra_bytes:,} B)\n"
        f"  capped (>3)   : {n_capped:,}\n"
        f"  oversize drop : {n_oversize:,}\n"
        f"  final         : {verified_rows + n_extra:,} rows, {final_bytes:,} B "
        f"(+{extra_bytes/verified_bytes*100:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
