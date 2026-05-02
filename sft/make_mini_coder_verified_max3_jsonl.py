"""Filter ricdomolm/mini-coder-trajs-400k -> JSONL.

Keeps only `verified=True` rows and caps to 3 samples per `instance_id`.
Source dataset already saved to disk at /fast/rolmedo/mini-coder-trajs/.
Output schema matches the JSONL consumed by tokenize_messages.py
(`instance_id`, `messages`); also keeps `patch`, `model`, `verified` for
downstream filtering/inspection.
"""
import json
import os
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/fast/rolmedo/.cache/huggingface")

import datasets

SRC = "/tmp/mini-coder-trajs"
DST = "/tmp/mini-coder-trajs-verified-max3.jsonl"
MAX_PER_INSTANCE = 3


def main():
    ds = datasets.load_from_disk(SRC)
    tr = ds["train"] if hasattr(ds, "keys") else ds
    print(f"loaded {len(tr):,} rows; cols={tr.column_names}", flush=True)

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    counts = defaultdict(int)
    n_in = n_verified = n_out = 0
    with open(DST, "w") as f:
        for row in tr:
            n_in += 1
            if not row["verified"]:
                continue
            n_verified += 1
            iid = row["instance_id"]
            if counts[iid] >= MAX_PER_INSTANCE:
                continue
            counts[iid] += 1
            out = {
                "instance_id": iid,
                "messages": row["messages"],
                "patch": row["patch"],
                "model": row["model"],
                "verified": row["verified"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 10000 == 0:
                print(f"  wrote {n_out:,}", flush=True)
    n_instances = len(counts)
    print(
        f"done -> {DST}\n"
        f"  scanned     : {n_in:,}\n"
        f"  verified    : {n_verified:,}\n"
        f"  written     : {n_out:,}\n"
        f"  instances   : {n_instances:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
