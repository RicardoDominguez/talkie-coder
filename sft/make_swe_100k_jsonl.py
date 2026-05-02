"""Subset final-400k.jsonl → 100K shuffled records → talkie-swe-100k.jsonl.

Streams source line-by-line, reservoir-samples 100K under seed=0, writes to
output. ~31 GB input → ~8 GB output.
"""
import json
import random
import sys

SRC = "/fast/rolmedo/swesmith/datasets/final-400k.jsonl"
DST = "/tmp/talkie-swe-100k.jsonl"
N = 100_000
SEED = 0


def main():
    rng = random.Random(SEED)
    reservoir = []
    n_seen = 0
    with open(SRC, "r") as f:
        for line in f:
            n_seen += 1
            if len(reservoir) < N:
                reservoir.append(line)
            else:
                j = rng.randint(0, n_seen - 1)
                if j < N:
                    reservoir[j] = line
            if n_seen % 50_000 == 0:
                print(f"  read {n_seen:,} lines", flush=True)
    print(f"read {n_seen:,} total, sampled {len(reservoir):,}", flush=True)
    rng.shuffle(reservoir)
    with open(DST, "w") as f:
        for line in reservoir:
            f.write(line)
    print(f"wrote {DST}", flush=True)


if __name__ == "__main__":
    main()
