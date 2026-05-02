"""Sanity-check the tokenized talkie SWE 100K dataset.

For a few random rows: decode the full input_ids, decode only the
completion_mask=1 tokens, and verify the unmasked region matches what
should be learned (assistant turns + the closing chat token), and the
masked region matches system/user turns.
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/tmp")

import datasets
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_dir",
        default="/fast/rolmedo/swesmith/datasets/talkie-1930-swe-100k-64k",
    )
    p.add_argument(
        "--tokenizer_path",
        default="/fast/rolmedo/models/talkie-1930-13b-base/",
    )
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"loading tokenizer from {args.tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"loading dataset from {args.dataset_dir}")
    if os.path.exists(os.path.join(args.dataset_dir, "dataset_info.json")):
        ds = datasets.load_from_disk(args.dataset_dir)
    else:
        subs = sorted(
            os.path.join(args.dataset_dir, x) for x in os.listdir(args.dataset_dir)
        )
        subs = [s for s in subs if os.path.isdir(s)]
        print(f"  concatenating {len(subs)} subdirs")
        parts = [datasets.load_from_disk(s) for s in subs]
        ds = datasets.concatenate_datasets(parts)
    print(f"  total rows: {len(ds):,}")
    print(f"  features: {list(ds.features.keys())}")

    lens = []
    n_unmasked = []
    for row in ds.select(range(min(2000, len(ds)))):
        L = len(row["input_ids"])
        lens.append(L)
        n_unmasked.append(sum(row["completion_mask"]))
    avg_len = sum(lens) / len(lens)
    avg_unmasked = sum(n_unmasked) / len(n_unmasked)
    print(
        f"\nfirst {len(lens)} rows: avg_len={avg_len:.0f} "
        f"(min={min(lens)}, max={max(lens)})  "
        f"avg_unmasked={avg_unmasked:.0f} "
        f"({100*avg_unmasked/max(avg_len,1):.1f}% of tokens)"
    )

    ds = ds.shuffle(seed=args.seed).select(range(args.n))

    end_id = tok.convert_tokens_to_ids("<|end|>")
    asst_id = tok.convert_tokens_to_ids("<|assistant|>")
    print(f"  end_id={end_id}  assistant_id={asst_id}")

    for i, row in enumerate(ds):
        ids = list(row["input_ids"])
        mask = list(row["completion_mask"])
        assert len(ids) == len(mask), f"row {i}: len mismatch {len(ids)} vs {len(mask)}"

        masked_count = sum(1 for m in mask if m == 0)
        unmasked_count = sum(1 for m in mask if m == 1)
        print(
            f"\n=== example {i+1}/{args.n}  "
            f"len={len(ids)}  masked={masked_count}  unmasked={unmasked_count} ==="
        )

        full_text = tok.decode(ids, skip_special_tokens=False)
        target_ids = [t for t, m in zip(ids, mask) if m == 1]
        prompt_ids = [t for t, m in zip(ids, mask) if m == 0]
        target_text = tok.decode(target_ids, skip_special_tokens=False)
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=False)

        print(f"--- full (first 400 chars): ---\n{full_text[:400]}")
        print(f"--- full (last 300 chars):  ---\n{full_text[-300:]}")
        print(f"--- prompt-only (first 300 chars, masked region): ---\n{prompt_text[:300]}")
        print(f"--- target-only (first 400 chars, unmasked region): ---\n{target_text[:400]}")
        print(f"--- target-only (last 200 chars): ---\n{target_text[-200:]}")

        runs = []
        cur_val = mask[0]
        cur_len = 1
        for m in mask[1:]:
            if m == cur_val:
                cur_len += 1
            else:
                runs.append((cur_val, cur_len))
                cur_val = m
                cur_len = 1
        runs.append((cur_val, cur_len))
        print(f"--- mask runs ({len(runs)} segments): ---")
        for j, (v, l) in enumerate(runs[:12]):
            label = "TARGET" if v == 1 else "prompt"
            print(f"  [{j}] {label}: {l} tokens")
        if len(runs) > 12:
            print(f"  ... ({len(runs) - 12} more)")

        for v, l in runs:
            if v == 1 and l > 0:
                pass

        boundaries_clean = True
        for j in range(1, len(runs)):
            prev_val, _ = runs[j - 1]
            cur_val, _ = runs[j]
            if prev_val == 1 and cur_val == 0:
                pos = sum(r[1] for r in runs[:j])
                last_target_id = ids[pos - 1]
                if last_target_id != end_id:
                    print(
                        f"  WARNING: target run #{j-1} ends at pos {pos} with "
                        f"id {last_target_id} ({tok.decode([last_target_id])!r}), "
                        f"not <|end|>"
                    )
                    boundaries_clean = False
        print(f"--- target runs end with <|end|>: {boundaries_clean} ---")


if __name__ == "__main__":
    main()
