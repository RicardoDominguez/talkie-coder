"""Repackage the SFT'd checkpoint so vLLM's transformers backend can load it.

Two transformations:
1. `lm_head` (nn.Parameter, shape (V, H)) → `lm_head.weight`. vLLM constructs
   its own `ParallelLMHead`-style module that expects the `.weight` suffix.
2. Bake `lm_head_gain.w_g` (scalar) into `lm_head.weight` and drop the gain
   tensor. vLLM does not know about our custom per-weight gain — multiplying
   it into the head once preserves correctness.

All other ~441 tensors copy through unchanged: TalkieHeadGain, TalkieActGain
and the per-block scalar gains stay where they are because vLLM's
`recursive_replace` walks the module tree and only swaps Linear/Conv/RMSNorm
classes, leaving our custom modules intact.

Output goes to a sibling directory so the original checkpoint is unaffected.
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", default="/fast/rolmedo/swe-models/talkie-13b-base-swe-1h"
    )
    p.add_argument(
        "--dst", default="/fast/rolmedo/swe-models/talkie-13b-base-swe-1h-vllm"
    )
    p.add_argument(
        "--tmp", default="/tmp",
        help="local fast disk to stage the safetensors before dd to Lustre",
    )
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    src_st = os.path.join(args.src, "model.safetensors")
    dst_st = os.path.join(args.dst, "model.safetensors")

    print(f"loading {src_st}")
    tensors = {}
    with safe_open(src_st, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            tensors[k] = f.get_tensor(k)

    assert "lm_head" in tensors, "expected key 'lm_head' in source safetensors"
    assert "lm_head_gain.w_g" in tensors, "expected key 'lm_head_gain.w_g'"

    lm_head = tensors.pop("lm_head")
    gain = tensors.pop("lm_head_gain.w_g")
    gain_val = float(gain.item())
    print(f"baking lm_head_gain.w_g = {gain_val:.6f} into lm_head; "
          f"shape={tuple(lm_head.shape)}, dtype={lm_head.dtype}")
    lm_head_scaled = lm_head.to(torch.float32) * gain_val
    tensors["lm_head.weight"] = lm_head_scaled.to(lm_head.dtype).contiguous()

    # Stage the safetensors on local fast disk (page-cached) then dd with
    # oflag=direct to Lustre. Plain safetensors.save_file straight to Lustre
    # tops out at ~20 MB/s; staging+dd is ~10x faster on 25 GB writes.
    with tempfile.NamedTemporaryFile(
        suffix=".safetensors", dir=args.tmp, delete=False
    ) as tmp:
        tmp_st = tmp.name
    try:
        print(f"staging to {tmp_st}")
        save_file(tensors, tmp_st, metadata={"format": "pt"})
        print(f"dd {tmp_st} -> {dst_st}")
        subprocess.check_call([
            "dd", f"if={tmp_st}", f"of={dst_st}",
            "bs=64M", "oflag=direct", "status=progress",
        ])
    finally:
        if os.path.exists(tmp_st):
            os.remove(tmp_st)

    # Copy the rest of the model dir so the new checkpoint is self-contained.
    for fname in os.listdir(args.src):
        if fname == "model.safetensors":
            continue
        src = os.path.join(args.src, fname)
        dst = os.path.join(args.dst, fname)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"copied {fname}")

    print(f"done -> {args.dst}")


if __name__ == "__main__":
    main()
