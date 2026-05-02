"""Convert talkie-lm/talkie-web-13b-base's `base.ckpt` (raw fp32 torch state-dict)
into a HuggingFace-loadable directory at /fast/rolmedo/models/talkie-web-13b-base/.

Mirrors `convert_base_to_hf.py` (which targeted talkie-1930-base). The talkie-web
ckpt has the same architecture but a different tokenizer. Pre-reqs:
  - /tmp/talkie-web-base/base.ckpt  exists (HF download finished)
  - /fast/rolmedo/models/talkie-web-13b-base/  has tokenizer.json built by
    build_talkie_web_tokenizer.py.

This script writes the safetensors shards + config.json into that dir.
"""
import os
import shutil
import torch

os.environ.setdefault("HF_HOME", "/tmp")

REF_DIR = "/fast/rolmedo/models/talkie-1930-13b-base"  # take config layout from here
SRC_CKPT = "/tmp/talkie-web-base/base.ckpt"
DST = "/fast/rolmedo/models/talkie-web-13b-base"

from transformers import AutoConfig, AutoModelForCausalLM

# Use the 1930-base config as the template (same architecture, same RoPE
# extension to 64K). vocab_size=65540, max_pos=65536, rope_theta=4e7.
cfg = AutoConfig.from_pretrained(REF_DIR, trust_remote_code=True)
print("config:", cfg)

print(f"loading {SRC_CKPT} ...")
ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False, mmap=True)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
    sd = ckpt["model"]
else:
    sd = ckpt

# strip torch.compile prefix if present
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

print("casting to bf16 ...")
sd = {k: v.to(torch.bfloat16) if hasattr(v, "to") else v for k, v in sd.items()}

# Pad embed and lm_head 65536 -> 65540 with mean of existing rows.
target_vocab = cfg.vocab_size
hidden = cfg.hidden_size
for k in ("embed.weight", "lm_head"):
    t = sd[k]
    cur_v = t.shape[0]
    if cur_v == target_vocab:
        continue
    if cur_v < target_vocab:
        pad_n = target_vocab - cur_v
        mean_row = t.float().mean(dim=0, keepdim=True).to(torch.bfloat16)
        new_rows = mean_row.expand(pad_n, hidden).contiguous().clone()
        sd[k] = torch.cat([t, new_rows], dim=0)
        print(f"padded {k}: {cur_v} -> {sd[k].shape[0]} (added {pad_n} rows from mean)")
    else:
        raise ValueError(f"{k} has {cur_v} > target {target_vocab}")

print(f"raw keys (first 6): {list(sd.keys())[:6]}")
print(f"total keys: {len(sd)}")

# Remap keys: everything goes under "model." except lm_head / lm_head_gain.
TOP_LEVEL = {"lm_head", "lm_head_gain"}
remap = {}
for k, v in sd.items():
    head = k.split(".")[0]
    if head in TOP_LEVEL:
        remap[k] = v
    else:
        remap[f"model.{k}"] = v
print(f"remapped first 6: {list(remap.keys())[:6]}")

print("instantiating TalkieForCausalLM ...")
model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print("loading state dict ...")
missing, unexpected = model.load_state_dict(remap, strict=False)
print(f"missing: {missing}")
print(f"unexpected: {unexpected}")
assert not unexpected, f"unexpected keys after remap: {unexpected}"

benign_missing = {k for k in missing if "_rope_" in k}
hard_missing = [k for k in missing if k not in benign_missing]
assert not hard_missing, f"hard-missing keys: {hard_missing}"
print(f"OK ({len(benign_missing)} benign rope-cache misses)")

os.makedirs(DST, exist_ok=True)
print(f"saving model to {DST} ...")
model = model.to(torch.bfloat16)
model.save_pretrained(DST)

# Copy modeling/config code + the 64K-extended config from 1930 base. The
# tokenizer.json must already be present (from build_talkie_web_tokenizer.py)
# and is NOT overwritten.
for f in (
    "configuration_talkie.py",
    "modeling_talkie.py",
    "config.json",       # 64K-extended config
):
    src = os.path.join(REF_DIR, f)
    dst = os.path.join(DST, f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"copied {f}")

# Sanity: confirm tokenizer.json was already built (we don't overwrite it).
assert os.path.exists(os.path.join(DST, "tokenizer.json")), \
    "tokenizer.json missing — run build_talkie_web_tokenizer.py first"
print("tokenizer.json present (built by build_talkie_web_tokenizer.py)")

print("DONE")
