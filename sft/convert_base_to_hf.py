"""Convert talkie-lm/talkie-1930-13b-base's `final.ckpt` (raw torch state-dict)
into a HuggingFace-loadable directory at /fast/rolmedo/models/talkie-1930-13b-base.

Notes:
  - Base ckpt is float32, ~53 GB. We cast to bf16 -> ~26 GB.
  - Base ckpt uses `_orig_mod.` prefix from torch.compile.
  - Base has vocab_size=65536; IT tokenizer expects 65540 (4 chat tokens added at
    65536-65539). We pad embed and lm_head to 65540 with the mean of existing
    rows so the new tokens start near the embedding centroid.
  - Wrapper layout: everything under `model.` except `lm_head` / `lm_head_gain`.
"""
import os
import shutil
import torch

os.environ.setdefault("HF_HOME", "/tmp")

IT_DIR = "/fast/rolmedo/models/talkie-1930-13b-it"
SRC_CKPT = "/tmp/talkie-13b-base/final.ckpt"
SRC_VOCAB = "/tmp/talkie-13b-base/vocab.txt"  # ignored — IT tokenizer.json is what we need
DST = "/tmp/talkie-1930-13b-base-hf-staging"  # was /fast/...; on /tmp for speed, rsync to /fast at end

# --- Step 1: load template config / modeling
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained(IT_DIR, trust_remote_code=True)
print("config:", cfg)

# --- Step 2: read the raw checkpoint
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

# Cast everything to bf16 (matches IT mirror).
print("casting to bf16 ...")
sd = {k: v.to(torch.bfloat16) if hasattr(v, "to") else v for k, v in sd.items()}

# --- Pad vocab from 65536 -> 65540 (add 4 rows for chat tokens at 65536-65539)
target_vocab = cfg.vocab_size
hidden = cfg.hidden_size
for k in ("embed.weight", "lm_head"):
    t = sd[k]
    cur_v = t.shape[0]
    if cur_v == target_vocab:
        continue
    if cur_v < target_vocab:
        pad_n = target_vocab - cur_v
        # Init new rows from the mean of existing rows -> centroid init.
        mean_row = t.float().mean(dim=0, keepdim=True).to(torch.bfloat16)
        new_rows = mean_row.expand(pad_n, hidden).contiguous().clone()
        sd[k] = torch.cat([t, new_rows], dim=0)
        print(f"padded {k}: {cur_v} -> {sd[k].shape[0]} (added {pad_n} rows from mean)")
    else:
        raise ValueError(f"{k} has {cur_v} > target {target_vocab}")

print(f"raw keys (first 6): {list(sd.keys())[:6]}")
print(f"total keys: {len(sd)}")

# --- Step 3: remap keys for HF wrapper.
TOP_LEVEL = {"lm_head", "lm_head_gain"}
remap = {}
for k, v in sd.items():
    head = k.split(".")[0]
    if head in TOP_LEVEL:
        remap[k] = v
    else:
        remap[f"model.{k}"] = v
print(f"remapped first 6: {list(remap.keys())[:6]}")

# --- Step 4: instantiate model and load
print("instantiating TalkieForCausalLM ...")
model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print("loading state dict ...")
missing, unexpected = model.load_state_dict(remap, strict=False)
print(f"missing: {missing}")
print(f"unexpected: {unexpected}")
assert not unexpected, f"unexpected keys after remap: {unexpected}"

# Model has a non-persistent buffer pattern? Check missing.
benign_missing = {k for k in missing if "_rope_" in k}
hard_missing = [k for k in missing if k not in benign_missing]
assert not hard_missing, f"hard-missing keys: {hard_missing}"
print(f"OK ({len(benign_missing)} benign rope-cache misses)")

# --- Step 5: save to staging dir
os.makedirs(DST, exist_ok=True)
print(f"saving to {DST} ...")
model = model.to(torch.bfloat16)
model.save_pretrained(DST)

# --- Step 6: copy tokenizer + custom code from IT dir
for f in (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "configuration_talkie.py",
    "modeling_talkie.py",
    "config.json",  # 64K-extended config from IT
):
    src = os.path.join(IT_DIR, f)
    dst = os.path.join(DST, f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"copied {f}")

print("DONE")
