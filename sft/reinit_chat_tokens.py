"""Re-initialize the 4 chat-token rows (65536-65539) in embed.weight and lm_head
from the existing <|endoftext|> row (id 65535) instead of the centroid mean.

In place: rewrites the safetensors shard(s) of /fast/rolmedo/models/talkie-1930-13b-base.

Rationale: centroid-init makes all 4 rows identical and produces wild gradients
during SFT. <|endoftext|> already has a trained, semantically-sensible
"sequence-boundary" representation; copying it gives the new tokens a
non-degenerate starting point. We add tiny noise so the 4 rows aren't bit-exact
duplicates (otherwise gradients would be identical and they couldn't
differentiate).
"""
import os
os.environ.setdefault("HF_HOME", "/tmp")
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

MODEL = "/tmp/talkie-1930-13b-base"  # was /fast/...; on /tmp for speed, rsync to /fast at end
SRC_ID = 65535      # <|endoftext|>
TARGET_IDS = [65536, 65537, 65538, 65539]  # <|end|>, <|user|>, <|assistant|>, <|system|>
NOISE = 1e-3        # std of additive Gaussian noise to break symmetry

torch.manual_seed(0)
print("loading model ...")
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16)

with torch.no_grad():
    embed = model.model.embed.weight  # (65540, 5120)
    lm_head = model.lm_head            # (65540, 5120)
    src_emb = embed[SRC_ID].clone()
    src_lm = lm_head[SRC_ID].clone()
    print(f"src embed[{SRC_ID}] norm={src_emb.float().norm():.4f}")
    print(f"src lm_head[{SRC_ID}] norm={src_lm.float().norm():.4f}")
    for tid in TARGET_IDS:
        old_e_norm = embed[tid].float().norm().item()
        old_l_norm = lm_head[tid].float().norm().item()
        embed[tid] = src_emb + torch.randn_like(src_emb) * NOISE
        lm_head[tid] = src_lm + torch.randn_like(src_lm) * NOISE
        print(f"  tid={tid}: embed norm {old_e_norm:.4f} -> {embed[tid].float().norm():.4f}, "
              f"lm_head norm {old_l_norm:.4f} -> {lm_head[tid].float().norm():.4f}")

print("saving ...")
model.save_pretrained(MODEL)
print("DONE")
