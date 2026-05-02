"""Re-initialize the 4 chat-token rows (65536-65539) in embed.weight and lm_head
of the talkie-web-13b-base model by cloning the <|endoftext|> row (id 65535)
plus 1e-3 Gaussian noise.

Why: convert_web_base_to_hf.py padded those 4 rows with the *mean* of existing
rows, producing low-norm AND bit-identical vectors (norm 0.11 embed / 0.20
lm_head). Mean-init means the 4 rows can never differentiate during training
(identical gradient → identical update), and the low norm leaves them unable
to win the logit competition at inference, so chat-template prompts fall over
(no <|end|> emission, nonsense generations). See talkie-eval/CHAT_TOKEN_COLLAPSE.md.

Cloning <|endoftext|> + small noise mirrors the talkie-1930 setup
(reinit_chat_tokens.py) and gives the chat tokens a non-degenerate starting
point: a trained, semantically meaningful "sequence boundary" representation,
with enough noise to break symmetry across the 4 rows.

Writes to a NEW directory (no overwrite of /fast/rolmedo/models/talkie-web-13b-base).
"""
import os, shutil
os.environ.setdefault("HF_HOME", "/tmp")
import torch
from transformers import AutoModelForCausalLM

SRC = "/fast/rolmedo/models/talkie-web-13b-base"
DST = "/fast/rolmedo/models/talkie-web-13b-base-reinit"
SRC_ID = 65535      # <|endoftext|>
TARGET_IDS = [65536, 65537, 65538, 65539]  # <|end|>, <|user|>, <|assistant|>, <|system|>
NOISE = 1e-3        # std of additive Gaussian noise to break symmetry

torch.manual_seed(0)
print(f"loading model from {SRC} ...")
model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=torch.bfloat16)

with torch.no_grad():
    embed = model.model.embed.weight        # (65540, 5120)
    lm_head = model.lm_head                  # (65540, 5120) — direct Parameter
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

print(f"saving to {DST} ...")
os.makedirs(DST, exist_ok=True)
model.save_pretrained(DST)

# Copy the small files (tokenizer, modeling code, chat template, etc.) that
# save_pretrained doesn't necessarily reproduce identically.
for fname in (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "configuration_talkie.py", "modeling_talkie.py",
    "generation_config.json",
):
    src = os.path.join(SRC, fname)
    dst = os.path.join(DST, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"  copied {fname}")
print("DONE")
