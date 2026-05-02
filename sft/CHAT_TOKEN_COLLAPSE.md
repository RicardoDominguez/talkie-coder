# Chat-token collapse in base→SFT models

## Summary

After SFT'ing the talkie-1930-13b-base model on either GSM8K or SWE-bench
data, the chat-template special tokens (`<|end|>`, `<|user|>`, `<|assistant|>`,
`<|system|>` — IDs 65536–65539) become unusable. Generations from any
chat-formatted prompt are nonsense, and the model never emits `<|end|>` so
generation runs to `max_tokens` regardless of how `eos_token_id` is configured.

The same SFT run on the talkie-1930-13b-**it** model produces a clean working
chat model. The bug is specific to the base→SFT pipeline.

## Symptoms

Tested on `/fast/rolmedo/swe-models/talkie-1930-13b-base-swe-12h/checkpoint-1000`
via HF `model.generate` (no vLLM, no custom inference path) with
`eos_token_id=[<|end|>, <|endoftext|>]`. Reproduction snippets below.

### Plain-text prompts (no chat template) — base capability preserved

```
prompt: "The capital of France is"
output: " Paris, which is the largest city in Europe. It is situated on
         the river Seine, and is noted for its magnificent buildings, …"
```

Coherent, factually correct, characteristic of the underlying base model.

### Plain-text code prompts — degraded into repetition

```
prompt: "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
output: "      return n\n    \n    if n > 1:\n        return n\n    \n
         if n < 1:\n        return n\n    \n    if n <= 1:\n        return n
         …"  (repeats indefinitely)
```

### Chat-formatted prompts — nonsense, no termination

All of the following used `tok.apply_chat_template(..., add_generation_prompt=True)`,
producing prompts that end in `<|assistant|>`. Each ran for the full
`max_new_tokens` budget; `<|end|>` was never emitted.

```
prompt: <|user|>What is 2+2?<|end|><|assistant|>
output: ", 2+2=4. \n\nWhat is 3+3? 3+3=6. What is 4+4? 4+4=8. …"
        — math correct, but generates an infinite Q&A loop, no termination.

prompt: <|user|>Name three colors. Answer briefly.<|end|><|assistant|>
output: ", 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. …"
        — content collapsed to a numeric ramp.

prompt: <|user|>Say hi.<|end|><|assistant|>
output: ", 1. 1. 1. 1. 1. 1. 1. 1. 1."
        — single token repeats.

prompt: <|system|>You are a helpful assistant…<|end|><|user|>
        <uploaded_files>/repo</uploaded_files>
        Fix the bug in /repo. Start by exploring.<|end|><|assistant|>
output: "THOUGHTOnline if you can find the bug. If not, go to
         https://www.gutenberg.org/backup/usercontent/1/2/Lairs/usercontent/
         usercontent/usercontent/usercontent/usercontent/…"
        — first token "THOUGHT" then random URL spam, then path-segment loop.
```

Same symptoms appear in the GSM8K-SFT'd base checkpoint
(`/fast/rolmedo/swe-models/talkie-13b-base-gsm8k-sft-8kctx`):

```
prompt: <|user|>What is 2+2?<|end|><|assistant|>
output: "2+2=<<2+2=4>>4\n#### 4 ﬁﬁﬁﬁﬁﬁﬁﬁﬁﬁﬁﬁﬁ�"
        — correct GSM8K format ("<<2+2=4>>4\n#### 4") followed by a single
        token repeating, no <|end|>.
```

The IT-derived counterpart (`/fast/rolmedo/swe-models/talkie-13b-it-gsm8k-sft-8kctx`)
produces a clean termination on the same prompt:

```
output: "2+2=4>>4\n#### 4<|end|>"   ← stops on <|end|>, finish_reason=stop
```

## Root cause

The chat-token rows of `model.embed.weight` and `lm_head` collapse during
base→SFT. Magnitudes (L2 norm) measured with `safetensors.safe_open`:

| Token | BASE 12h SWE-SFT | BASE GSM8K-SFT | IT pre-SFT | IT GSM8K-SFT |
|---|---|---|---|---|
| `<\|endoftext\|>` (65535) embed / lm | 0.864 / 0.887 | 0.864 / 0.887 | 0.864 / 0.887 | 0.864 / 0.887 |
| `<\|end\|>` (65536) embed / lm | **0.123 / 0.224** | **0.123 / 0.205** | 1.452 / 1.438 | 1.452 / 1.438 |
| `<\|user\|>` (65537) embed / lm | **0.123 / 0.204** | **0.123 / 0.205** | 1.432 / 1.475 | 1.432 / 1.475 |
| `<\|assistant\|>` (65538) embed / lm | **0.123 / 0.204** | **0.123 / 0.205** | 1.418 / 1.443 | 1.418 / 1.443 |

Two SFT runs on different data (GSM8K and SWE-bench, hours-of-training apart)
arrived at near bit-identical norms (e.g., embed_norm 0.1228 vs 0.1229,
lm_head_norm 0.2044 vs 0.2049). The IT model — and any model SFT'd from
it — preserves chat-token norms in the ~1.4 range.

### Mechanism

1. `sft/reinit_chat_tokens.py` initializes the four added chat-token rows
   (65536–65539) by cloning the `<|endoftext|>` row (65535) and adding
   N(0, 1e-3) noise. Initial norm ≈ 0.86 / 0.89.

2. SFT runs with `--weight_decay 0.1` and `--completion_only_loss 1`. In
   completion-only mode, the only chat token that is ever a loss *target* is
   `<|end|>`, and even then only at the end of an assistant turn — roughly
   13 occurrences per 65 536-token packed sequence (≈0.02% of positions).
   `<|user|>`, `<|assistant|>`, `<|system|>` are never targets.

3. AdamW per step: `θ -= lr * (m̂ / (√v̂ + eps) + wd * θ)`. The decoupled
   weight-decay term `lr * wd * θ` is applied every step regardless of
   whether the parameter received a useful gradient that step. For
   `<|user|>`/`<|assistant|>`/`<|system|>`, `m̂` and `v̂` for the lm_head
   row stay near zero (no targets); only the wd term acts, and the row
   monotonically shrinks toward zero. For `<|end|>`, the rare gradient
   updates are not enough to outpace wd, so it shrinks too — just slightly
   less than the others (the small remaining norm gap, 0.22 vs 0.20, is the
   only signature of useful learning on `<|end|>`).

4. `lm_head_gain.w_g` (a global scalar that multiplies all of `lm_head`
   before the output projection) grew from 1.0 → 3.89 over training, an
   apparent compensation for the shrunken rows. But `w_g` scales *all*
   rows uniformly, so `<|endoftext|>` ends up with effective output norm
   `3.89 × 0.887 ≈ 3.45` while `<|end|>` gets `3.89 × 0.224 ≈ 0.87`.
   `<|end|>` cannot win the logit competition against any neighboring
   vocabulary token.

5. The same collapse on the input side (`embed.weight` rows) means that
   when a chat-formatted prompt ends with `<|assistant|>`, the embedding
   fed into the model at that position is near-zero — a poor conditioning
   signal — explaining why the model's first generated token in chat mode
   is structurally wrong (`THOUGHTOnline`, `, 1. 2.`, etc.).

The IT checkpoint avoids the bug because the chat tokens were trained
during instruction tuning to norm ≈ 1.4 before any SFT started; further
SFT preserves them.

## How to reproduce

The two snippets below run on a single H100 in the existing `tflatest`
environment.

### 1. Symptom — chat-formatted generation fails to terminate

```python
import os, torch
os.environ.setdefault("HF_HOME", "/tmp")
from transformers import AutoTokenizer, AutoModelForCausalLM

MDIR = "/fast/rolmedo/swe-models/talkie-1930-13b-base-swe-12h/checkpoint-1000"
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, trust_remote_code=True, dtype=torch.bfloat16
).cuda().eval()

end_id = tok.convert_tokens_to_ids("<|end|>")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok([prompt], return_tensors="pt").input_ids.cuda()
out = model.generate(
    input_ids=ids, max_new_tokens=120, do_sample=False,
    pad_token_id=tok.pad_token_id,
    eos_token_id=[end_id, tok.eos_token_id],
)
new = out[0, ids.shape[1]:]
print("emit_<|end|>:", end_id in new.tolist())   # → False
print(repr(tok.decode(new, skip_special_tokens=False)))
# ", 2+2=4. \n\nWhat is 3+3? 3+3=6. What is 4+4? 4+4=8. ..."
```

Compare against `/fast/rolmedo/swe-models/talkie-13b-it-gsm8k-sft-8kctx` with
the same code: produces `'2+2=4>>4\\n#### 4<|end|>'` and stops cleanly.

### 2. Root cause — collapsed chat-token rows in lm_head and embed

```python
import torch
from safetensors import safe_open

CHECKPOINTS = {
    "BASE 12h SWE-SFT": "/fast/rolmedo/swe-models/talkie-1930-13b-base-swe-12h/checkpoint-1000/model.safetensors",
    "BASE GSM8K-SFT":   "/fast/rolmedo/swe-models/talkie-13b-base-gsm8k-sft-8kctx/model.safetensors",
    "IT pre-SFT":       "/fast/rolmedo/models/talkie-1930-13b-it/model.safetensors",
    "IT GSM8K-SFT":     "/fast/rolmedo/swe-models/talkie-13b-it-gsm8k-sft-8kctx/model.safetensors",
}
TOKENS = {65535: "<|endoftext|>", 65536: "<|end|>", 65537: "<|user|>",
          65538: "<|assistant|>", 65539: "<|system|>"}

for name, path in CHECKPOINTS.items():
    with safe_open(path, framework="pt", device="cpu") as f:
        embed = f.get_tensor("model.embed.weight")
        lm = f.get_tensor("lm_head")
    print(f"--- {name} ---")
    for tid, tname in TOKENS.items():
        print(f"  {tid} {tname:14s} "
              f"embed={embed[tid].float().norm():.3f}  "
              f"lm_head={lm[tid].float().norm():.3f}")
```

Expected: BASE→SFT runs show chat-token norms in the 0.12 / 0.20 range;
IT→SFT shows ~1.4. The collapse is deterministic across SFT runs that share
hyperparameters, so any base→SFT run with the same `--weight_decay` and
`--completion_only_loss` settings reproduces it.

## Configuration provenance

- Base model used: `/fast/rolmedo/models/talkie-1930-13b-base`,
  with chat tokens added by `sft/reinit_chat_tokens.py`
  (clone `<|endoftext|>` row + 1e-3 Gaussian noise, vocab 65536→65540).
- SFT scripts that exhibit the bug: `sft/run_swe_sft_1h.sh`,
  `sft/run_swe_sft_12h.sh`, `sft/run_gsm8k_sft.sh`,
  `sft/run_gsm8k_sft_8k_base.sh`. All use:
  `--learning_rate 2e-6 --weight_decay 0.1 --max_grad_norm 100
  --completion_only_loss 1 --packing --padding_free
  --optim adamw_torch_fused`, with `tie_word_embeddings: false` in
  `configuration_talkie.py`.
- Loss-mask audit on `/fast/rolmedo/swesmith/datasets/talkie-swe-8k-64k`:
  every assistant turn ends with `<|end|>`, and the `completion_mask` value
  at the `<|end|>` position is 1 (so the loss is computed for predicting
  `<|end|>`). Roughly 90–150 such targets per 64K-token packed sequence
  across the dataset's 2122 examples. The data is correctly labeled; the
  failure is on the optimization side.
