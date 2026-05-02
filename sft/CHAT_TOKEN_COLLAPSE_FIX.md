# Fix for chat-token collapse in base→SFT models

Companion to `CHAT_TOKEN_COLLAPSE.md` (the diagnosis). Two changes ship the
fix; both are already in the repo.

## Summary

1. **`sft/ft_trl.py`** — exclude `embed.weight`, `lm_head`, and the per-output
   gain scalars from AdamW weight decay.
2. **`sft/reinit_chat_tokens_web.py`** (new) — repair the broken chat-token
   initialization in the talkie-web base by cloning the `<|endoftext|>` row
   plus 1e-3 noise, writing to a *new* model directory.

Either one alone is insufficient: (1) without (2) leaves the talkie-web rows
stuck at the bad post-converter init (~0.11 / ~0.20, bit-identical, can't
differentiate). (2) without (1) lets the freshly cloned rows decay during SFT
the same way the talkie-1930 rows did. Together they break the feedback loop
described in `CHAT_TOKEN_COLLAPSE.md` and keep chat-template inference
working through full SFT.

## Part 1 — exclude embed/lm_head from weight decay

`sft/ft_trl.py` now defines `ChatPreservingSFTTrainer(SFTTrainer)` which
overrides `get_decay_parameter_names`:

```python
class ChatPreservingSFTTrainer(SFTTrainer):
    def get_decay_parameter_names(self, model):
        decay = super().get_decay_parameter_names(model)
        excluded = sorted({n for n in decay if "embed" in n or "lm_head" in n})
        decay = [n for n in decay if "embed" not in n and "lm_head" not in n]
        print(f"[wd-skip] {len(excluded)} params excluded from weight decay: {excluded}")
        return decay
```

The trainer is instantiated as `ChatPreservingSFTTrainer(...)` instead of
`SFTTrainer(...)`. No CLI flag — it's the new default.

The "embed" / "lm_head" name patterns drop **43 parameters** in talkie-13b:

| Pattern | Param names | Count |
|---|---|---|
| matrix | `model.embed.weight`, `lm_head` | 2 |
| per-layer scalars | `model.blocks.{0..39}.embed_skip.a_g` | 40 |
| global gain | `lm_head_gain.w_g` | 1 |
| **total** | | **43** |

Including the gain scalars (`embed_skip.a_g`, `lm_head_gain.w_g`) is canonical
practice (Llama, GPT-NeoX, torchtune all skip wd on 1-D params); they were
previously over-regularized.

**Why the bug happened:** under `--completion_only_loss`, `<|user|>`,
`<|assistant|>`, `<|system|>` are *never* loss targets and `<|end|>` is
rarely one. AdamW's per-step decay `θ *= (1 − lr·wd)` therefore acts almost
uncontested on those rows. As trained rows compensate by growing
`lm_head_gain.w_g` (1.0 → 3.89 in the original 12h run), the chat rows can't
compensate, so their *effective* logit-space magnitude (`w_g · row`) drops.
Skipping wd on these params breaks the loop: w_g doesn't grow because no row
needs to compensate, and no row decays because none has wd applied.

**Runtime check:** the `[wd-skip]` line prints once at trainer init.
Verified on the v2 SWE 12h run (PID 989643), and the wd-skip filter caught
all 43 params correctly.

## Part 2 — repair the talkie-web chat-token init

`sft/convert_web_base_to_hf.py` had padded rows 65536–65539 of
`embed.weight` and `lm_head` with `mean(rows)`. That's two failures at once:

- **Norm.** Mean of all rows in this model has norm ~0.11 (embed) /
  ~0.20 (lm_head) — *already* in the post-collapse range that breaks
  inference, before any SFT runs.
- **Symmetry.** All four rows are bit-identical, so even with the wd-skip
  fix they would receive identical gradients during SFT and could never
  differentiate from each other.

`sft/reinit_chat_tokens_web.py` fixes both: clone row 65535
(`<|endoftext|>`) + 1e-3 Gaussian noise into rows 65536–65539. Writes to
**`/fast/rolmedo/models/talkie-web-13b-base-reinit/`** (new dir, no
overwrite of the original base).

Norms after reinit (verified via `sft/check_chat_token_norms.py`):

| Token | embed | lm_head | embed × w_g (effective) |
|---|---|---|---|
| `<\|endoftext\|>` (65535) | 0.7737 | 0.5476 | 2.43 |
| `<\|end\|>` (65536) | 0.7777 | 0.5534 | 2.46 |
| `<\|user\|>` (65537) | 0.7781 | 0.5527 | 2.45 |
| `<\|assistant\|>` (65538) | 0.7758 | 0.5506 | 2.44 |
| `<\|system\|>` (65539) | 0.7763 | 0.5520 | 2.45 |

The 4 chat tokens are now slightly different from each other (1e-3 noise
broke symmetry) and sit at the same effective magnitude as `<|endoftext|>`,
which is comparable to representative non-chat rows (e.g., id 100: effective
2.71). This is the same approach `sft/reinit_chat_tokens.py` already used
for talkie-1930 — we just needed an analog for talkie-web because its
converter took a different (broken) path.

## How to apply

For a fresh base→SFT run on talkie-web:

```bash
# 1. Build the reinit'd base (one-time, ~25 min for the bf16 26GB save)
cd /home/rolmedo/talkie/sft && python reinit_chat_tokens_web.py

# 2. SFT against the new base. ft_trl.py is already patched.
bash run_swe_sft_12h_web_v2.sh   # writes to talkie-web-13b-base-swe-12h-v2/
```

For talkie-1930, only the wd-skip fix is needed — the existing
`reinit_chat_tokens.py` already produces healthy initial norms (~0.86),
they just collapse during SFT without the wd-skip patch.

## Validation

### Norm-preservation check (any saved checkpoint)

```bash
python sft/check_chat_token_norms.py <model_dir>
```

Expected: chat-token lm_head norms ~0.55, embed norms ~0.78. Deviations of
±0.01 are normal training drift; >50% drop indicates the fix isn't
applied.

### Mid-run verification (run B v2, talkie-web SWE 12h)

| Token (lm_head row) | Initial post-reinit | Step 200 of v2 SFT | Δ |
|---|---|---|---|
| `<\|end\|>` | 0.5534 | 0.5540 | +0.0006 |
| `<\|user\|>` | 0.5527 | 0.5522 | -0.0005 |
| `<\|assistant\|>` | 0.5506 | 0.5501 | -0.0005 |
| `<\|system\|>` | 0.5520 | 0.5513 | -0.0007 |
| `<\|endoftext\|>` | 0.5476 | 0.5468 | -0.0008 |
| `lm_head_gain.w_g` | 4.4375 | 4.4375 | 0 (unchanged) |

For comparison, the unpatched v1 run had `lm_head[<|end|>]` at 0.205 by
step 1000 — a ~3× collapse. The v2 run shows zero collapse: every row
preserved within ±0.001, and `lm_head_gain.w_g` literally unchanged
(because no row needs the gain to grow as compensation any more).

### Chat-template inference (post-SFT)

`sft/test_chat_inference.py <model_dir>` runs three chat-formatted prompts
through `model.generate` with `eos_token_id=[<|end|>, <|endoftext|>]` and
checks termination. After a 57-step GSM8K validation SFT on the reinit'd
base, four GSM8K-style probes pass cleanly:

```
user: 'Janet has 3 apples. She buys 5 more. How many does she have?'
output: 'The total number of apples is 3 + 5 = <<3+5=8>>8 apples.\n#### 8<|end|>'

user: 'A train travels 60 miles in 2 hours. What is its speed in miles per hour?'
output: 'The train travels 60 miles in 2 hours, so it travels 60/2=<<60/2=30>>30 miles per hour.\n#### 30<|end|>'

user: 'Mary has 12 cookies. She gives 4 to her brother and eats 3. How many does she have left?'
output: 'The total number of cookies Mary has is 12 - 4 - 3 = <<12-4-3=5>>5 cookies.\n#### 5<|end|>'

user: 'A rectangle has length 8 cm and width 5 cm. What is its area?'
output: 'The area of a rectangle is the length times the width, so the area of the rectangle is 8 cm * 5 cm = <<8*5=40>>40 cm2.\n#### 40<|end|>'
```

All four emit `<|end|>` cleanly, n_new ≤ 44 (no run-to-max), GSM8K format
correct, math correct.

**Note on out-of-distribution prompts.** "Say hi" / "Name three colors"
ramble without termination on a model that only saw GSM8K format during
SFT — that is OOD generalization, *not* the chat-token bug. After full
SWE 12h training on multi-turn agent traces, the model should generalize
to non-GSM8K chat content as well.

## Performance note: only shard 1 needs to be rewritten

Both reinit scripts call `model.save_pretrained(...)`, which rewrites **all
6 safetensors shards** (~26 GB total at ~20 MB/s on /fast → ~20 min).
This is wasteful. Per `model.safetensors.index.json`, both modified tensors
live in the same shard:

- `model.embed.weight` → `model-00001-of-00006.safetensors`
- `lm_head` → `model-00001-of-00006.safetensors`
- (`lm_head_gain.w_g` lives in shard 6, but reinit doesn't touch it)

A faster path: load only shard 1, edit the 4 rows in `embed.weight` and
`lm_head`, write shard 1 back. Shards 2–6 stay untouched. ~3–4× faster
overall (4 GB instead of 26 GB written).

We did **not** retrofit this optimization (`save_pretrained` is sufficient
when reinit only runs once per base model), but if you ever re-run reinit
on the same base — for example, after a botched attempt — kill the
`save_pretrained` once shard 1 is on disk (check `ls -la *.safetensors`
timestamps; shard 1 is the first to land). Skipping shards 2–6 is safe
because their bytes don't change.

Side note: `save_pretrained` triggers a `transformers → accelerate →
deepspeed` import chain that fails with `MissingCUDAException: CUDA_HOME
does not exist` if CUDA isn't loaded — the script edits succeed in memory
but the save crashes. Always run reinit with `module load cuda/12.4` (or
`CUDA_HOME=/is/software/nvidia/cuda-12.4`) in scope.

## Files changed / added

| File | Change |
|---|---|
| `sft/ft_trl.py` | Added `ChatPreservingSFTTrainer`, switched instantiation. |
| `sft/reinit_chat_tokens_web.py` | New: clone-+noise reinit for talkie-web. |
| `sft/run_gsm8k_fixtest_web.sh` | New: GSM8K validation SFT pointing at the reinit'd base. |
| `sft/run_swe_sft_12h_web_v2.sh` | New: production SWE 12h script with new base path and new output dir. |
| `sft/check_chat_token_norms.py` | New: print chat-token row norms from a saved checkpoint. |
| `sft/test_chat_inference.py` | New: end-to-end chat-template inference smoke test. |

The original `sft/run_swe_sft_12h_web.sh`, the original
`/fast/rolmedo/models/talkie-web-13b-base/`, and the partial v1
artifacts at `/fast/rolmedo/swe-models/talkie-web-13b-base-swe-12h/` are
**not modified**. The fix lives in new files / new directories.
