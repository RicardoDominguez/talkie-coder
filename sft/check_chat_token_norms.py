"""Inspect chat-token row norms in embed.weight and lm_head of a saved
talkie checkpoint. Used to validate the chat-token-collapse fix in ft_trl.py.

Usage:
    python check_chat_token_norms.py <model_dir>

Prints norms for tokens 65535..65539 plus a few "control" rows for context.
"""
import sys
from pathlib import Path
from safetensors import safe_open

CHAT_TOKENS = {
    65535: "<|endoftext|>",
    65536: "<|end|>",
    65537: "<|user|>",
    65538: "<|assistant|>",
    65539: "<|system|>",
}
CONTROL_TOKENS = [0, 100, 1000, 30000]  # arbitrary non-chat rows for comparison


def find_safetensors(model_dir: Path):
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    return files


def collect_norms(model_dir: Path):
    files = find_safetensors(model_dir)
    embed_norms, lm_norms, w_g = {}, {}, None
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            keys = set(h.keys())
            if "model.embed.weight" in keys:
                embed = h.get_tensor("model.embed.weight").float()
                for tid in list(CHAT_TOKENS) + CONTROL_TOKENS:
                    embed_norms[tid] = embed[tid].norm().item()
            if "lm_head" in keys:
                lm = h.get_tensor("lm_head").float()
                for tid in list(CHAT_TOKENS) + CONTROL_TOKENS:
                    lm_norms[tid] = lm[tid].norm().item()
            if "lm_head_gain.w_g" in keys:
                w_g = h.get_tensor("lm_head_gain.w_g").float().item()
    return embed_norms, lm_norms, w_g


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    embed_norms, lm_norms, w_g = collect_norms(model_dir)

    print(f"\n{model_dir}")
    print(f"  lm_head_gain.w_g = {w_g:.4f}" if w_g is not None else "  lm_head_gain.w_g = (not found)")
    print(f"\n  CHAT TOKENS")
    print(f"  {'tid':>5}  {'name':<14}  {'embed':>8}  {'lm_head':>8}  {'lm * w_g':>9}")
    for tid, name in CHAT_TOKENS.items():
        e = embed_norms.get(tid, float("nan"))
        l = lm_norms.get(tid, float("nan"))
        eff = (l * w_g) if (w_g is not None) else float("nan")
        print(f"  {tid:>5}  {name:<14}  {e:8.4f}  {l:8.4f}  {eff:9.4f}")
    print(f"\n  CONTROL ROWS (non-chat, for comparison)")
    print(f"  {'tid':>5}  {'embed':>8}  {'lm_head':>8}  {'lm * w_g':>9}")
    for tid in CONTROL_TOKENS:
        e = embed_norms.get(tid, float("nan"))
        l = lm_norms.get(tid, float("nan"))
        eff = (l * w_g) if (w_g is not None) else float("nan")
        print(f"  {tid:>5}  {e:8.4f}  {l:8.4f}  {eff:9.4f}")

    print()
    chat_lm = [lm_norms[tid] for tid in CHAT_TOKENS if tid != 65535 and tid in lm_norms]
    if chat_lm:
        avg = sum(chat_lm) / len(chat_lm)
        print(f"  avg chat-token lm_head norm (excl <|endoftext|>): {avg:.4f}")
        print("  expected ranges:")
        print("    healthy (no collapse): ~0.85+ (preserves cloned <|endoftext|> norm)")
        print("    collapsed (bug):       ~0.12-0.22")


if __name__ == "__main__":
    main()
