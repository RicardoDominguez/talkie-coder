"""Build a HF-compatible tokenizer for talkie-web from its vocab.txt.

The talkie inference repo (github.com/talkie-lm/talkie/src/talkie/tokenizer.py)
loads vocab.txt as a tiktoken BPE: each line is "<base64-token> <id>", and the
final encoder is a tiktoken.Encoding with the talkie regex pattern.

We replicate that as a HF `PreTrainedTokenizerFast` by:
  1. Running transformers' `TikTokenConverter` on vocab.txt → BPE vocab + merges.
  2. Wrapping with PreTrainedTokenizerFast and adding the 5 chat special tokens
     at ids 65535-65539 (matching the talkie IT layout used by talkie-1930-base).
  3. Verifying round-trip + parity with the tiktoken reference on plain text.

The talkie-1930 base tokenizer was built the same way (see
/fast/rolmedo/models/talkie-1930-13b-base/tokenizer.json — model_type=BPE,
ByteLevel pre/post, 65540 total tokens). We reuse the same chat_template.jinja
and tokenizer_config.json.
"""
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/tmp")

from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

VOCAB_PATH = "/tmp/talkie-web-base/vocab.txt"
DST = "/fast/rolmedo/models/talkie-web-13b-base"
REF_DIR = "/fast/rolmedo/models/talkie-1930-13b-base"  # for chat_template

# talkie regex pattern (matches src/talkie/tokenizer.py).
TALKIE_PAT = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

BASE_VOCAB_SIZE = 65536  # base model emb table is (65536, 5120); endoftext is at id 65535
SPECIAL_TOKENS = [
    ("<|endoftext|>", BASE_VOCAB_SIZE - 1),  # 65535
    ("<|end|>", BASE_VOCAB_SIZE),            # 65536
    ("<|user|>", BASE_VOCAB_SIZE + 1),       # 65537
    ("<|assistant|>", BASE_VOCAB_SIZE + 2),  # 65538
    ("<|system|>", BASE_VOCAB_SIZE + 3),     # 65539
]


def main():
    os.makedirs(DST, exist_ok=True)

    # talkie's tokenizer.py filters mergeable_ranks to {v < BASE_VOCAB_SIZE-1};
    # vocab.txt actually contains 262144 entries but the model embed table only
    # has 65536 rows. Write a truncated copy to a temp file for TikTokenConverter.
    truncated_vocab = "/tmp/talkie-web-base/vocab_65535.txt"
    with open(VOCAB_PATH) as fin, open(truncated_vocab, "w") as fout:
        kept = 0
        for line in fin:
            parts = line.rstrip("\n").split()
            if len(parts) != 2:
                continue
            try:
                rank = int(parts[1])
            except ValueError:
                continue
            if rank < BASE_VOCAB_SIZE - 1:  # ranks 0..65534
                fout.write(line)
                kept += 1
    print(f"vocab.txt: filtered to {kept} entries (rank < {BASE_VOCAB_SIZE - 1})")

    # 1) tiktoken-style BPE -> HF Tokenizer.
    print("converting vocab.txt -> HF BPE tokenizer ...")
    converter = TikTokenConverter(
        vocab_file=truncated_vocab,
        pattern=TALKIE_PAT,
        add_prefix_space=False,
        additional_special_tokens=[t for t, _ in SPECIAL_TOKENS],
    )
    tk = converter.converted()
    bpe_vocab = tk.get_vocab()
    print(f"BPE vocab size after conversion: {len(bpe_vocab)} (expected ~65535 + 5 specials)")

    # 2) Wrap as PreTrainedTokenizerFast.
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token=AddedToken("<|endoftext|>", normalized=False, special=True),
        pad_token=AddedToken("<|endoftext|>", normalized=False, special=True),
        bos_token=None,
        unk_token=None,
        clean_up_tokenization_spaces=False,
        model_max_length=65536,
    )
    print(f"tokenizer len: {len(fast)} vocab_size: {fast.vocab_size}")

    # 3) Verify chat tokens land on the right ids (must match _IT_SPECIAL_TOKENS).
    for s, want in SPECIAL_TOKENS:
        got = fast.convert_tokens_to_ids(s)
        print(f"  {s!r}: id={got} (expected {want}) {'OK' if got == want else 'MISMATCH'}")
        assert got == want, f"{s} got {got} expected {want}"

    # 4) Round-trip a plain English sentence + a sentence containing chat tokens.
    text = "The cat sat on the mat. SWE-bench evaluates patch correctness."
    ids = fast.encode(text, add_special_tokens=False)
    rt = fast.decode(ids, skip_special_tokens=False)
    print(f"plain text encode -> {len(ids)} ids; decode roundtrip == input: {rt == text}")
    if rt != text:
        print(f"  expected: {text!r}")
        print(f"  got:      {rt!r}")

    # 5) Parity check vs. the tiktoken reference (the source of truth for talkie).
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        ranks = load_tiktoken_bpe(truncated_vocab)
        ref = tiktoken.Encoding(
            name="talkie-base-ref",
            pat_str=TALKIE_PAT,
            mergeable_ranks=ranks,
            special_tokens={"<|endoftext|>": BASE_VOCAB_SIZE - 1},
        )
        ref_ids = ref.encode(text)
        print(f"tiktoken-ref ids:   {ref_ids[:20]} ... ({len(ref_ids)})")
        print(f"hf-converted ids:   {ids[:20]} ... ({len(ids)})")
        if ref_ids == ids:
            print("PARITY OK (HF tokenizer matches tiktoken on plain text)")
        else:
            print("PARITY MISMATCH — investigate before training!")
    except Exception as e:
        print(f"tiktoken parity skipped: {e}")

    # 6) Save tokenizer.json + tokenizer_config.json (chat template injected later).
    fast.save_pretrained(DST)
    print(f"saved tokenizer to {DST}")

    # 7) Copy chat_template.jinja from the 1930 base (same chat layout).
    if os.path.exists(os.path.join(REF_DIR, "chat_template.jinja")):
        shutil.copy(os.path.join(REF_DIR, "chat_template.jinja"),
                    os.path.join(DST, "chat_template.jinja"))
        print("copied chat_template.jinja from 1930 base")

    print("DONE")


if __name__ == "__main__":
    main()
