#!/usr/bin/env python3
"""Rebuild a complete tokenizer with Semantic-ID (SID) tokens.

Why this exists
---------------
`transformers.Trainer` does not save the tokenizer into every intermediate
checkpoint, so a mid-training `checkpoint-*/` folder ends up with model
weights whose embedding matrix already contains the added SID-token rows
(`<a_*>`, `<b_*>`, `<c_*>`, optional `<d_*>`), but with a *base* tokenizer
that no longer knows about those tokens. Loading such a checkpoint naively
gives `len(tokenizer) == base_vocab` while the model has `base_vocab + N`
embedding rows, and the SID tokens decode/encode incorrectly.

This script reconstructs the SID tokens **deterministically** and in the
exact order training used, so the recovered token ids line up 1:1 with the
trained embedding rows.

It mirrors `sft.py`'s logic exactly:

    new_tokens = set()
    for index in indices.values():
        for token in index:
            new_tokens.add(token)
    new_tokens = sorted(list(new_tokens))   # <-- deterministic, lexicographic
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

Because the order is `sorted(set(...))`, token id
`base_vocab + k` always maps to the k-th SID token in lexicographic order,
which is identical to what the freeze_LLM SFT run produced. Reproducing the
same call therefore restores the correct token<->embedding alignment.

Usage
-----
    python tools/rebuild_sid_tokenizer.py \
        --model_dir   /path/to/checkpoint-or-modelscope-dir \
        --index_path  /path/to/Games.index.json \
        --out_dir     /path/to/checkpoint-or-modelscope-dir   # in-place is fine

After running, the directory will contain `added_tokens.json` /
`special_tokens_map.json` and an updated `tokenizer.json`, and
`len(tokenizer)` will match the model's embedding row count.
"""
import argparse
import json
import sys

from transformers import AutoTokenizer


def build_sid_tokens(index_path: str) -> list:
    """Return SID tokens in the same order `sft.py` adds them: sorted(set(...))."""
    with open(index_path, "r", encoding="utf-8") as f:
        indices = json.load(f)
    tokens = set()
    for code in indices.values():
        for tok in code:
            tokens.add(tok)
    return sorted(tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True,
                    help="Directory with the (incomplete) base tokenizer files.")
    ap.add_argument("--index_path", required=True,
                    help="Path to <Dataset>.index.json (item -> SID mapping).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save the fixed tokenizer (default: --model_dir, in place).")
    ap.add_argument("--expected_vocab", type=int, default=None,
                    help="Optional: assert the final tokenizer length equals this "
                         "(e.g. the model's embedding row count, 152460 for the Games SFT ckpt).")
    args = ap.parse_args()

    out_dir = args.out_dir or args.model_dir

    new_tokens = build_sid_tokens(args.index_path)
    print(f"[info] {len(new_tokens)} unique SID tokens from {args.index_path}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base_len = len(tok)
    added = tok.add_tokens(new_tokens)
    print(f"[info] base vocab = {base_len}, added = {added}, new len = {len(tok)}")

    # Verify the added ids are contiguous and in sorted order (== training order).
    ids = [tok.convert_tokens_to_ids(t) for t in new_tokens]
    expected_ids = list(range(base_len, base_len + len(new_tokens)))
    if ids != expected_ids:
        print("[error] added-token ids are not contiguous / not in sorted order; "
              "alignment with trained embeddings would be broken.", file=sys.stderr)
        return 1

    if args.expected_vocab is not None and len(tok) != args.expected_vocab:
        print(f"[error] final vocab {len(tok)} != expected {args.expected_vocab}; "
              "the index file may not match this checkpoint.", file=sys.stderr)
        return 1

    tok.save_pretrained(out_dir)
    print(f"[ok] saved complete tokenizer to {out_dir}")
    print(f"[ok] id range for SID tokens: {ids[0]}..{ids[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
