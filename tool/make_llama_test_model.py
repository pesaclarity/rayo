#!/usr/bin/env python3
"""Writes the tiny Llama-style language model rayo's C tests and the chat smoke test use
(tests/models/llama*.weights, llama.tokenizer.json) and what the NumPy reference computes for
it (tests/models/llama_expected.json): random weights from a fixed seed, a byte-level BPE
tokeniser trained here on a few synthetic sentences, and a ChatML-style chat template's
pieces. NumPy only.

  python3 tool/make_llama_test_model.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "convert"))
sys.path.insert(0, str(ROOT / "convert" / "tests"))
from llama_ref import forward, tiny_config, tiny_weights, write_model  # noqa: E402

from rayo_convert import hf  # noqa: E402

OUT = ROOT / "tests" / "models"
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CORPUS = ("the cat sat on the mat. the dog sat on the log. a cat and a dog are friends. "
          "hello there, how are you today? i am fine, thank you.")
TURNS = {"first": "<|im_start|>user\n", "user_end": "<|im_end|>\n<|im_start|>assistant\n",
         "next": "<|im_end|>\n<|im_start|>user\n", "system": "<|im_start|>system\n",
         "after_system": "<|im_end|>\n<|im_start|>user\n"}


def byte_chars() -> list[str]:
    keep = [*range(33, 127), *range(161, 173), *range(174, 256)]
    out, n = {}, 0
    for b in range(256):
        if b in keep:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + n)
            n += 1
    return [out[b] for b in range(256)]


def tokenizer(merges_wanted: int) -> dict:
    chars = byte_chars()
    vocab = {chars[b]: b for b in range(256)}
    words = [[chars[b] for b in (" " + w if i else w).encode()]
             for i, w in enumerate(CORPUS.split(" "))]
    merges = []
    for _ in range(merges_wanted):
        pairs = Counter(p for w in words for p in zip(w, w[1:]))
        if not pairs:
            break
        (a, b), _n = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append(f"{a} {b}")
        vocab[a + b] = len(vocab)
        for w in words:
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i: i + 2] = [a + b]
                i += 1
    added = []
    for s in SPECIALS:
        added.append({"id": len(vocab), "content": s, "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False, "special": True})
        vocab[s] = len(vocab)
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                              "use_regex": True},
            "post_processor": None, "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "fuse_unk": False,
                      "byte_fallback": False, "vocab": vocab, "merges": merges}}


def main() -> None:
    tok = tokenizer(40)
    vocab = len(tok["model"]["vocab"])
    end = tok["model"]["vocab"]["<|im_end|>"]
    cfg = tiny_config(vocab_size=vocab, max_position_embeddings=256, eos_token_id=end,
                      bos_token_id=tok["model"]["vocab"]["<|endoftext|>"],
                      intermediate_size=128)
    w = tiny_weights(cfg, seed=20261118)
    work = OUT / "llama-src"
    write_model(work, cfg, w, extra={"tokenizer.json": tok,
                                     "generation_config.json": {"eos_token_id": end}})
    meta = {"chat_turns": TURNS}
    expected: dict = {}
    for weights, name in (("float32", "llama.weights"), ("q4", "llama-q4.weights")):
        hf.convert_dir(work, OUT / name, hf.HfOptions(name="test-llama", weights=weights,
                                                      chunk=16, metadata=meta))
    (OUT / "llama-q4.tokenizer.json").unlink()
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    # A prompt of 20 ids (two chunks of 16), its last logits, and 8 greedy tokens after it.
    prompt = [int(x) for x in np.random.default_rng(7).integers(0, vocab - 3, 20)]
    ids = list(prompt)
    for _ in range(8):
        ids.append(int(forward(cfg, w, ids)[-1].argmax()))
    deq = {k: (hf.dequantize_int4(*hf.quantize_int4(v)) if v.ndim == 2 else v)
           for k, v in w.items()}
    expected = {"prompt": prompt, "greedy": ids[len(prompt):],
                "logits": [round(float(x), 6) for x in forward(cfg, w, prompt)[-1]],
                "logits_q4": [round(float(x), 6) for x in forward(cfg, deq, prompt)[-1]],
                "text": "the cat sat on the mat, hello there!", "eos": end}
    (OUT / "llama_expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {OUT}/llama*")


if __name__ == "__main__":
    main()
