#!/usr/bin/env python3
"""Writes a rayo model with the shapes of a published small language model and random
weights, for measuring speed and memory where the real weights cannot be downloaded. The
shapes are the models' config.json values; the weights are synthetic, so the text it writes
means nothing, but every matrix product, the cache and the memory are the real model's size.

  python3 tool/make_synthetic_slm.py smollm2-135m q4 out.weights [--context 2048]
      [--scales float16|float32] [--q3 ROLES] [--tokenizer]

With --tokenizer it also makes a synthetic byte-level BPE tokeniser of the model's vocabulary
size (random merges of the byte tokens, and ChatML's <|im_start|> and <|im_end|>, as SmolLM2
and Qwen2.5 use), writes its table into the file and its tokenizer.json next to it
(OUT.tokenizer.json), and gives the file ChatML's chat template, so the chat program runs on
it. Its ids mean nothing either; its size is the real tokeniser's.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "convert"))
from rayo_convert import hf  # noqa: E402

# config.json values of the models (Apache-2.0), shapes only.
SHAPES = {
    "smollm2-135m": {"num_hidden_layers": 30, "hidden_size": 576, "num_attention_heads": 9,
                     "num_key_value_heads": 3, "intermediate_size": 1536, "vocab_size": 49152,
                     "rope_theta": 100000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
                     "max_position_embeddings": 8192, "hidden_act": "silu"},
    "smollm2-360m": {"num_hidden_layers": 32, "hidden_size": 960, "num_attention_heads": 15,
                     "num_key_value_heads": 5, "intermediate_size": 2560, "vocab_size": 49152,
                     "rope_theta": 100000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
                     "max_position_embeddings": 8192, "hidden_act": "silu"},
    "qwen2.5-0.5b": {"num_hidden_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
                     "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
                     "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
                     "max_position_embeddings": 32768, "hidden_act": "silu", "bias": True},
}


class Synthetic:
    """Stands in for the safetensors: random tensors of the model's names and shapes, made
    one at a time when the writer asks."""

    def __init__(self, cfg: dict) -> None:
        H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
        hd, inter, V = D // H, cfg["intermediate_size"], cfg["vocab_size"]
        s = {"model.embed_tokens.weight": (V, D), "model.norm.weight": (D,)}
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}"
            s |= {f"{p}.self_attn.q_proj.weight": (H * hd, D),
                  f"{p}.self_attn.k_proj.weight": (KV * hd, D),
                  f"{p}.self_attn.v_proj.weight": (KV * hd, D),
                  f"{p}.self_attn.o_proj.weight": (D, H * hd),
                  f"{p}.mlp.gate_proj.weight": (inter, D), f"{p}.mlp.up_proj.weight": (inter, D),
                  f"{p}.mlp.down_proj.weight": (D, inter),
                  f"{p}.input_layernorm.weight": (D,),
                  f"{p}.post_attention_layernorm.weight": (D,)}
            if cfg.get("bias"):
                s |= {f"{p}.self_attn.q_proj.bias": (H * hd,),
                      f"{p}.self_attn.k_proj.bias": (KV * hd,),
                      f"{p}.self_attn.v_proj.bias": (KV * hd,)}
        self.shapes = s

    def __contains__(self, name: str) -> bool:
        return name in self.shapes

    def names(self) -> list[str]:
        return list(self.shapes)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get(self, name: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        shape = self.shapes[name]
        if len(shape) == 1:
            return (1 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
        return (0.02 * rng.standard_normal(shape)).astype(np.float32)


def synthetic_tokenizer(vocab: int, seed: int = 7) -> dict:
    """A byte-level BPE tokenizer.json of `vocab` tokens: the 256 byte tokens, merges of
    random pairs of earlier tokens (more often letters and words with a leading space, as
    real merges are), then the three special tokens."""
    rng = np.random.default_rng(seed)
    keep = [*range(33, 127), *range(161, 173), *range(174, 256)]
    chars, n = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + n)
            n += 1
    tokens = [chars[b] for b in range(256)]
    seen = set(tokens)
    letters = [chars[b] for b in range(97, 123)]
    space = chars[32]
    merges = []
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    while len(tokens) < vocab - len(specials):
        r = rng.random()
        if r < 0.15:
            a = space
        else:
            a = tokens[int(rng.integers(256, len(tokens)))] if len(tokens) > 300 and r < 0.8 \
                else letters[int(rng.integers(0, 26))]
        b = letters[int(rng.integers(0, 26))] if rng.random() < 0.7 else \
            tokens[int(rng.integers(256, len(tokens)))] if len(tokens) > 300 else \
            letters[int(rng.integers(0, 26))]
        if a.startswith(space) and b.startswith(space):
            continue
        t = a + b
        if t in seen or len(t) > 14:
            continue
        seen.add(t)
        tokens.append(t)
        merges.append(f"{a} {b}")
    vocab_map = {t: i for i, t in enumerate(tokens)}
    added = []
    for s_ in specials:
        added.append({"id": len(vocab_map), "content": s_, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False, "special": True})
        vocab_map[s_] = len(vocab_map)
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
            "normalizer": None,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                 "use_regex": True}]},
            "post_processor": None, "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "fuse_unk": False,
                      "byte_fallback": False, "vocab": vocab_map, "merges": merges}}


CHATML = {"first": "<|im_start|>user\n", "user_end": "<|im_end|>\n<|im_start|>assistant\n",
          "next": "<|im_end|>\n<|im_start|>user\n", "system": "<|im_start|>system\n",
          "after_system": "<|im_end|>\n<|im_start|>user\n"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("weights", choices=["float32", "int8", "q4"])
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--scales", choices=["float16", "float32"], default="float16")
    ap.add_argument("--q3", default="", help="as rayo-convert's --q3")
    ap.add_argument("--tokenizer", action="store_true",
                    help="a synthetic tokeniser of the vocabulary's size, in the file and next "
                         "to it, and ChatML's chat template")
    a = ap.parse_args()
    cfg = dict(SHAPES[a.shape])
    st = Synthetic(cfg)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(cfg))
        hf.open_safetensors = lambda _d: st  # the synthetic tensors instead of files
        header, tensors, notes = hf.build(Path(d), hf.HfOptions(
            name=f"synthetic-{a.shape}", weights=a.weights, context=a.context,
            scales=a.scales, q3=hf.q3_roles(a.q3)))
    table = None
    if a.tokenizer:
        from rayo_convert import tokenizer_table

        tok = synthetic_tokenizer(cfg["vocab_size"])
        end = tok["model"]["vocab"]["<|im_end|>"]
        header["metadata"].update({"chat_turns": CHATML, "eos_token_ids": [end],
                                   "add_bos_token": False})
        Path(a.out).with_suffix(".tokenizer.json").write_text(json.dumps(tok))
        table = tokenizer_table.table(tok)
        notes.append(f"synthetic tokeniser: {len(tok['model']['vocab'])} tokens, "
                     f"{len(tok['model']['merges'])} merges, table {len(table) / 1e6:.1f} MB")
    size = hf.write_file(Path(a.out), header, tensors, table)
    print(f"wrote {a.out}: {size / 1e6:.1f} MB; " + "; ".join(notes))


if __name__ == "__main__":
    main()
