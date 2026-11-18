#!/usr/bin/env python3
"""Writes a rayo model with the shapes of a published small language model and random
weights, for measuring speed and memory where the real weights cannot be downloaded. The
shapes are the models' config.json values; the weights are synthetic, so the text it writes
means nothing, but every matrix product, the cache and the memory are the real model's size.

  python3 tool/make_synthetic_slm.py smollm2-135m q4 out.weights [--context 2048]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("weights", choices=["float32", "int8", "q4"])
    ap.add_argument("out")
    ap.add_argument("--context", type=int, default=2048)
    a = ap.parse_args()
    cfg = dict(SHAPES[a.shape])
    st = Synthetic(cfg)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(cfg))
        hf.open_safetensors = lambda _d: st  # the synthetic tensors instead of files
        header, tensors, notes = hf.build(Path(d), hf.HfOptions(name=f"synthetic-{a.shape}",
                                                                weights=a.weights,
                                                                context=a.context))
    size = hf.write_file(Path(a.out), header, tensors)
    print(f"wrote {a.out}: {size / 1e6:.1f} MB; " + "; ".join(notes))


if __name__ == "__main__":
    main()
