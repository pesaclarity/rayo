#!/usr/bin/env python3
"""Writes the test models in test/models and the outputs NumPy computes for them.

Random weights from a fixed seed: no trained model, no real data. Each model exercises a set
of ops; test/runtime_test.dart checks rayo's outputs against the recorded NumPy ones.

  python3 tool/make_test_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf import F16, F32, Q8_0, GgufFile  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "test" / "models"
rng = np.random.default_rng(20261117)


def w(*shape: int, scale: float = 0.5) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def model(name: str, graph: str) -> GgufFile:
    f = GgufFile()
    f.metadata.update(
        {
            "general.architecture": "rayo",
            "general.name": name,
            "rayo.format_version": 1,
            "rayo.graph": graph,
            "test.labels": ["a", "b"],
            "test.count": 3,
            "test.ratio": 0.5,
            "test.flag": True,
        }
    )
    return f


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    from math import erf

    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


OPS_GRAPH = """\
input ids i32 T
input x f32 8 T
e = get_rows emb ids
m = mask_pos ids
h = concat e x dim=0
c = conv1d h conv.w conv.b k=3 d=2 act=relu
d = dense c dense.w dense.b act=gelu_erf
r = add d e
n = layer_norm r ln.g ln.b eps=0.00001
q = dense n wide.w
pm = masked_max n m empty=-7
pa = masked_mean n m
sm = softmax n
last = cols n start=T-1 len=1
s = scale last s=2
output conv c
output norm n
output wide q
output max pm
output mean pa
output softmax sm
output last s
"""


def ops() -> dict:
    f = model("ops", OPS_GRAPH)
    emb, conv_w, conv_b = w(10, 8), w(8, 3 * 16), w(8)
    dense_w, dense_b, ln_g, ln_b, q8_w = w(8, 8), w(8), w(8) + 1, w(8), w(32, 8)
    f.add("emb", emb, F16)
    f.add("conv.w", conv_w, F32)
    f.add("conv.b", conv_b)
    f.add("dense.w", dense_w, F16)
    f.add("dense.b", dense_b)
    f.add("ln.g", ln_g)
    f.add("ln.b", ln_b)
    f.add("wide.w", q8_w, F32)
    emb = f.tensors["emb"].values()
    dense_w = f.tensors["dense.w"].values()

    cases = []
    for ids in ([3, 1, 4, 1, 5, 9, 2, 0, 0], [7], [0, 0, 0]):
        t = len(ids)
        x = w(t, 8)
        e = emb[ids]
        mask = np.array(ids) > 0
        h = np.concatenate([e, x], axis=1)
        k = conv_w.reshape(8, 3, 16)
        hp = np.pad(h, ((2, 2), (0, 0)))
        c = np.maximum(sum(hp[j * 2 : j * 2 + t] @ k[:, j, :].T for j in range(3)) + conv_b, 0)
        d = gelu_erf(c @ dense_w.T + dense_b)
        n = layer_norm(d + e, ln_g, ln_b)
        cases.append(
            {
                "ids": ids,
                "x": x.ravel().tolist(),
                "conv": c.ravel().tolist(),
                "norm": n.ravel().tolist(),
                "wide": (n @ q8_w.T).ravel().tolist(),
                "max": (n[mask].max(axis=0) if mask.any() else np.full(8, -7.0)).tolist(),
                "mean": (n[mask].mean(axis=0) if mask.any() else np.zeros(8)).tolist(),
                "softmax": softmax(n).ravel().tolist(),
                "last": (2 * n[-1]).tolist(),
            }
        )
    f.write(OUT / "ops.gguf")
    return {"cases": cases}


Q8_GRAPH = """\
input x f32 64 T
y = dense x w b
output y y
"""


def q8() -> dict:
    f = model("q8", Q8_GRAPH)
    f.add("w", w(16, 64), Q8_0)
    f.add("b", w(16))
    x = w(5, 64)
    y = x @ f.tensors["w"].values().T + f.tensors["b"].values()
    f.write(OUT / "q8.gguf")
    return {"x": x.ravel().tolist(), "y": y.ravel().tolist()}


DECODER_GRAPH = """\
# A one-layer causal transformer with a key/value cache: T new tokens at positions P..P+T-1.
input tokens i32 T
state kc f32 16 12
state vc f32 16 12
pos = range start=P len=T
e = get_rows emb tokens
pe = get_rows pos_emb pos
h = add e pe
hn = layer_norm h ln.g ln.b
q = dense hn wq bq
k = dense hn wk bk
v = dense hn wv bv
ks = write kc k at=P
vs = write vc v at=P
a = attention q ks vs heads=2 causal=1
o = dense a wo bo
y = add h o
logits = dense y wout
output logits logits
"""


def decoder() -> dict:
    f = model("decoder", DECODER_GRAPH)
    p = {
        "emb": w(20, 16), "pos_emb": w(12, 16), "ln.g": w(16) + 1, "ln.b": w(16),
        "wq": w(16, 16), "bq": w(16), "wk": w(16, 16), "bk": w(16), "wv": w(16, 16),
        "bv": w(16), "wo": w(16, 16), "bo": w(16), "wout": w(20, 16),
    }
    for name, value in p.items():
        f.add(name, value)
    tokens = [5, 2, 17, 9, 0, 11]
    t = len(tokens)
    h = p["emb"][tokens] + p["pos_emb"][:t]
    hn = layer_norm(h, p["ln.g"], p["ln.b"])
    q, k, v = (hn @ p[a].T + p[b] for a, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
    heads, hd = 2, 8
    out = np.zeros_like(q)
    for i in range(heads):
        sl = slice(i * hd, (i + 1) * hd)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
        s = np.where(np.tril(np.ones((t, t))) > 0, s, -np.inf)
        out[:, sl] = softmax(s) @ v[:, sl]
    y = h + out @ p["wo"].T + p["bo"]
    logits = y @ p["wout"].T
    f.write(OUT / "decoder.gguf")
    return {"tokens": tokens, "logits": logits.ravel().tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {"ops": ops(), "q8": q8(), "decoder": decoder()}
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
