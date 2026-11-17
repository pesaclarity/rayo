#!/usr/bin/env python3
"""Writes rayo's test models (tests/models/*.weights) and the outputs NumPy computes for them
(tests/models/expected.json). Random weights from a fixed seed: no trained model, no data.

``reader`` has every op the Ask reader uses, run as TFLite runs a dynamic-range-quantised
model (int8 kernels, inputs quantised on the fly). ``float`` has float32 kernels.

  python3 tool/make_test_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcwt import int8_per_channel, quantised, write  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "models"
T_MAX = 12


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def conv(x, k, b, size, dil, quant):
    """Same-padded conv1d; k is [out, 1, size, in] (real values)."""
    t = x.shape[0]
    x = x.astype(np.float64).copy()
    if quant:
        for p in range(dil):
            x[p::dil] = quantised(x[p::dil])
    pad = dil * (size - 1) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    y = sum(xp[j * dil: j * dil + t] @ k[:, 0, j, :].T.astype(np.float64) for j in range(size))
    return np.maximum(y + b, 0)


def dense(x, k, b, quant, relu):
    rows = np.atleast_2d(x).astype(np.float64)
    if quant:
        rows = np.stack([quantised(r) for r in rows])
    y = rows @ k.T.astype(np.float64) + b
    y = np.maximum(y, 0) if relu else y
    return y.reshape(*x.shape[:-1], -1)


def reader(rng: np.random.Generator) -> dict:
    def w(*shape, s=0.4):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    tok_real = w(20, 6)
    tok_q, tok_s = int8_per_channel(tok_real.reshape(-1, 1), 1)
    tok_q, tok_s = tok_q.reshape(20, 6), tok_s[:1]
    tok = tok_q.astype(np.float32) * tok_s[0]
    shp = w(5, 2)
    k1, k1s = int8_per_channel(w(8, 1, 3, 8))
    k2, k2s = int8_per_channel(w(8, 1, 3, 8))
    hk, hks = int8_per_channel(w(5, 16))
    ok = w(3, 5)
    ik, iks = int8_per_channel(w(4, 16))
    b1, b2, hb, ob, ib = w(8), w(8), w(5), w(3), w(4)
    tensors = {
        "tok": (tok_q, tok_s, None), "shp": shp,
        "c1.k": (k1, k1s, 0), "c1.b": b1, "c2.k": (k2, k2s, 0), "c2.b": b2,
        "h.k": (hk, hks, 0), "h.b": hb, "o.k": ok, "o.b": ob, "i.k": (ik, iks, 0), "i.b": ib,
    }
    nodes = [
        {"op": "gather", "inputs": ["tok", "tokens"], "output": "te"},
        {"op": "gather", "inputs": ["shp", "shapes"], "output": "se"},
        {"op": "concat", "inputs": ["te", "se"], "output": "emb"},
        {"op": "mask", "inputs": ["shapes"], "output": "m"},
        {"op": "conv1d", "inputs": ["emb", "c1.k", "c1.b"], "output": "c1", "size": 3,
         "dilation": 1, "activation": "relu", "input_quantization": "tensor_phases"},
        {"op": "conv1d", "inputs": ["c1", "c2.k", "c2.b"], "output": "c2", "size": 3,
         "dilation": 2, "activation": "relu"},
        {"op": "add", "inputs": ["c1", "c2"], "output": "x"},
        {"op": "concat", "inputs": ["x", "emb"], "output": "xe"},
        {"op": "dense", "inputs": ["xe", "h.k", "h.b"], "output": "h", "activation": "relu",
         "input_quantization": "rows"},
        {"op": "dense", "inputs": ["h", "o.k", "o.b"], "output": "ol"},
        {"op": "softmax", "inputs": ["ol"], "output": "slots"},
        {"op": "masked_max", "inputs": ["x", "m"], "output": "pmax", "fill": 10000},
        {"op": "masked_mean", "inputs": ["x", "m"], "output": "pmean"},
        {"op": "concat", "inputs": ["pmax", "pmean"], "output": "pooled"},
        {"op": "dense", "inputs": ["pooled", "i.k", "i.b"], "output": "il"},
        {"op": "softmax", "inputs": ["il"], "output": "intent"},
    ]
    graph = {"version": 1, "symbols": {"T": T_MAX},
             "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]},
                        {"name": "shapes", "dtype": "int32", "shape": ["T"]}],
             "nodes": nodes,
             "outputs": [{"name": "slots", "node": "slots"}, {"name": "intent", "node": "intent"}]}
    (OUT / "reader.weights").write_bytes(
        write("test-reader", tensors, graph, {"labels": ["a", "b", "c"], "count": 3}))
    real = lambda q, s: q.astype(np.float32) * s.reshape(-1, *([1] * (q.ndim - 1)))  # noqa: E731
    cases = []
    for tokens, shapes in (([3, 7, 1, 19, 0, 0], [1, 2, 4, 1, 0, 0]),
                           ([5] * T_MAX, [3] * T_MAX), ([0, 0], [0, 0]), ([4], [2])):
        t, s = np.array(tokens), np.array(shapes)
        m = (s > 0).astype(np.float64)[:, None]
        emb = np.concatenate([tok[t], shp[s]], axis=1).astype(np.float64)
        c1 = conv(emb, real(k1, k1s), b1, 3, 1, True)
        x = c1 + conv(c1, real(k2, k2s), b2, 3, 2, True)
        h = dense(np.concatenate([x, emb], axis=1), real(hk, hks), hb, True, True)
        slots = softmax(dense(h, ok, ob, False, False))
        n = max(m.sum(), 1.0)
        pooled = np.concatenate([(x * m - (1 - m) * 1e4).max(axis=0), (x * m).sum(axis=0) / n])
        intent = softmax(dense(pooled, real(ik, iks), ib, True, False))
        cases.append({"T": len(tokens), "tokens": tokens, "shapes": shapes,
                      "slots": slots.ravel().tolist(), "intent": intent.ravel().tolist()})
    return {"file": "reader.weights", "cases": cases}


def float_model(rng: np.random.Generator) -> dict:
    def w(*shape, s=0.4):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    emb, ck, cb, dk, db = w(9, 4), w(6, 1, 5, 4), w(6), w(2, 6), w(2)
    tensors = {"emb": emb, "c.k": ck, "c.b": cb, "d.k": dk, "d.b": db}
    nodes = [
        {"op": "gather", "inputs": ["emb", "ids"], "output": "e"},
        {"op": "conv1d", "inputs": ["e", "c.k", "c.b"], "output": "c", "size": 5, "dilation": 1,
         "activation": "relu"},
        {"op": "dense", "inputs": ["c", "d.k", "d.b"], "output": "y"},
    ]
    graph = {"version": 1, "symbols": {"N": 8},
             "inputs": [{"name": "ids", "dtype": "int32", "shape": ["N"]}], "nodes": nodes,
             "outputs": [{"name": "y", "node": "y"}]}
    (OUT / "float.weights").write_bytes(write("test-float", tensors, graph))
    ids = [1, 8, 2, 2, 0, 5, 7]
    c = conv(emb[ids], ck, cb, 5, 1, False)
    y = dense(c, dk, db, False, False)
    return {"file": "float.weights", "cases": [{"N": len(ids), "ids": ids, "y": y.ravel().tolist()}]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117)
    expected = {"reader": reader(rng), "float": float_model(rng)}
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
