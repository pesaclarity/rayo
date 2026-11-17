#!/usr/bin/env python3
"""Writes classifier.weights, a small model for "Using rayo with your own model" (README):
a bag-of-words classifier over token ids, with made-up weights from a fixed seed.

  ids [T] -> embedding -> mean over the real tokens (id > 0) -> dense, relu -> dense -> softmax

The first dense layer is int8 (dynamic-range quantised, run as TFLite runs it); the rest are
float32. It also writes expected.json: NumPy's probabilities for a few inputs, which
tool/ci/examples.sh checks the C example against.

  python3 examples/classifier/make_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tool"))
from pcwt import int8_per_channel, quantised, write  # noqa: E402

VOCAB, DIM, HIDDEN, LABELS, T_MAX = 50, 8, 16, ["greeting", "question", "thanks"], 32


def main() -> None:
    rng = np.random.default_rng(7)
    emb = (rng.standard_normal((VOCAB, DIM)) * 0.5).astype(np.float32)
    k1, s1 = int8_per_channel((rng.standard_normal((HIDDEN, DIM)) * 0.5).astype(np.float32))
    b1 = (rng.standard_normal(HIDDEN) * 0.1).astype(np.float32)
    k2 = (rng.standard_normal((len(LABELS), HIDDEN)) * 0.5).astype(np.float32)
    b2 = np.zeros(len(LABELS), np.float32)
    tensors = {"embedding": emb, "hidden.kernel": (k1, s1, 0), "hidden.bias": b1,
               "out.kernel": k2, "out.bias": b2}
    graph = {
        "version": 1,
        "symbols": {"T": T_MAX},
        "inputs": [{"name": "ids", "dtype": "int32", "shape": ["T"]}],
        "nodes": [
            {"op": "gather", "inputs": ["embedding", "ids"], "output": "vectors"},
            {"op": "compare", "inputs": ["ids"], "output": "real", "comparison": "greater",
             "value": 0},
            {"op": "masked_reduce", "inputs": ["vectors", "real"], "output": "mean",
             "reduction": "mean"},
            {"op": "dense", "inputs": ["mean", "hidden.kernel", "hidden.bias"],
             "output": "hidden", "activation": "relu", "input_quantization": "rows"},
            {"op": "dense", "inputs": ["hidden", "out.kernel", "out.bias"], "output": "logits"},
            {"op": "softmax", "inputs": ["logits"], "output": "probabilities"},
        ],
        "outputs": [{"name": "probabilities", "node": "probabilities"}],
    }
    (HERE / "classifier.weights").write_bytes(
        write("example-classifier", tensors, graph, {"labels": LABELS}))

    real_k1 = k1.astype(np.float64) * s1[:, None]
    cases = []
    for ids in ([4, 17, 9], [1, 2, 3, 0, 0], [42]):
        a = np.array(ids)
        keep = a > 0
        mean = emb[a[keep]].astype(np.float64).mean(axis=0) if keep.any() else np.zeros(DIM)
        h = np.maximum(quantised(mean.astype(np.float32)) @ real_k1.T + b1, 0)
        logits = h.astype(np.float32).astype(np.float64) @ k2.T.astype(np.float64) + b2
        e = np.exp(logits - logits.max())
        cases.append({"ids": ids, "probabilities": (e / e.sum()).tolist()})
    (HERE / "expected.json").write_text(json.dumps({"labels": LABELS, "cases": cases}, indent=1)
                                        + "\n")


if __name__ == "__main__":
    main()
