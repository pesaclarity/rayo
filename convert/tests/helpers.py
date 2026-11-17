"""Running a source model with the reference and its conversion with rayo."""

from __future__ import annotations

import numpy as np

from rayo_convert import Options, convert, ref, tflite_frontend
from rayo_convert.runtime import Model, Session


def reference(raw: bytes, *inputs: np.ndarray) -> list[np.ndarray]:
    """The source model's outputs, from the NumPy reference (TFLite's meaning of each op)."""
    g = tflite_frontend.read(raw)
    v = ref.run(g, {i: np.asarray(x).reshape(g.tensors[i].shape).astype(g.tensors[i].dtype)
                    for i, x in zip(g.inputs, inputs, strict=True)})
    return [v[o] for o in g.outputs]


def session(lib, raw: bytes, file: str = "model.tflite", **opts) -> tuple[Session, object]:
    r = convert(raw, file, Options(**opts))
    return Session(Model(lib, r.data)), r


def decode(s: Session, tokens: list[int], prompt: int) -> list[np.ndarray]:
    """Logits after the prompt, then after each further token, one run each."""
    s.reset()
    s.bind("P", 0)
    s.bind("T", prompt)
    s.run([np.asarray(tokens[:prompt], np.int32)])
    out = [s.output(0)]
    for p in range(prompt, len(tokens)):
        s.bind("P", p)
        s.bind("T", 1)
        s.run([np.asarray(tokens[p : p + 1], np.int32)])
        out.append(s.output(0))
    return out
