"""int8 weights from a float graph (``--weights int8``): rayo's dynamic-range scheme, applied
by the converter, so the scheme lives in one place.

The weights are quantised here; the activations are quantised on the fly by rayo's kernels, as
for a TFLite dynamic-range model, so the result is the int8 file TFLite's converter would have
written and runs on the same kernels (and the app's Dart reader):

- **dense** (``FULLY_CONNECTED``; ONNX ``Gemm`` and ``MatMul`` with a constant weight): the
  kernel ``[out, in]`` symmetric int8, one scale per output (axis 0); each row of the input
  quantised asymmetrically when the layer runs (``input_quantization: rows``).
- **conv1d** (``CONV_2D`` over one row; ONNX's 1-D ``Conv``): the kernel ``[out, 1, size, in]``
  symmetric int8, one scale per output (axis 0); the input quantised as one tensor, a dilated
  convolution's each phase alone (``input_quantization: tensor_phases``).
- **embedding** (``GATHER`` along axis 0 of a constant ``[rows, width]`` table): the table
  symmetric int8 with one scale, each row dequantised as it is read.

Symmetric: zero point 0, ``scale = max |w| / 127`` over what shares a scale (1 where every
value is 0), ``q = round(w / scale)`` with halves away from zero, so the largest magnitude is
±127. Tensors with fewer than ``min_size`` values stay float32, as TFLite's converter keeps
them (its threshold, 1024, is the default). The reference (``rayo_convert.ref``) gives every
quantised op this meaning, so a converted file is checked against it. NumPy alone, no state:
the same graph gives the same bytes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ir import ConvertError, Graph, Op, Quant
from .ref import round_half_away

CLASSES = ("dense", "conv1d", "embedding")
MIN_SIZE = 1024  # TFLite's: weights with fewer values stay float


@dataclass
class Int8:
    classes: tuple[str, ...] = CLASSES  # which tensor classes are quantised
    min_size: int = MIN_SIZE


def symmetric(w: np.ndarray, axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    """``w`` as symmetric int8 and its float32 scales: one per index along ``axis``, or one in
    all (``axis`` None)."""
    w = np.asarray(w, np.float32)
    if axis is None:
        top = np.asarray([np.abs(w).max() if w.size else 0.0], np.float32)
        shape = [1] * w.ndim
    else:
        moved = np.moveaxis(np.abs(w), axis, 0).reshape(w.shape[axis], -1)
        top = moved.max(axis=1).astype(np.float32)
        shape = [1] * w.ndim
        shape[axis] = -1
    scales = (top / np.float32(127.0)).astype(np.float32)
    scales[scales == 0] = np.float32(1.0)
    q = round_half_away(w.astype(np.float64) / scales.reshape(shape).astype(np.float64))
    return np.clip(q, -127, 127).astype(np.int8), scales


def parse_classes(text: str) -> tuple[str, ...]:
    """``dense,conv1d,embedding`` (or ``all``, or ``none``) to the classes."""
    words = [w.strip() for w in text.split(",") if w.strip()]
    if words == ["all"]:
        return CLASSES
    if words == ["none"]:
        return ()
    bad = [w for w in words if w not in CLASSES]
    if bad:
        raise ConvertError(f"--int8-tensors: {', '.join(bad)}: the classes are "
                           f"{', '.join(CLASSES)} (or all, none)")
    return tuple(c for c in CLASSES if c in words)


def _slot(o: Op) -> tuple[str, int] | None:
    """The tensor class an op's weight belongs to, and the input it is."""
    if o.kind == "FULLY_CONNECTED":
        return "dense", 1
    if o.kind == "CONV_2D":
        return "conv1d", 1
    if o.kind == "GATHER" and int(o.options.get("axis", 0)) == 0 and \
            not int(o.options.get("batch_dims", 0)):
        return "embedding", 0
    if o.kind == "EMBEDDING_LOOKUP":
        return "embedding", 1
    return None


def apply(g: Graph, opts: Int8) -> list[str]:
    """Quantises ``g``'s float32 weights in place; the notes say what was and was not."""
    users = g.consumers()
    notes: list[str] = []
    done: Counter[str] = Counter()
    kept_small: list[str] = []
    seen: set[int] = set()
    for o in list(g.ops):
        slot = _slot(o)
        if slot is None:
            continue
        cls, at = slot
        if at >= len(o.inputs) or o.inputs[at] < 0:
            continue
        t = g.tensors[o.inputs[at]]
        if t.index in seen or not t.const or t.dtype != "float32":
            continue
        seen.add(t.index)
        want = {2: ("dense", "embedding"), 4: ("conv1d",)}.get(t.data.ndim, ())
        if cls not in want or cls not in opts.classes:
            continue
        others = [u for u in users.get(t.index, []) if u is not o]
        if any(_slot(u) != slot or u.inputs[at] != t.index for u in others):
            name = g.hints.get(t.index, t.name)
            notes.append(f"int8: {name} stays float32: it is read by more than "
                         f"{'an embedding lookup' if cls == 'embedding' else 'a ' + cls}")
            continue
        if t.data.size < opts.min_size:
            kept_small.append(f"{g.hints[t.index]} ({cls})" if t.index in g.hints else t.name)
            continue
        q, scales = symmetric(t.data, None if cls == "embedding" else 0)
        t.data, t.dtype = q, "int8"
        t.quant = Quant(scales, np.zeros(scales.size, np.int64),
                        None if cls == "embedding" else 0)
        done[cls] += 1
        for u in [o, *others]:
            _hybrid(g, u, cls)
    if done:
        notes.append("int8 weights: " + ", ".join(f"{n} {c}" for c, n in done.items() if n)
                     + " (symmetric; the activations are quantised as the layers run)")
    if kept_small:
        notes.append(f"int8: {len(kept_small)} weights with fewer than {opts.min_size} values "
                     f"stay float32 ({', '.join(kept_small)})")
    return notes


def _hybrid(g: Graph, o: Op, cls: str) -> None:
    """``o``'s meaning with its weight now int8, as rayo's kernel computes it."""
    if cls == "dense":
        o.options["asymmetric_quantize_inputs"] = True  # each input row, asymmetric
    elif cls == "conv1d":
        if int(o.options.get("dilation_w_factor", 1)) > 1:
            o.options["input_phases"] = True  # each dilation phase alone
    else:  # the lookup gives int8 rows; a DEQUANTIZE (folded into rayo's gather) reads them
        table = g.tensors[o.inputs[0 if o.kind == "GATHER" else 1]]
        out = g.tensors[o.outputs[0]]
        q = g.add_tensor(out.name + "/int8", "int8", out.shape, quant=table.quant)
        if out.index in g.hints or table.index in g.hints:  # named as the lookup was
            g.hints[q] = g.hints.get(out.index) or g.hints[table.index]
        deq = Op(o.index, "DEQUANTIZE", [q], [out.index])
        o.outputs = [q]
        g.ops.insert(g.ops.index(o) + 1, deq)
