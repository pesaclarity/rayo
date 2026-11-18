"""A NumPy interpreter of the source graph, with TFLite's meaning for each op.

It is the converter's reference: constant folding, the shapes of every tensor, the index
tracing that recognises attention, and the tests' expected outputs. int8 weights run as
TFLite's dynamic-range ("hybrid") kernels do: the float input quantised to int8 on the fly
(asymmetrically, per row for FULLY_CONNECTED, per batch item for CONV_2D, and per dilation
phase for a CONV_2D marked ``input_phases``, as rayo's dilated conv1d), the product exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .ir import ConvertError, Graph, Op

Fn = Callable[[Graph, Op, list[Any]], list[np.ndarray]]
OPS: dict[str, Fn] = {}


def op(*kinds: str) -> Callable[[Fn], Fn]:
    def reg(fn: Fn) -> Fn:
        for k in kinds:
            OPS[k] = fn
        return fn

    return reg


def round_half_away(x: Any) -> Any:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantised(x: np.ndarray) -> np.ndarray:
    """``x``'s int8 round trip as TFLite's AsymmetricQuantizeFloats: one scale and zero point
    for all of ``x``, the range widened to include 0, halves away from zero."""
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return x.astype(np.float64)
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if lo == hi or not (math.isfinite(lo) and math.isfinite(hi)):
        return np.zeros_like(x, dtype=np.float64)
    scale = (hi - lo) / 255.0
    zp_min, zp_max = -128 - lo / scale, 127 - hi / scale
    zp = zp_min if 128 + abs(lo / scale) < 127 + abs(hi / scale) else zp_max
    zero = -128 if zp <= -128 else 127 if zp >= 127 else int(round_half_away(zp))
    s = np.float32(scale)
    inv = np.float32(1.0) / s
    q = np.clip(round_half_away(np.float32(zero) + (x * np.float64(inv)).astype(np.float32)),
                -128, 127)
    return (q - zero) * np.float64(s)


def quantised_symmetric(x: np.ndarray) -> np.ndarray:
    """As TFLite's SymmetricQuantizeFloats."""
    x = np.asarray(x, np.float32)
    top = float(np.abs(x).max()) if x.size else 0.0
    if top == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    s = np.float32(top / 127.0)
    q = np.clip(round_half_away(x * (np.float32(1.0) / s)), -127, 127)
    return q * np.float64(s)


def activation(x: np.ndarray, code: Any) -> np.ndarray:
    """TFLite's fused activation (ActivationFunctionType)."""
    code = int(code or 0)
    if code == 0:
        return x
    if code == 1:
        return np.maximum(x, 0)
    if code == 2:
        return np.clip(x, -1, 1)
    if code == 3:
        return np.clip(x, 0, 6)
    if code == 4:
        return np.tanh(x)
    raise ConvertError(f"fused activation {code} is not supported")


def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, np.float32)


def _weight(g: Graph, op: Op, i: int, value: Any) -> np.ndarray:
    """A float view of input i: dequantised when it is an int8 constant."""
    t = g.tensors[op.inputs[i]]
    if t.dtype == "int8" and t.quant is not None:
        return t.real() if t.const else _dequant(t, value)
    return np.asarray(value)


def _dequant(t: Any, value: np.ndarray) -> np.ndarray:
    q = t.quant
    s, z = q.scales.astype(np.float32), q.zero_points.astype(np.float32)
    if q.axis is None or s.size == 1:
        return ((value.astype(np.float32) - z.reshape(-1)[0]) * s.reshape(-1)[0]).astype(
            np.float32)
    shape = [1] * value.ndim
    shape[q.axis] = -1
    return ((value.astype(np.float32) - z.reshape(shape)) * s.reshape(shape)).astype(np.float32)


def _hybrid(g: Graph, op: Op, i: int) -> bool:
    t = g.tensors[op.inputs[i]]
    return t.dtype == "int8" and t.const


_BINARY = {
    "ADD": np.add, "SUB": np.subtract, "MUL": np.multiply, "DIV": np.divide,
    "MAXIMUM": np.maximum, "MINIMUM": np.minimum, "POW": np.power,
    "SQUARED_DIFFERENCE": lambda a, b: (a - b) * (a - b),
    "FLOOR_DIV": np.floor_divide, "FLOOR_MOD": np.mod,
}


@op(*_BINARY)
def _binary(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x, y = np.asarray(a[0]), np.asarray(a[1])
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        x, y = _f32(x), _f32(y)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = _f32(_BINARY[o.kind](x, y))
        return [_f32(activation(r, o.options.get("fused_activation_function")))]
    if o.kind == "DIV":
        return [np.floor_divide(x, y).astype(x.dtype)]
    return [_BINARY[o.kind](x, y).astype(np.result_type(x, y))]


def _gelu(x: np.ndarray, approximate: bool) -> np.ndarray:
    x = _f32(x)
    if approximate:
        return _f32(0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3))))
    erf = np.frompyfunc(math.erf, 1, 1)
    return _f32(0.5 * x * (1 + erf(x.astype(np.float64) / math.sqrt(2)).astype(np.float64)))


_UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "NEG": np.negative, "ABS": np.abs, "SQUARE": np.square, "SQRT": np.sqrt,
    "RSQRT": lambda x: np.float32(1) / np.sqrt(x), "EXP": np.exp, "LOG": np.log,
    "TANH": np.tanh, "LOGISTIC": lambda x: 1 / (1 + np.exp(-x.astype(np.float64))),
    "RELU": lambda x: np.maximum(x, 0), "RELU6": lambda x: np.clip(x, 0, 6),
    "RELU_N1_TO_1": lambda x: np.clip(x, -1, 1), "FLOOR": np.floor, "CEIL": np.ceil,
    "SIN": np.sin, "COS": np.cos, "ROUND": np.round,
}


@op(*_UNARY)
def _unary(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    if x.dtype.kind != "f" and o.kind in ("NEG", "ABS", "SQUARE"):
        return [_UNARY[o.kind](x).astype(x.dtype)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return [_f32(_UNARY[o.kind](_f32(x)))]


@op("GELU")
def _gelu_op(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [_gelu(a[0], bool(o.options.get("approximate")))]


@op("SOFTMAX")
def _softmax(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = _f32(a[0]).astype(np.float64) * float(o.options.get("beta", 1.0))
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return [_f32(e / e.sum(axis=-1, keepdims=True))]


@op("LOG_SOFTMAX")
def _log_softmax(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = _f32(a[0]).astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return [_f32(x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))]


@op("FULLY_CONNECTED")
def _fc(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = _f32(a[0])
    w = _weight(g, o, 1, a[1]).astype(np.float64)
    n_in = w.shape[1]
    rows = x.reshape(-1, n_in)
    if _hybrid(g, o, 1):
        sym = not o.options.get("asymmetric_quantize_inputs", False)
        rows = np.stack([quantised_symmetric(r) if sym else quantised(r) for r in rows])
    y = rows.astype(np.float64) @ w.T
    if len(o.inputs) > 2 and o.inputs[2] >= 0 and a[2] is not None:
        y = y + _f32(a[2]).astype(np.float64)
    y = activation(_f32(y), o.options.get("fused_activation_function"))
    if o.options.get("keep_num_dims"):
        return [y.reshape(*x.shape[:-1], -1)]
    return [y]


def _pad_same(size: int, k: int, stride: int, dilation: int) -> tuple[int, int]:
    eff = (k - 1) * dilation + 1
    out = -(-size // stride)
    total = max((out - 1) * stride + eff - size, 0)
    return total // 2, total - total // 2


@op("CONV_2D")
def _conv2d(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = _f32(a[0])  # [N, H, W, C]
    w = _weight(g, o, 1, a[1]).astype(np.float64)  # [O, kh, kw, C]
    opt = o.options
    sh, sw = int(opt.get("stride_h", 1)), int(opt.get("stride_w", 1))
    dh, dw = int(opt.get("dilation_h_factor", 1)), int(opt.get("dilation_w_factor", 1))
    n, h, wd, c = x.shape
    oc, kh, kw, _ = w.shape
    if _hybrid(g, o, 1) and opt.get("input_phases") and dw > 1:
        # rayo's dilated conv1d (and TFLite's SPACE_TO_BATCH_ND form): each phase alone
        q = np.zeros(x.shape, np.float64)
        for b in range(n):
            for p in range(dw):
                q[b, :, p::dw, :] = quantised(x[b, :, p::dw, :])
        x = q
    elif _hybrid(g, o, 1):
        x = np.stack([quantised(b) for b in x]).astype(np.float64)
    else:
        x = x.astype(np.float64)
    if int(opt.get("padding", 0)) == 0:  # SAME
        ph, pw = _pad_same(h, kh, sh, dh), _pad_same(wd, kw, sw, dw)
    else:
        ph, pw = (0, 0), (0, 0)
    xp = np.pad(x, ((0, 0), ph, pw, (0, 0)))
    oh = (xp.shape[1] - ((kh - 1) * dh + 1)) // sh + 1
    ow = (xp.shape[2] - ((kw - 1) * dw + 1)) // sw + 1
    y = np.zeros((n, oh, ow, oc))
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, i * dh : i * dh + (oh - 1) * sh + 1 : sh,
                       j * dw : j * dw + (ow - 1) * sw + 1 : sw, :]
            y += patch @ w[:, i, j, :].T
    if len(o.inputs) > 2 and o.inputs[2] >= 0 and a[2] is not None:
        y = y + _f32(a[2])
    return [activation(_f32(y), opt.get("fused_activation_function"))]


@op("BATCH_MATMUL")
def _bmm(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x, y = _f32(_weight(g, o, 0, a[0])), _f32(_weight(g, o, 1, a[1]))
    if o.options.get("adj_x"):
        x = np.swapaxes(x, -1, -2)
    if o.options.get("adj_y"):
        y = np.swapaxes(y, -1, -2)
    return [_f32(np.matmul(x.astype(np.float64), y.astype(np.float64)))]


@op("GATHER")
def _gather(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    axis = int(o.options.get("axis", 0))
    if int(o.options.get("batch_dims", 0)):
        raise ConvertError(f"{g.describe(o)}: GATHER with batch_dims is not supported")
    params = np.asarray(a[0])
    idx = np.asarray(a[1]).astype(np.int64)
    n = params.shape[axis]
    idx = np.where(idx < 0, idx + n, idx)
    return [np.take(params, idx, axis=axis)]


@op("EMBEDDING_LOOKUP")
def _lookup(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.take(np.asarray(a[1]), np.asarray(a[0]).astype(np.int64), axis=0)]


@op("DEQUANTIZE")
def _dequantize(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    t = g.tensors[o.inputs[0]]
    if t.quant is None:
        return [_f32(a[0])]
    return [_dequant(t, np.asarray(a[0]))]


@op("CONCATENATION")
def _concat(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    y = np.concatenate([np.asarray(v) for v in a], axis=int(o.options.get("axis", 0)))
    if y.dtype.kind == "f":
        y = activation(y, o.options.get("fused_activation_function"))
    return [y]


@op("PACK")
def _pack(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.stack([np.asarray(v) for v in a], axis=int(o.options.get("axis", 0)))]


@op("UNPACK")
def _unpack(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    axis = int(o.options.get("axis", 0))
    x = np.asarray(a[0])
    return [np.take(x, i, axis=axis) for i in range(x.shape[axis])]


def _shape_from(g: Graph, o: Op, a: list[Any], x: np.ndarray) -> list[int]:
    if len(a) > 1 and a[1] is not None:
        shape = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    else:
        shape = [int(v) for v in o.options.get("new_shape", [])]
    return shape


@op("RESHAPE")
def _reshape(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    return [x.reshape(_shape_from(g, o, a, x))]


@op("EXPAND_DIMS")
def _expand(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.expand_dims(np.asarray(a[0]), int(np.asarray(a[1]).reshape(-1)[0]))]


@op("SQUEEZE")
def _squeeze(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    dims = o.options.get("squeeze_dims") or [i for i, d in enumerate(x.shape) if d == 1]
    return [np.squeeze(x, axis=tuple(int(d) % x.ndim for d in dims))]


@op("TRANSPOSE")
def _transpose(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.transpose(np.asarray(a[0]), [int(p) for p in np.asarray(a[1]).reshape(-1)])]


def strided_slice_spec(shape: tuple[int, ...], begin: list[int], end: list[int],
                       strides: list[int], opt: dict[str, Any]) -> tuple[list[slice], list[int]]:
    """TFLite's STRIDED_SLICE on ``shape``: the slices per axis and the axes shrunk."""
    if opt.get("ellipsis_mask") or opt.get("new_axis_mask"):
        raise ConvertError("STRIDED_SLICE with ellipsis or new axes is not supported")
    if opt.get("offset"):
        end = [b + e for b, e in zip(begin, end, strict=True)]
    bm, em, sm = int(opt.get("begin_mask", 0)), int(opt.get("end_mask", 0)), int(
        opt.get("shrink_axis_mask", 0))
    sl: list[slice] = []
    shrink: list[int] = []
    for i, n in enumerate(shape):
        if i >= len(begin):
            sl.append(slice(None))
            continue
        b, e, s = begin[i], end[i], strides[i]
        if sm >> i & 1:
            b = b + n if b < 0 else b
            sl.append(slice(b, b + 1, 1))
            shrink.append(i)
            continue
        sl.append(slice(None if bm >> i & 1 else b, None if em >> i & 1 else e, s))
    return sl, shrink


@op("STRIDED_SLICE")
def _strided_slice(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    vals = [[int(v) for v in np.asarray(a[i]).reshape(-1)] for i in (1, 2, 3)]
    sl, shrink = strided_slice_spec(x.shape, vals[0], vals[1], vals[2], o.options)
    y = x[tuple(sl)]
    return [np.squeeze(y, axis=tuple(shrink)) if shrink else y]


@op("SLICE")
def _slice(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    b = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    s = [int(v) for v in np.asarray(a[2]).reshape(-1)]
    sl = tuple(slice(bi, None if si == -1 else bi + si) for bi, si in zip(b, s, strict=True))
    return [x[sl]]


@op("SPLIT")
def _split(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    axis = int(np.asarray(a[0]).reshape(-1)[0])
    return list(np.split(np.asarray(a[1]), int(o.options.get("num_splits", len(o.outputs))),
                         axis=axis))


@op("SPLIT_V")
def _split_v(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    sizes = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    axis = int(np.asarray(a[2]).reshape(-1)[0])
    if -1 in sizes:
        sizes[sizes.index(-1)] = x.shape[axis] - (sum(sizes) + 1)
    return list(np.split(x, np.cumsum(sizes)[:-1], axis=axis))


_REDUCE = {"MEAN": np.mean, "SUM": np.sum, "REDUCE_MAX": np.max, "REDUCE_MIN": np.min,
           "REDUCE_PROD": np.prod, "REDUCE_ANY": np.any, "REDUCE_ALL": np.all}


@op(*_REDUCE)
def _reduce(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    axes = tuple(int(v) % max(x.ndim, 1) for v in np.asarray(a[1]).reshape(-1))
    keep = bool(o.options.get("keep_dims"))
    if x.dtype.kind == "f":
        return [_f32(_REDUCE[o.kind](x.astype(np.float64), axis=axes, keepdims=keep))]
    return [np.asarray(_REDUCE[o.kind](x, axis=axes, keepdims=keep)).astype(x.dtype)]


_COMPARE = {"GREATER": np.greater, "GREATER_EQUAL": np.greater_equal, "LESS": np.less,
            "LESS_EQUAL": np.less_equal, "EQUAL": np.equal, "NOT_EQUAL": np.not_equal}


@op(*_COMPARE)
def _compare(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [_COMPARE[o.kind](np.asarray(a[0]), np.asarray(a[1]))]


@op("LOGICAL_AND", "LOGICAL_OR")
def _logical(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    f = np.logical_and if o.kind == "LOGICAL_AND" else np.logical_or
    return [f(np.asarray(a[0]), np.asarray(a[1]))]


@op("LOGICAL_NOT")
def _not(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.logical_not(np.asarray(a[0]))]


@op("CAST")
def _cast(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.asarray(a[0]).astype(g.tensors[o.outputs[0]].dtype)]


@op("SELECT", "SELECT_V2")
def _select(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.where(np.asarray(a[0]), np.asarray(a[1]), np.asarray(a[2]))]


@op("SHAPE")
def _shape(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.asarray(np.asarray(a[0]).shape, dtype=g.tensors[o.outputs[0]].dtype)]


@op("RANK")
def _rank(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.asarray(np.asarray(a[0]).ndim, np.int32)]


@op("FILL")
def _fill(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.full([int(v) for v in np.asarray(a[0]).reshape(-1)], np.asarray(a[1]))]


@op("ZEROS_LIKE", "ONES_LIKE")
def _like(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    return [np.zeros_like(x) if o.kind == "ZEROS_LIKE" else np.ones_like(x)]


@op("RANGE")
def _range(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    start, limit, delta = (np.asarray(v).reshape(-1)[0] for v in a)
    return [np.arange(start, limit, delta).astype(g.tensors[o.outputs[0]].dtype)]


@op("TILE")
def _tile(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.tile(np.asarray(a[0]), [int(v) for v in np.asarray(a[1]).reshape(-1)])]


@op("BROADCAST_TO")
def _broadcast(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    shape = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    return [np.broadcast_to(np.asarray(a[0]), shape).copy()]


@op("PAD", "PADV2")
def _pad(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    value = np.asarray(a[2]).reshape(-1)[0] if len(a) > 2 and a[2] is not None else 0
    return [np.pad(np.asarray(a[0]), np.asarray(a[1]).astype(np.int64), constant_values=value)]


@op("SPACE_TO_BATCH_ND")
def _s2b(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    block = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    pads = np.asarray(a[2]).reshape(-1, 2)
    x = np.pad(x, [(0, 0), *[(int(p[0]), int(p[1])) for p in pads],
                   *[(0, 0)] * (x.ndim - 1 - len(block))])
    n = x.shape[0]
    spatial = x.shape[1 : 1 + len(block)]
    rest = x.shape[1 + len(block) :]
    shape = [n]
    for s, b in zip(spatial, block, strict=True):
        shape += [s // b, b]
    x = x.reshape(*shape, *rest)
    m = len(block)
    perm = [2 * i + 2 for i in range(m)] + [0] + [2 * i + 1 for i in range(m)]
    perm += list(range(1 + 2 * m, x.ndim))
    x = x.transpose(perm)
    return [x.reshape(n * int(np.prod(block)), *[s // b for s, b in zip(spatial, block,
                                                                        strict=True)], *rest)]


@op("BATCH_TO_SPACE_ND")
def _b2s(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    block = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    crops = np.asarray(a[2]).reshape(-1, 2)
    m = len(block)
    nb = int(np.prod(block))
    n = x.shape[0] // nb
    spatial = x.shape[1 : 1 + m]
    rest = x.shape[1 + m :]
    x = x.reshape(*block, n, *spatial, *rest)
    perm = [m]
    for i in range(m):
        perm += [m + 1 + i, i]
    perm += list(range(1 + 2 * m, x.ndim))
    x = x.transpose(perm).reshape(n, *[s * b for s, b in zip(spatial, block, strict=True)], *rest)
    sl = [slice(None)] + [slice(int(c[0]), x.shape[1 + i] - int(c[1])) for i, c in
                          enumerate(crops)]
    return [x[tuple(sl)]]


@op("ARG_MAX", "ARG_MIN")
def _argmax(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    f = np.argmax if o.kind == "ARG_MAX" else np.argmin
    axis = int(np.asarray(a[1]).reshape(-1)[0])
    return [f(np.asarray(a[0]), axis=axis).astype(g.tensors[o.outputs[0]].dtype)]


@op("LAYER_NORM")
def _layer_norm(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    """The converter's own fused op (from ONNX LayerNormalization, or fusion)."""
    x = _f32(a[0]).astype(np.float64)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + float(o.options.get("epsilon", 1e-5)))
    if len(a) > 1 and a[1] is not None:
        y = y * _f32(a[1])
    if len(a) > 2 and a[2] is not None:
        y = y + _f32(a[2])
    return [_f32(y)]


def run(g: Graph, feeds: dict[int, np.ndarray], keep: bool = True) -> dict[int, np.ndarray]:
    """Every tensor's value with graph inputs ``feeds`` (tensor index to array)."""
    values: dict[int, np.ndarray] = {}
    for t in g.tensors:
        if t.data is not None:
            values[t.index] = t.data
    values.update(feeds)
    for o in g.ops:
        fn = OPS.get(o.kind)
        if fn is None:
            raise ConvertError(f"{g.describe(o)}: {o.kind} is not supported")
        args = [values.get(i) if i >= 0 else None for i in o.inputs]
        outs = fn(g, o, args)
        for t, v in zip(o.outputs, outs, strict=False):
            values[t] = np.asarray(v)
    return values


def evaluate(g: Graph, o: Op, args: list[Any]) -> list[np.ndarray]:
    fn = OPS.get(o.kind)
    if fn is None:
        raise ConvertError(f"{g.describe(o)}: {o.kind} is not supported")
    return [np.asarray(v) for v in fn(g, o, args)]


# ONNX ops with no TFLite twin (rayo_convert.onnx_frontend).


@op("ONNX_ERF")
def _erf(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    erf = np.frompyfunc(math.erf, 1, 1)
    return [_f32(erf(_f32(a[0]).astype(np.float64)).astype(np.float64))]


@op("IDENTITY")
def _identity(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    return [np.asarray(a[0])]


@op("ONNX_RESHAPE")
def _onnx_reshape(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    shape = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    if not o.options.get("allowzero"):
        shape = [x.shape[i] if v == 0 else v for i, v in enumerate(shape)]
    return [x.reshape(shape)]


@op("ONNX_FLATTEN")
def _flatten(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    axis = int(o.options.get("axis", 1)) % max(x.ndim, 1)
    return [x.reshape(int(np.prod(x.shape[:axis])), -1)]


def _axes(o: Op, a: list[Any], rank: int) -> list[int]:
    if "axes" in o.options:
        axes = o.options["axes"]
    elif len(a) > 1 and a[1] is not None:
        axes = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    else:
        axes = []
    return [int(v) % rank for v in axes]


@op("ONNX_SQUEEZE")
def _onnx_squeeze(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    axes = _axes(o, a, x.ndim) or [i for i, d in enumerate(x.shape) if d == 1]
    return [np.squeeze(x, axis=tuple(axes))]


@op("ONNX_UNSQUEEZE")
def _onnx_unsqueeze(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    n = len(o.options["axes"]) if "axes" in o.options else np.asarray(a[1]).size
    for ax in sorted(_axes(o, a, x.ndim + n)):
        x = np.expand_dims(x, ax)
    return [x]


def onnx_slice_spec(shape: tuple[int, ...], a: list[Any]) -> list[slice]:
    starts = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    ends = [int(v) for v in np.asarray(a[2]).reshape(-1)]
    axes = ([int(v) % len(shape) for v in np.asarray(a[3]).reshape(-1)]
            if len(a) > 3 and a[3] is not None else list(range(len(starts))))
    steps = ([int(v) for v in np.asarray(a[4]).reshape(-1)] if len(a) > 4 and a[4] is not None
             else [1] * len(starts))
    spec = [slice(None)] * len(shape)
    for s, e, ax, st in zip(starts, ends, axes, steps, strict=True):
        n = shape[ax]
        e = max(min(e, n), -n - 1) if e < 0 else min(e, n)
        spec[ax] = slice(s, e, st)
    return spec


@op("ONNX_SLICE")
def _onnx_slice(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    return [x[tuple(onnx_slice_spec(x.shape, a))]]


def onnx_split_sizes(o: Op, a: list[Any], n: int) -> list[int]:
    if len(a) > 1 and a[1] is not None:
        return [int(v) for v in np.asarray(a[1]).reshape(-1)]
    if "split" in o.options:
        return list(o.options["split"])
    k = len(o.outputs)
    return [-(-n // k)] * (k - 1) + [n - (k - 1) * -(-n // k)]


@op("ONNX_SPLIT")
def _onnx_split(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    axis = int(o.options.get("axis", 0)) % x.ndim
    sizes = onnx_split_sizes(o, a, x.shape[axis])
    return list(np.split(x, np.cumsum(sizes)[:-1], axis=axis))


@op("ONNX_CONSTANT_OF_SHAPE")
def _cos(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    shape = [int(v) for v in np.asarray(a[0]).reshape(-1)]
    return [np.full(shape, o.options.get("value", 0),
                    dtype=g.tensors[o.outputs[0]].dtype)]


@op("ONNX_EXPAND")
def _expand(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    shape = [int(v) for v in np.asarray(a[1]).reshape(-1)]
    return [(x * np.ones(shape, dtype=x.dtype)).astype(x.dtype)]


@op("ONNX_TRILU")
def _trilu(g: Graph, o: Op, a: list[Any]) -> list[np.ndarray]:
    x = np.asarray(a[0])
    k = int(np.asarray(a[1]).reshape(-1)[0]) if len(a) > 1 and a[1] is not None else 0
    return [np.triu(x, k) if o.options.get("upper", 1) else np.tril(x, k)]
