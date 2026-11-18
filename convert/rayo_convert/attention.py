"""Recognising scaled dot-product self-attention in a source graph.

A whole-sequence graph (TFLite or ONNX) holds attention as ops on [.., heads, T, head_size]
tensors: the query, key and value projections reshaped and transposed into heads, the scores
``q · kᵀ`` (BATCH_MATMUL), a scale and a mask, SOFTMAX, the weighted sum of the values
(BATCH_MATMUL) and the heads put back side by side. This module finds such a block and checks
it is exactly what rayo's ``attention`` op computes (docs/CONVERTING.md lists the conditions):

- the scores are one BATCH_MATMUL of a query and a key tensor, both laid out from projections
  [.., T, F] by reshapes, transposes, slices, splits and scalings only;
- between the scores and SOFTMAX only constant scalings and constant masks;
- the mask is causal (0 where key j <= query i, at most -1e4 elsewhere) or absent;
- after SOFTMAX at most a multiplication by that same pattern of ones, then one BATCH_MATMUL
  with the value tensor;
- the heads come back as [.., T, heads * head_size] by reshapes and transposes;
- query, key and value have the same heads and head size, and the scale is positive.

Instead of matching op patterns name by name, it traces indices: it feeds the projections' row
and column numbers through the layout ops (with the reference interpreter) and reads, for every
head and dimension, which projection column it is and whether every query row is its own
position. Whatever the layout (one fused qkv projection split three ways, Keras'
MultiHeadAttention's separate projections, heads first or last), the result is the projection
columns of each head, which become rayo's [T, D] q, k and v.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import ref
from .ir import Graph, Op
from .passes import const_value, uniform

LAYOUT = {"RESHAPE", "TRANSPOSE", "STRIDED_SLICE", "SLICE", "SQUEEZE", "EXPAND_DIMS", "SPLIT",
          "SPLIT_V", "GATHER", "ONNX_RESHAPE", "ONNX_SLICE", "ONNX_SQUEEZE", "ONNX_UNSQUEEZE",
          "ONNX_SPLIT", "ONNX_FLATTEN", "IDENTITY"}
TAIL = {"TRANSPOSE", "RESHAPE", "SQUEEZE", "EXPAND_DIMS", "ONNX_RESHAPE", "ONNX_SQUEEZE",
        "ONNX_UNSQUEEZE", "ONNX_FLATTEN", "IDENTITY"}
MASK_MIN = -1.0e4  # a hidden key's score: exp(-1e4 + anything sane) is 0 in float32


@dataclass
class Block:
    softmax: Op
    ops: list[Op]  # every op of the block, to be replaced
    sources: dict[str, int]  # q, k, v: the projection tensors [.., T, F]
    cols: dict[str, list[int]]  # their columns, head by head
    heads: int
    head_size: int
    scale: float
    causal: bool
    output: int  # the [.., T, D] tensor the block gives
    out_perm: list[int] | None = None  # output column c is rayo's column out_perm[c]
    notes: list[str] = field(default_factory=list)


class Declined(Exception):
    """Not an attention block rayo computes exactly; the message says why."""


def find(g: Graph, sh: dict[int, tuple[int, ...]]) -> tuple[list[Block], list[str]]:
    """The attention blocks of ``g``, and why each SOFTMAX that looked like one was declined."""
    blocks, declined = [], []
    for o in g.ops:
        if o.kind != "SOFTMAX":
            continue
        prod = g.producer()
        p = prod.get(o.inputs[0])
        if p is None or p.kind not in ("ADD", "MUL", "DIV", "SELECT_V2", "SELECT",
                                       "BATCH_MATMUL"):
            continue
        try:
            blocks.append(_match(g, sh, o))
        except Declined as e:
            declined.append(f"{g.describe(o)}: attention not rewritten: {e}")
    return blocks, declined


def _users(g: Graph, t: int) -> list[Op]:
    return [o for o in g.ops if t in o.inputs]


def _one_user(g: Graph, t: int, why: str) -> Op:
    users = _users(g, t)
    if len(users) != 1 or t in g.outputs:
        raise Declined(f"{why} is read by {len(users)} ops")
    return users[0]


def _match(g: Graph, sh: dict[int, tuple[int, ...]], sm: Op) -> Block:
    prod = g.producer()
    ops: list[Op] = [sm]
    beta = float(sm.options.get("beta", 1.0))
    scale = beta
    # Back from SOFTMAX to the scores' BATCH_MATMUL: scalings and masks.
    masks: list[tuple[np.ndarray, str, float]] = []  # (value, kind, scale applied after it)
    t = sm.inputs[0]
    while True:
        p = prod.get(t)
        if p is None:
            raise Declined("the scores do not come from a BATCH_MATMUL")
        if p is not sm and _users(g, t) != [ops[-1]]:
            raise Declined(f"{g.tensors[t].name} is read outside the attention")
        if p.kind == "BATCH_MATMUL":
            scores_op = p
            ops.append(p)
            break
        if p.kind in ("MUL", "DIV"):
            a, b = p.inputs
            ca, cb = const_value(g, a), const_value(g, b)
            c, x = (cb, a) if cb is not None else (ca, b)
            if c is None or (p.kind == "DIV" and cb is None):
                raise Declined(f"{g.describe(p)} scales the scores by a non-constant")
            u = uniform(c)
            if u is None:
                raise Declined(f"{g.describe(p)} scales the scores by a constant that varies")
            if int(p.options.get("fused_activation_function", 0)):
                raise Declined(f"{g.describe(p)} has a fused activation")
            f = u if p.kind == "MUL" else 1.0 / u
            scale *= f
            masks = [(m, k, s * f) for m, k, s in masks]
            ops.append(p)
            t = x
            continue
        if p.kind == "ADD":
            a, b = p.inputs
            ca, cb = const_value(g, a), const_value(g, b)
            if ca is None and cb is None:
                raise Declined(f"{g.describe(p)} adds a mask that is not a constant (a padding "
                               "mask from an input, say)")
            c, x = (cb, a) if cb is not None else (ca, b)
            if int(p.options.get("fused_activation_function", 0)):
                raise Declined(f"{g.describe(p)} has a fused activation")
            masks.append((np.asarray(c, np.float64), "add", 1.0))
            ops.append(p)
            t = x
            continue
        if p.kind in ("SELECT", "SELECT_V2"):
            cond, x, fill = p.inputs
            cc, cf = const_value(g, cond), const_value(g, fill)
            if cc is None or cf is None or uniform(cf) is None:
                raise Declined(f"{g.describe(p)} masks the scores with a non-constant")
            m = np.where(np.asarray(cc, bool), 0.0, float(uniform(cf)))
            masks.append((m, "select", 1.0))
            ops.append(p)
            t = x
            continue
        raise Declined(f"{g.describe(p)} ({p.kind}) between the scores and the softmax")
    # Forward from SOFTMAX: an optional MUL by the mask's ones, then BATCH_MATMUL with v.
    after = _one_user(g, sm.outputs[0], "the softmax")
    weights_t = sm.outputs[0]
    post = None
    if after.kind == "MUL":
        c = const_value(g, _other(after, sm.outputs[0]))
        if c is None:
            raise Declined(f"{g.describe(after)} multiplies the weights by a non-constant")
        post = np.asarray(c, np.float64)
        ops.append(after)
        weights_t = after.outputs[0]
        after = _one_user(g, after.outputs[0], "the masked weights")
    if after.kind != "BATCH_MATMUL" or after.inputs[0] != weights_t or \
            after.options.get("adj_x"):
        raise Declined(f"the weights go to {g.describe(after)}, not a weighted sum of values")
    sum_op = after
    ops.append(sum_op)
    # The heads back side by side.
    y = sum_op.outputs[0]
    tail: list[Op] = []
    while True:
        users = _users(g, y)
        if len(users) == 1 and y not in g.outputs and users[0].kind in TAIL:
            tail.append(users[0])
            y = users[0].outputs[0]
            continue
        break
    ops += tail
    # q, k and v, traced to their projections.
    q_t, k_t = scores_op.inputs
    v_t = sum_op.inputs[1]
    qi = _trace(g, sh, q_t, ops)
    ki = _trace(g, sh, k_t, ops)
    vi = _trace(g, sh, v_t, ops)
    scale *= qi.scale * ki.scale
    if vi.scale != 1.0:
        raise Declined("the values are scaled before the weighted sum")
    # Matrix views: q [b.., M, K], k [b.., N, K], v [b.., N, K].
    qr, qc = qi.rows, qi.cols
    if scores_op.options.get("adj_x"):
        qr, qc = np.swapaxes(qr, -1, -2), np.swapaxes(qc, -1, -2)
    kr, kc = ki.rows, ki.cols
    if not scores_op.options.get("adj_y"):
        kr, kc = np.swapaxes(kr, -1, -2), np.swapaxes(kc, -1, -2)
    vr, vc = vi.rows, vi.cols
    if sum_op.options.get("adj_y"):
        vr, vc = np.swapaxes(vr, -1, -2), np.swapaxes(vc, -1, -2)
    heads, q_cols, m_len, d = _heads(qr, qc, "query")
    hk, k_cols, n_len, dk = _heads(kr, kc, "key")
    hv, v_cols, nv, dv = _heads(vr, vc, "value")
    if not (heads == hk == hv):
        raise Declined(f"heads differ: query {heads}, key {hk}, value {hv}")
    if dk != d:
        raise Declined(f"head sizes differ: query {d}, key {dk}")
    if dv != d:
        raise Declined(f"the value head size {dv} is not the query's {d}; rayo's attention "
                       "gives v's size per head as q's")
    if nv != n_len:
        raise Declined(f"{n_len} keys but {nv} values")
    if not (scale > 0 and np.isfinite(scale)):
        raise Declined(f"the scale {scale} is not positive")
    # The mask: 0 where a key is seen, at most -1e4 where hidden.
    causal = _mask(masks, post, sh[scores_op.outputs[0]], heads, m_len, n_len)
    out_perm = _output(g, sh, sum_op, tail, y, heads, m_len, d)
    blk = Block(sm, ops, {"q": qi.source, "k": ki.source, "v": vi.source},
                {"q": q_cols, "k": k_cols, "v": v_cols}, heads, d, float(scale), causal, y,
                out_perm)
    inner = {t for o in ops for t in o.outputs} - {y}
    for o in ops:
        for t in o.outputs:
            if t in inner and (t in g.outputs or any(u not in ops for u in _users(g, t))):
                raise Declined(f"{g.tensors[t].name} is read outside the attention")
    return blk


def _other(o: Op, t: int) -> int:
    return o.inputs[1] if o.inputs[0] == t else o.inputs[0]


@dataclass
class Traced:
    source: int
    rows: np.ndarray
    cols: np.ndarray
    scale: float


def _trace(g: Graph, sh: dict[int, tuple[int, ...]], t: int, ops: list[Op]) -> Traced:
    """Back from ``t`` through layout ops to the projection it is laid out from, and each of
    its elements' row and column there."""
    prod = g.producer()
    chain: list[Op] = []
    scale = 1.0
    x = t
    while True:
        p = prod.get(x)
        if p is None:
            break
        if p.kind in ("MUL", "DIV"):
            a, b = p.inputs
            cb = const_value(g, b)
            ca = const_value(g, a)
            c, src = (cb, a) if cb is not None else (ca, b)
            u = uniform(c) if c is not None else None
            if u is None or (p.kind == "DIV" and cb is None):
                break
            scale *= u if p.kind == "MUL" else 1.0 / u
            chain.append(p)
            x = src
            continue
        if p.kind not in LAYOUT:
            break
        if p.kind == "GATHER" and const_value(g, p.inputs[1]) is None:
            break
        chain.append(p)
        x = p.inputs[0] if p.kind not in ("SPLIT",) else p.inputs[1]
    shape = sh[x]
    if len(shape) < 2 or any(d != 1 for d in shape[:-2]):
        raise Declined(f"{g.tensors[x].name} {list(shape)} is not a projection [.., T, F]")
    rows = np.broadcast_to(np.arange(shape[-2], dtype=np.float64)[:, None], shape).copy()
    cols = np.broadcast_to(np.arange(shape[-1], dtype=np.float64), shape).copy()
    for p in reversed(chain):
        if p.kind in ("MUL", "DIV"):
            continue
        out_index = p.outputs.index(next(o for o in p.outputs if _leads_to(g, o, t, chain)))
        rows = _through(g, p, rows, out_index)
        cols = _through(g, p, cols, out_index)
    for p in chain:
        if p not in ops:
            ops.append(p)
    return Traced(x, rows, cols, scale)


def _leads_to(g: Graph, start: int, target: int, chain: list[Op]) -> bool:
    if start == target:
        return True
    for p in chain:
        if start in p.inputs:
            return any(_leads_to(g, o, target, chain) for o in p.outputs)
    return False


def _through(g: Graph, p: Op, value: np.ndarray, out_index: int) -> np.ndarray:
    args: list[Any] = []
    data_pos = 1 if p.kind == "SPLIT" else 0
    for j, i in enumerate(p.inputs):
        if j == data_pos:
            args.append(value)
        else:
            args.append(g.tensors[i].data if i >= 0 else None)
    return ref.evaluate(g, p, args)[out_index]


def _heads(rows: np.ndarray, cols: np.ndarray, what: str) -> tuple[int, list[int], int, int]:
    """For a matrix view [b.., T, K]: the heads, the projection column of each (head, k), the
    sequence length and the head size. Every row t must be position t."""
    shape = rows.shape
    batch = [d for d in shape[:-2] if d != 1]
    if len(batch) > 1:
        raise Declined(f"the {what} has more than one batch of heads {list(shape)}")
    heads = batch[0] if batch else 1
    r = rows.reshape(heads, shape[-2], shape[-1])
    c = cols.reshape(heads, shape[-2], shape[-1])
    t, k = shape[-2], shape[-1]
    if not np.array_equal(r, np.broadcast_to(np.arange(t)[None, :, None], r.shape)):
        raise Declined(f"the {what}'s rows are not the positions in order")
    if not np.all(c == c[:, :1, :]):
        raise Declined(f"the {what}'s columns change from position to position")
    cl = [int(v) for v in c[:, 0, :].reshape(-1)]
    if len(set(cl)) != len(cl):
        raise Declined(f"the {what} repeats projection columns")
    return heads, cl, t, k


def _mask(masks: list[tuple[np.ndarray, str, float]], post: np.ndarray | None,
          scores: tuple[int, ...], heads: int, m: int, n: int) -> bool:
    """Whether the masks are causal (True) or absent (False); Declined otherwise."""

    def fit(value: np.ndarray, what: str) -> np.ndarray:
        try:
            return np.broadcast_to(value, scores).reshape(heads, m, n)
        except ValueError as e:
            raise Declined(f"{what} of shape {list(value.shape)} does not fit the scores "
                           f"{list(scores)}") from e

    total = np.zeros((heads, m, n))
    for value, _kind, s in masks:
        total = total + fit(value, "a mask") * s
    seen = total == 0
    hidden = total <= MASK_MIN
    if not np.all(seen | hidden):
        bad = total[~(seen | hidden)]
        raise Declined(f"the mask holds {float(bad.reshape(-1)[0]):g}, neither 0 nor at most "
                       f"{MASK_MIN:g}: rayo's attention gives a hidden key no weight at all")
    if not np.all(seen == seen[:1]):
        raise Declined("the mask differs from head to head")
    tril = np.arange(n)[None, :] <= np.arange(m)[:, None] + (n - m)
    if post is not None:
        pv = fit(post, "the weights' mask")
        if not np.all((pv == 1) | (pv == 0)) or not np.all((pv == 1) == seen):
            raise Declined("the weights are multiplied by something other than the mask's ones")
    if np.all(seen):
        return False
    if np.array_equal(seen[0], tril):
        return True
    raise Declined("the mask is not causal (key j hidden from query i exactly when j > i)")


def _output(g: Graph, sh: dict[int, tuple[int, ...]], sum_op: Op, tail: list[Op], y: int,
            heads: int, m: int, d: int) -> list[int] | None:
    """How the weighted sums [.., heads, T, d] come back as [.., T, D]: the rayo column of each
    output column, None when they are rayo's order (head by head)."""
    shape = sh[sum_op.outputs[0]]
    hi = np.zeros(shape)
    ti = np.zeros(shape)
    ei = np.zeros(shape)
    view = (heads, m, d)
    if int(np.prod(shape)) != heads * m * d:
        raise Declined("the weighted sums are not [heads, T, head size]")
    hi[...] = np.broadcast_to(np.arange(heads)[:, None, None], view).reshape(shape)
    ti[...] = np.broadcast_to(np.arange(m)[None, :, None], view).reshape(shape)
    ei[...] = np.broadcast_to(np.arange(d)[None, None, :], view).reshape(shape)
    for p in tail:
        hi, ti, ei = (_through(g, p, a, 0) for a in (hi, ti, ei))
    ys = sh[y]
    if len(ys) < 2 or any(v != 1 for v in ys[:-2]) or ys[-2] != m or ys[-1] != heads * d:
        raise Declined(f"the heads come back as {list(ys)}, not [T, heads * head size]")
    hi, ti, ei = (a.reshape(m, heads * d) for a in (hi, ti, ei))
    if not np.array_equal(ti, np.broadcast_to(np.arange(m)[:, None], ti.shape)):
        raise Declined("the output rows are not the positions in order")
    col = (hi * d + ei)[0].astype(int)
    if not np.all((hi * d + ei) == col[None, :]) or len(set(col.tolist())) != heads * d:
        raise Declined("the output columns are not one per head and dimension")
    return None if np.array_equal(col, np.arange(heads * d)) else [int(c) for c in col]


def apply(g: Graph, sh: dict[int, tuple[int, ...]], blk: Block, layer_of: Any) -> None:
    """Replaces the block by one ``rayo:attention`` op on [.., T, D] q, k and v. A projection
    that is a dense layer is split into one dense per role, of the columns that role reads;
    otherwise its columns must already be the role's, in order."""
    prod = g.producer()
    d_all = blk.heads * blk.head_size
    plan: dict[str, tuple[str, Any]] = {}
    shared = {r: [r2 for r2 in "qkv" if blk.sources[r2] == blk.sources[r]] for r in "qkv"}
    for r in "qkv":
        src, cols = blk.sources[r], blk.cols[r]
        p = prod.get(src)
        if cols == list(range(sh[src][-1])):
            plan[r] = ("as_is", src)
        elif p is not None and p.kind == "rayo:dense":
            plan[r] = ("split", p)
        else:
            raise Declined(f"the {dict(q='query', k='key', v='value')[r]} is columns of "
                           f"{g.tensors[src].name}, which is not a dense layer's output")
    if blk.out_perm is not None:
        users = [o for o in g.ops if blk.output in o.inputs]
        if not users or any(u.kind != "rayo:dense" or u.inputs[0] != blk.output for u in users):
            raise Declined("the heads come back in another order and do not go to a dense layer")
    # Now change the graph.
    role = {"q": "query", "k": "key", "v": "value"}
    ins: list[int] = []
    new_ops: list[tuple[Op, Op]] = []
    for r in "qkv":
        how, x = plan[r]
        if how == "as_is":
            ins.append(x)
            continue
        p = x
        cols = blk.cols[r]
        k = g.tensors[p.options["kernel"]]
        kq = None
        if k.quant is not None and k.quant.scales.size > 1:
            if k.quant.axis != 0:
                raise Declined("the projection's kernel is quantised along its inputs")
            kq = type(k.quant)(k.quant.scales[cols], k.quant.zero_points[cols]
                               if k.quant.zero_points.size > 1 else k.quant.zero_points, 0)
        elif k.quant is not None:
            kq = k.quant
        layer = layer_of(k.name) or g.tensors[p.outputs[0]].name
        base = layer.rsplit(".", 1)[0] if len(shared[r]) > 1 and "." in layer else layer
        name = f"{base}.{role[r]}" if len(shared[r]) > 1 else layer
        g.hints.setdefault(blk.output, base)
        kid = g.add_tensor(k.name, k.dtype, (len(cols), k.shape[1]), k.data[cols].copy(), kq)
        g.hints[kid] = name
        bias = p.options.get("bias", -1)
        bid = -1
        if bias >= 0:
            bid = g.add_const(g.tensors[bias].name,
                              g.tensors[bias].data.reshape(-1)[cols].astype(np.float32))
            g.hints[bid] = name
        out_shape = (*sh[p.outputs[0]][:-1], len(cols))
        t = g.add_tensor(f"{name}:out", "float32", out_shape)
        sh[t] = out_shape
        g.hints[t] = name
        new = Op(p.index, "rayo:dense", list(p.inputs), [t],
                 {**p.options, "kernel": kid, "bias": bid})
        new_ops.append((p, new))
        ins.append(t)
    if blk.out_perm is not None:
        for u in [o for o in g.ops if blk.output in o.inputs]:
            k = g.tensors[u.options["kernel"]]
            data = np.empty_like(k.data)
            data[:, blk.out_perm] = k.data
            kid = g.add_tensor(k.name, k.dtype, k.shape, data, k.quant)
            u.options = {**u.options, "kernel": kid}
    after: dict[int, int] = {}
    for p, new in new_ops:
        at = g.ops.index(p) + 1 + after.get(id(p), 0)
        g.ops.insert(at, new)
        after[id(p)] = after.get(id(p), 0) + 1
    at = max(g.ops.index(o) for o in blk.ops if o in g.ops)
    att = Op(blk.softmax.index, "rayo:attention", ins, [blk.output],
             {"heads": blk.heads, "scale": blk.scale, "causal": blk.causal})
    g.ops.insert(at + 1, att)
    for o in blk.ops:
        if o in g.ops:
            g.ops.remove(o)
    del d_all
