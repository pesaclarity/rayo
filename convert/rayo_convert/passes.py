"""Rewrites of the source graph before it is lowered to rayo's ops: constant folding, shapes,
ops that are no-ops in rayo, and fusions into rayo's larger ops (dense with its bias and
activation, 1-D convolutions, dilated convolutions, layer norms).

Fused ops get kinds of the form ``rayo:<op>`` with the rayo attributes in their options. Each
fusion checks the conditions under which it computes exactly what the source ops compute (as
rayo implements them, docs/OPS.md); where they do not hold the ops are left alone and lowered
one to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import ref
from .ir import ConvertError, Graph, Op

NO_FOLD = {"CUSTOM"}


@dataclass
class Notes:
    lines: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        if text not in self.lines:
            self.lines.append(text)


def rshape(shape: tuple[int, ...] | list[int]) -> list[int]:
    """The shape rayo holds for a source shape: without its leading ones (the batch), at least
    rank 1. Dropping leading ones changes no broadcast."""
    s = list(shape)
    while len(s) > 1 and s[0] == 1:
        s.pop(0)
    return s or [1]


def fold(g: Graph) -> None:
    """Evaluates every op whose inputs are all constants, once."""
    kept: list[Op] = []
    for o in g.ops:
        ins = [i for i in o.inputs if i >= 0]
        if (ins and all(g.tensors[i].const for i in ins) and o.kind in ref.OPS
                and not o.kind.startswith(tuple(NO_FOLD))):
            try:
                outs = ref.evaluate(g, o, [g.tensors[i].data if i >= 0 else None
                                           for i in o.inputs])
            except ConvertError:
                kept.append(o)
                continue
            for t, v in zip(o.outputs, outs, strict=False):
                tt = g.tensors[t]
                v = np.asarray(v)
                if tt.dtype == "int8" and tt.quant is not None and v.dtype == np.int8:
                    tt.data = v
                else:
                    tt.data = v.astype(tt.dtype if tt.dtype != "int8" else v.dtype)
                    if tt.dtype == "int8":
                        tt.dtype = tt.data.dtype.name
                        tt.quant = None
                tt.shape = tuple(v.shape)
            continue
        kept.append(o)
    g.ops = kept


def check_ops(g: Graph) -> None:
    """Every op the converter knows, or an error naming each one it does not."""
    from .lower import why_not

    bad = [f"{g.describe(o)}: {why_not(o.kind)}" for o in g.ops if o.kind not in ref.OPS]
    if bad:
        raise ConvertError("\n".join(bad))


def shapes(g: Graph, overrides: dict[int, tuple[int, ...]] | None = None
           ) -> dict[int, tuple[int, ...]]:
    """Every tensor's shape, from one run of the reference on zeros (ids 0)."""
    feeds = {}
    for i in g.inputs:
        t = g.tensors[i]
        shape = (overrides or {}).get(i, t.shape)
        feeds[i] = np.zeros(shape, dtype=t.dtype)
    values = ref.run(g, feeds)
    out = {t.index: tuple(t.shape) for t in g.tensors}
    for k, v in values.items():
        out[k] = tuple(np.asarray(v).shape)
    return out


def alias(g: Graph, old: int, new: int) -> None:
    """Every use of tensor ``old`` reads ``new`` instead."""
    for o in g.ops:
        o.inputs = [new if i == old else i for i in o.inputs]
    g.outputs = [new if i == old else i for i in g.outputs]
    if old in g.output_names and new not in g.output_names:
        g.output_names[new] = g.output_names[old]


def single_use(g: Graph, t: int) -> Op | None:
    """The one op reading ``t``, unless more read it or it is a graph output."""
    if t in g.outputs:
        return None
    users = [o for o in g.ops if t in o.inputs]
    return users[0] if len(users) == 1 else None


def const_value(g: Graph, t: int) -> np.ndarray | None:
    return g.tensors[t].data if t >= 0 and g.tensors[t].const else None


def uniform(v: np.ndarray | None) -> float | None:
    """A constant's one value, when every element is the same."""
    if v is None or v.size == 0 or v.dtype.kind not in "fiu":
        return None
    flat = v.astype(np.float64).reshape(-1)
    return float(flat[0]) if np.all(flat == flat[0]) else None


def simplify(g: Graph, sh: dict[int, tuple[int, ...]], notes: Notes) -> None:
    """Drops ops that do nothing on rayo's shapes: reshapes between shapes alike but for
    leading ones, casts from comparisons (rayo's are already float 0/1), and the
    negative-index wrap TFLite puts before an embedding."""
    changed = True
    while changed:
        changed = False
        for o in list(g.ops):
            src, dst = g.tensors[o.inputs[0]], g.tensors[o.outputs[0]]
            shape_only = (o.kind in SHAPE_ONLY and rshape(sh[src.index]) == rshape(sh[dst.index])
                          and src.dtype == dst.dtype
                          and _rank_free(g, dst.index, len(sh[src.index]) == len(sh[dst.index])))
            from_compare = (o.kind == "CAST" and src.dtype in ("bool", "float32")
                            and dst.dtype == "float32")
            if shape_only or from_compare:
                g.ops.remove(o)
                alias(g, o.outputs[0], o.inputs[0])
                changed = True
            elif o.kind in ("SELECT", "SELECT_V2") and _negative_wrap(g, o):
                g.ops.remove(o)
                alias(g, o.outputs[0], o.inputs[2])
                notes.add("ids: TFLite's wrap of negative ids before a lookup is dropped; rayo "
                          "refuses an id outside the table (RAYO_ERR_SHAPE) instead")
                changed = True
    _dead(g)


SHAPE_ONLY = {"RESHAPE", "EXPAND_DIMS", "SQUEEZE", "ONNX_RESHAPE", "ONNX_FLATTEN", "ONNX_SQUEEZE",
              "ONNX_UNSQUEEZE", "IDENTITY"}

# Ops that read an input the same whatever leading ones its shape has: an alias of a reshape
# that only adds or drops leading ones is safe before them. Ops that read axes are not.
RANK_FREE = {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE", "POW",
             "NEG", "ABS", "SQUARE", "SQRT", "RSQRT", "EXP", "LOG", "TANH", "LOGISTIC", "RELU",
             "RELU6", "GELU", "SOFTMAX", "FULLY_CONNECTED", "CAST", "GREATER", "GREATER_EQUAL",
             "LESS", "LESS_EQUAL", "EQUAL", "NOT_EQUAL", "DEQUANTIZE", "CONCATENATION",
             "CONV_2D", "rayo:dense", "rayo:conv1d", "rayo:layer_norm", "LAYER_NORM",
             "SELECT", "SELECT_V2", "GATHER", "RESHAPE", "ONNX_RESHAPE", "ONNX_ERF"}


def _rank_free(g: Graph, t: int, same_rank: bool) -> bool:
    if same_rank:
        return True
    for u in g.ops:
        if t not in u.inputs:
            continue
        if u.kind not in RANK_FREE:
            return False
        if u.kind in ("GATHER",) and u.inputs[0] == t:
            return False
        if u.kind == "CONCATENATION" and int(u.options.get("axis", -1)) >= 0:
            return False
    return True


def _negative_wrap(g: Graph, o: Op) -> bool:
    """SELECT(LESS(i, 0), ADD(i, n), i) where every reader is a lookup of n rows."""
    prod = g.producer()
    cond, wrapped, ids = o.inputs
    pc, pw = prod.get(cond), prod.get(wrapped)
    if not pc or not pw or pc.kind != "LESS" or pw.kind != "ADD":
        return False
    if pc.inputs[0] != ids or uniform(const_value(g, pc.inputs[1])) != 0:
        return False
    if pw.inputs[0] != ids:
        return False
    n = uniform(const_value(g, pw.inputs[1]))
    users = [u for u in g.ops if o.outputs[0] in u.inputs]
    if n is None or not users:
        return False
    for u in users:
        if u.kind != "GATHER" or u.inputs[1] != o.outputs[0] or int(u.options.get("axis", 0)):
            return False
        if g.tensors[u.inputs[0]].shape[0] != int(n):
            return False
    return True


def _dead(g: Graph) -> None:
    """Removes ops whose outputs nothing reads."""
    while True:
        used = set(g.outputs) | {i for o in g.ops for i in o.inputs}
        keep = [o for o in g.ops if any(t in used for t in o.outputs)]
        if len(keep) == len(g.ops):
            return
        g.ops = keep


def split_inputs(g: Graph, sh: dict[int, tuple[int, ...]], splits: dict[str, list[tuple[str, int]]],
                 names: dict[int, str]) -> None:
    """Presents a model input as several: ``splits`` maps an input's name to its parts (name,
    size) along its last axis. Every reader must be a slice that lies within one part."""
    for name, parts in splits.items():
        idx = [i for i in g.inputs if names.get(i) == name]
        if not idx:
            raise ConvertError(f"--split-input: no input {name!r} (inputs: "
                               f"{', '.join(names[i] for i in g.inputs)})")
        x = idx[0]
        t = g.tensors[x]
        total = sh[x][-1]
        if sum(s for _, s in parts) != total:
            raise ConvertError(f"--split-input {name}: the parts add up to "
                               f"{sum(s for _, s in parts)}, the input's last axis is {total}")
        starts = np.cumsum([0] + [s for _, s in parts])
        new_ids = []
        for (pname, size), start in zip(parts, starts, strict=False):
            shape = (*sh[x][:-1], size)
            new = g.add_tensor(pname, t.dtype, shape)
            sh[new] = shape
            names[new] = pname
            new_ids.append((new, int(start), size))
        for o in [o for o in g.ops if x in o.inputs]:
            if o.kind not in ("STRIDED_SLICE", "SLICE") or o.inputs[0] != x:
                raise ConvertError(f"--split-input {name}: {g.describe(o)} reads it whole")
            probe = np.arange(total, dtype=np.int64)
            probe = np.broadcast_to(probe, sh[x]).copy()
            args = [probe] + [g.tensors[i].data for i in o.inputs[1:]]
            y = ref.evaluate(g, o, args)[0]
            flat = y.reshape(-1)
            hit = [p for p in new_ids if p[1] <= flat.min() and flat.max() < p[1] + p[2]]
            if not hit:
                raise ConvertError(f"--split-input {name}: {g.describe(o)} reads across parts")
            new, start, size = hit[0]
            # The same slice of the part: its columns shifted by the part's start.
            if flat.size == np.prod(sh[new]) and np.array_equal(
                    y.reshape(sh[new]), np.broadcast_to(np.arange(start, start + size),
                                                        sh[new])):
                g.ops.remove(o)
                alias(g, o.outputs[0], new)
                continue
            raise ConvertError(f"--split-input {name}: {g.describe(o)} reads only some of part "
                               f"{g.tensors[new].name}")
        g.inputs = [i for j in g.inputs for i in ([p[0] for p in new_ids] if j == x else [j])]
        del names[x]


# Fusions.


def _act_name(code: Any) -> str | None:
    """rayo's name for TFLite's fused activation, where dense and conv1d fuse it."""
    return {0: "none", 1: "relu"}.get(int(code or 0))


def fuse_conv1d(g: Graph, sh: dict[int, tuple[int, ...]], notes: Notes) -> None:
    """CONV_2D over one row ([1, 1, T, C], kernel [O, 1, k, C], stride 1, same padding), and the
    dilated form TFLite emits (SPACE_TO_BATCH_ND, CONV_2D valid, BATCH_TO_SPACE_ND), to
    rayo's conv1d."""
    prod = g.producer()
    for o in list(g.ops):
        if o.kind != "CONV_2D" or o not in g.ops:
            continue
        x = o.inputs[0]
        k = g.tensors[o.inputs[1]]
        opt = o.options
        ys = sh[o.outputs[0]]
        if not k.const or len(k.shape) != 4 or k.shape[1] != 1 or len(ys) != 4 or \
                sh[x][-1] != k.shape[3]:
            continue
        if int(opt.get("stride_w", 1)) != 1 or int(opt.get("stride_h", 1)) != 1:
            continue
        size = k.shape[2]
        dil = int(opt.get("dilation_w_factor", 1))
        bias = o.inputs[2] if len(o.inputs) > 2 else -1
        same = int(opt.get("padding", 0)) == 0
        one_row = ys[0] == 1 and ys[1] == 1 and len(rshape(sh[x])) <= 2
        if not one_row:
            same = False  # only the SPACE_TO_BATCH_ND form below
        if same and dil == 1 and (size - 1) % 2 == 0:
            o.kind = "rayo:conv1d"
            o.options = {"size": size, "dilation": 1, "kernel": o.inputs[1], "bias": bias,
                         "fused": opt.get("fused_activation_function", 0)}
            o.inputs = [x]
            continue
        if same and dil > 1 and k.dtype == "int8":
            continue  # TFLite quantises the whole input; rayo's dilated conv1d each phase
        if same and dil > 1 and (size - 1) * dil % 2 == 0:
            o.kind = "rayo:conv1d"
            o.options = {"size": size, "dilation": dil, "kernel": o.inputs[1], "bias": bias,
                         "fused": opt.get("fused_activation_function", 0)}
            o.inputs = [x]
            continue
        # SPACE_TO_BATCH_ND -> [EXPAND_DIMS] -> CONV_2D (valid) -> [RESHAPE] -> BATCH_TO_SPACE_ND
        if same or dil != 1:
            continue
        chain_in = x
        p = prod.get(chain_in)
        expand = None
        if p is not None and p.kind in ("EXPAND_DIMS", "RESHAPE"):
            expand, p = p, prod.get(p.inputs[0])
        if p is None or p.kind != "SPACE_TO_BATCH_ND":
            continue
        s2b = p
        block = const_value(g, s2b.inputs[1])
        pads = const_value(g, s2b.inputs[2])
        if block is None or pads is None or block.size != 1:
            continue
        d = int(block.reshape(-1)[0])
        out = single_use(g, o.outputs[0])
        squeeze = None
        if out is not None and out.kind in ("RESHAPE", "SQUEEZE"):
            squeeze, out = out, single_use(g, out.outputs[0])
        if out is None or out.kind != "BATCH_TO_SPACE_ND":
            continue
        b2s = out
        crops = const_value(g, b2s.inputs[2])
        bb = const_value(g, b2s.inputs[1])
        if crops is None or bb is None or int(bb.reshape(-1)[0]) != d:
            continue
        pb, crop0 = int(pads.reshape(-1)[0]), int(crops.reshape(-1)[0])
        src = s2b.inputs[0]
        if (size - 1) * d % 2 or pb != (size - 1) * d // 2 or crop0 != 0 or \
                rshape(sh[b2s.outputs[0]]) != rshape(sh[src])[:-1] + [k.shape[0]]:
            notes.add(f"{g.describe(s2b)}: not the dilated convolution rayo's conv1d computes "
                      f"(padding {pads.reshape(-1).tolist()}, crops {crops.reshape(-1).tolist()})")
            continue
        inner = [t for t in (s2b.outputs[0], expand.outputs[0] if expand else None,
                             o.outputs[0], squeeze.outputs[0] if squeeze else None)
                 if t is not None]
        if any(single_use(g, t) is None for t in inner):
            continue
        new = Op(o.index, "rayo:conv1d", [src], [b2s.outputs[0]],
                 {"size": size, "dilation": d, "kernel": o.inputs[1], "bias": bias,
                  "fused": opt.get("fused_activation_function", 0)})
        at = g.ops.index(b2s)
        g.ops[at] = new
        for dead in (s2b, expand, o, squeeze):
            if dead is not None:
                g.ops.remove(dead)
        prod = g.producer()


def fuse_bias_activation(g: Graph) -> None:
    """A dense or conv1d followed by the ADD of a constant bias, or by RELU or GELU, takes them
    in: TFLite leaves the bias out of the op where the layer was a Tensordot or a dilated
    convolution."""
    changed = True
    while changed:
        changed = False
        for o in list(g.ops):
            if o.kind == "FULLY_CONNECTED":
                _fc_to_dense(g, o)
            if o.kind not in ("rayo:dense", "rayo:conv1d"):
                continue
            nxt = single_use(g, o.outputs[0])
            if nxt is None:
                continue
            opt = o.options
            free = int(opt.get("fused", 0)) == 0 and opt.get("activation", "none") == "none"
            if nxt.kind == "ADD" and free:
                other = nxt.inputs[1] if nxt.inputs[0] == o.outputs[0] else nxt.inputs[0]
                c = const_value(g, other)
                out_size = g.tensors[opt["kernel"]].shape[0]
                if c is None or c.dtype.kind != "f" or c.size != out_size or \
                        c.reshape(-1).shape[0] != c.shape[-1]:
                    continue
                if _act_name(nxt.options.get("fused_activation_function")) is None:
                    continue
                b = c.reshape(-1).astype(np.float32)
                if opt["bias"] >= 0:
                    b = (g.tensors[opt["bias"]].data.astype(np.float32).reshape(-1) + b).astype(
                        np.float32)
                bias_name = g.tensors[other].name
                opt["bias"] = g.add_const(bias_name, b)
                opt["fused"] = nxt.options.get("fused_activation_function", 0)
                o.outputs = nxt.outputs
                g.ops.remove(nxt)
                changed = True
            elif nxt.kind in ("RELU", "GELU") and free:
                if nxt.kind == "GELU" and nxt.options.get("approximate"):
                    continue
                opt["activation"] = "relu" if nxt.kind == "RELU" else "gelu_erf"
                o.outputs = nxt.outputs
                g.ops.remove(nxt)
                changed = True


def _fc_to_dense(g: Graph, o: Op) -> None:
    k = g.tensors[o.inputs[1]]
    if not k.const or len(k.shape) != 2:
        return
    if k.dtype == "int8" and not o.options.get("asymmetric_quantize_inputs", False):
        raise ConvertError(f"{g.describe(o)}: an int8 FULLY_CONNECTED that quantises its input "
                           "symmetrically; rayo runs TFLite's asymmetric hybrid kernel only "
                           "(convert with asymmetric_quantize_inputs, TFLite's default since 2.6)")
    bias = o.inputs[2] if len(o.inputs) > 2 else -1
    o.kind = "rayo:dense"
    o.options = {"kernel": o.inputs[1], "bias": bias,
                 "fused": o.options.get("fused_activation_function", 0),
                 "keep_num_dims": bool(o.options.get("keep_num_dims"))}
    o.inputs = [o.inputs[0]]


def fuse_layer_norm(g: Graph, sh: dict[int, tuple[int, ...]]) -> None:
    """Keras' LayerNormalization as TFLite decomposes it, to rayo's layer_norm:

        mean = MEAN(x, -1); var = MEAN(SQUARED_DIFFERENCE(x, mean), -1)
        m = RSQRT(var + eps) * gamma
        y = x * m + (beta + NEG(mean) * m)       (or beta - mean * m)

    and the plain form ``(x - mean) * RSQRT(var + eps) * gamma + beta``."""
    for o in list(g.ops):
        if o.kind != "RSQRT" or o not in g.ops:
            continue
        m = _match_layer_norm(g, sh, o)
        if m is None:
            continue
        x, gamma, beta, eps, out, ops = m
        new = Op(o.index, "rayo:layer_norm", [x], [out],
                 {"gamma": gamma, "beta": beta, "epsilon": eps})
        at = min(g.ops.index(p) for p in ops)
        for p in ops:
            g.ops.remove(p)
        g.ops.insert(at, new)


def _other(o: Op, t: int) -> int:
    return o.inputs[1] if o.inputs[0] == t else o.inputs[0]


def _users(g: Graph, t: int) -> list[Op]:
    return [o for o in g.ops if t in o.inputs]


def _match_layer_norm(g: Graph, sh: dict[int, tuple[int, ...]], rs: Op) -> Any:
    prod = g.producer()
    add_eps = prod.get(rs.inputs[0])
    if not add_eps or add_eps.kind != "ADD":
        return None
    var_t = next((i for i in add_eps.inputs if not g.tensors[i].const), None)
    eps_t = _other(add_eps, var_t) if var_t is not None else None
    eps = uniform(const_value(g, eps_t)) if eps_t is not None else None
    var_op = prod.get(var_t) if var_t is not None else None
    if eps is None or not var_op or var_op.kind != "MEAN" or not _last_axis(g, sh, var_op):
        return None
    sq = prod.get(var_op.inputs[0])
    if not sq or sq.kind != "SQUARED_DIFFERENCE":
        return None
    mean_op = None
    x = None
    for a, b in ((sq.inputs[0], sq.inputs[1]), (sq.inputs[1], sq.inputs[0])):
        p = prod.get(b)
        if p and p.kind == "MEAN" and p.inputs[0] == a and _last_axis(g, sh, p):
            mean_op, x = p, a
    if mean_op is None or x is None:
        return None
    mean = mean_op.outputs[0]
    c = sh[x][-1]
    # m = rsqrt * gamma (or rsqrt alone)
    r = rs.outputs[0]
    ops = [mean_op, sq, var_op, add_eps, rs]
    m = r
    gamma = None
    gamma_t = beta_t = -1
    users = _users(g, r)
    gm = [u for u in users if u.kind == "MUL" and g.tensors[_other(u, r)].const]
    if len(users) == 1 and gm:
        gamma_t = _other(gm[0], r)
        gamma = const_value(g, gamma_t)
        m = gm[0].outputs[0]
        ops.append(gm[0])
    # Form A: y = x * m + (beta + neg(mean) * m)  /  y = x * m + (beta - mean * m)
    mu = _users(g, m)
    xm = next((u for u in mu if u.kind == "MUL" and _other(u, m) == x), None)
    shift = None
    beta = None
    if xm is not None and len(mu) == 2:
        rest = next(u for u in mu if u is not xm)
        nm = None
        if rest.kind == "MUL":
            nm = prod.get(_other(rest, m))
            if not nm or nm.kind != "NEG" or nm.inputs[0] != mean:
                return None
            t = single_use(g, rest.outputs[0])
            if not t or t.kind != "ADD" or not g.tensors[_other(t, rest.outputs[0])].const:
                return None
            beta_t = _other(t, rest.outputs[0])
            beta = const_value(g, beta_t)
            ops += [nm, rest, t]
            shift = t.outputs[0]
        else:
            return None
        final = single_use(g, xm.outputs[0])
        if not final or final.kind != "ADD" or _other(final, xm.outputs[0]) != shift:
            return None
        ops += [xm, final]
        out = final.outputs[0]
    else:
        # Form B: (x - mean) * r [* gamma] [+ beta]
        sub = next((u for u in _users(g, mean) if u.kind == "SUB" and u.inputs == [x, mean]),
                   None)
        if sub is None or len(_users(g, mean)) != 2:
            return None
        mul = single_use(g, sub.outputs[0])
        if not mul or mul.kind != "MUL" or _other(mul, sub.outputs[0]) != r or gamma is not None:
            return None
        ops += [sub, mul]
        out = mul.outputs[0]
        nxt = single_use(g, out)
        if nxt and nxt.kind == "MUL" and g.tensors[_other(nxt, out)].const:
            gamma_t = _other(nxt, out)
            gamma = const_value(g, gamma_t)
            ops.append(nxt)
            out = nxt.outputs[0]
            nxt = single_use(g, out)
        if nxt and nxt.kind == "ADD" and g.tensors[_other(nxt, out)].const:
            beta_t = _other(nxt, out)
            beta = const_value(g, beta_t)
            ops.append(nxt)
            out = nxt.outputs[0]
    gamma_v = np.ones(c, np.float32) if gamma is None else gamma
    beta_v = np.zeros(c, np.float32) if beta is None else beta
    if gamma_v.size != c or beta_v.size != c or gamma_v.dtype.kind != "f":
        return None
    inner = {t for p in ops for t in p.outputs} - {out}
    for t in inner:
        if t in g.outputs or any(u not in ops for u in _users(g, t)):
            return None
    near = g.tensors[rs.outputs[0]].name
    gid = g.add_const(g.tensors[gamma_t].name if gamma_t >= 0 else near,
                      gamma_v.reshape(-1).astype(np.float32))
    bid = g.add_const(g.tensors[beta_t].name if beta_t >= 0 else near,
                      beta_v.reshape(-1).astype(np.float32))
    return x, gid, bid, float(eps), out, ops


def _last_axis(g: Graph, sh: dict[int, tuple[int, ...]], o: Op) -> bool:
    axes = const_value(g, o.inputs[1])
    if axes is None or axes.size != 1:
        return False
    rank = len(sh[o.inputs[0]])
    return int(axes.reshape(-1)[0]) % rank == rank - 1 and bool(o.options.get("keep_dims"))


def sink_slices(g: Graph, sh: dict[int, tuple[int, ...]]) -> None:
    """A TRANSPOSE read only by slices becomes, for each slice, the slice of its input and a
    transpose of that: the same values, and no tensor of the transposed rank (a fused qkv
    projection's [3, 1, H, T, d], which rayo could not hold, becomes three [H, T, d])."""
    for t in [o for o in g.ops if o.kind == "TRANSPOSE"]:
        if t not in g.ops or const_value(g, t.inputs[1]) is None:
            continue
        users = [u for u in g.ops if t.outputs[0] in u.inputs]
        if not users or t.outputs[0] in g.outputs or any(
                u.kind != "STRIDED_SLICE" or u.inputs[0] != t.outputs[0]
                or any(const_value(g, i) is None for i in u.inputs[1:]) for u in users):
            continue
        perm = [int(p) for p in const_value(g, t.inputs[1]).reshape(-1)]
        src = sh[t.inputs[0]]
        out = sh[t.outputs[0]]
        new_ops: list[tuple[Op, list[Op]]] = []
        for u in users:
            vals = [[int(v) for v in g.tensors[i].data.reshape(-1)] for i in u.inputs[1:]]
            try:
                spec, shrink = ref.strided_slice_spec(out, *vals, u.options)
            except ConvertError:
                break
            if any(s.step not in (None, 1) for s in spec):
                break
            begin = [0] * len(src)
            end = list(src)
            for a, s in enumerate(spec):
                b, e, _ = s.indices(out[a])
                begin[perm[a]], end[perm[a]] = b, e
            x = t.inputs[0]
            part_shape = tuple(e - b for b, e in zip(begin, end, strict=True))
            part = g.add_tensor(g.tensors[u.outputs[0]].name + "/part", g.tensors[x].dtype,
                                part_shape)
            sh[part] = part_shape
            sl = Op(u.index, "STRIDED_SLICE",
                    [x, g.add_const(f"{g.tensors[part].name}/begin", np.asarray(begin, np.int32)),
                     g.add_const(f"{g.tensors[part].name}/end", np.asarray(end, np.int32)),
                     g.add_const(f"{g.tensors[part].name}/strides",
                                 np.ones(len(src), np.int32))], [part], {})
            moved_shape = tuple(part_shape[p] for p in perm)
            moved = g.add_tensor(g.tensors[u.outputs[0]].name + "/moved", g.tensors[x].dtype,
                                 moved_shape)
            sh[moved] = moved_shape
            tr = Op(t.index, "TRANSPOSE", [part, t.inputs[1]], [moved], {})
            rs = Op(u.index, "RESHAPE",
                    [moved, g.add_const(g.tensors[u.outputs[0]].name + "/shape",
                                        np.asarray(sh[u.outputs[0]], np.int32))],
                    list(u.outputs), {})
            new_ops.append((u, [sl, tr, rs]))
            del shrink
        else:
            for u, ops in new_ops:
                at = g.ops.index(u)
                g.ops[at : at + 1] = ops
            g.ops.remove(t)
