"""Lowering the rewritten source graph to rayo's graph (docs/FORMAT.md) and weights.

Shapes: rayo holds each tensor without its leading ones (the batch of 1), which changes no
broadcast. Modes:

- ``fixed``: every size as in the source.
- ``sequence``: the sequence inputs' length is the size symbol ``T`` (at most the source's
  length), so a caller runs as many positions as it has. Every op must be expressible for any
  ``T``.
- ``decode``: also ``P``, the positions already run: each run takes the new tokens at ``P`` ..
  ``P + T - 1``, keeps every attention's keys and values in a cache (graph states), and gives
  the outputs at the new positions (by default the last one). Every op must work on each
  position alone, except the recognised causal attention blocks.

An op that cannot be expressed in a mode raises SequenceError, and the converter tries the next
mode; an op rayo cannot run at all raises ConvertError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import ref
from .ir import ConvertError, Graph, Op
from .naming import Namer
from .passes import Notes, const_value, rshape, uniform


class SequenceError(ConvertError):
    """An op that does not fit the sequence or decode mode."""


@dataclass
class Value:
    name: str
    shape: list[int]  # concrete, with T at its largest
    seq: int | None = None  # the axis of T
    dtype: str = "float32"


@dataclass
class Weight:
    name: str
    data: np.ndarray  # float32 or int8
    scales: np.ndarray | None = None
    axis: int | None = None
    zero_point: int = 0


@dataclass
class Lowered:
    graph: dict[str, Any]
    weights: list[Weight]
    notes: list[str] = field(default_factory=list)


ROWWISE_UNARY = {"neg", "abs", "square", "sqrt", "rsqrt", "exp", "log", "tanh", "sigmoid", "relu",
                 "relu6", "gelu_erf", "gelu_tanh"}
_UNARY = {"NEG": "neg", "ABS": "abs", "SQUARE": "square", "SQRT": "sqrt", "RSQRT": "rsqrt",
          "EXP": "exp", "LOG": "log", "TANH": "tanh", "LOGISTIC": "sigmoid", "RELU": "relu",
          "RELU6": "relu6"}
_BINARY = {"ADD": "add", "SUB": "sub", "MUL": "mul", "DIV": "div", "MAXIMUM": "maximum",
           "MINIMUM": "minimum", "SQUARED_DIFFERENCE": "squared_difference"}
_COMPARE = {"GREATER": "greater", "GREATER_EQUAL": "greater_equal", "LESS": "less",
            "LESS_EQUAL": "less_equal", "EQUAL": "equal", "NOT_EQUAL": "not_equal"}
_FLIP = {"greater": "less", "less": "greater", "greater_equal": "less_equal",
         "less_equal": "greater_equal", "equal": "equal", "not_equal": "not_equal"}
_REDUCE = {"MEAN": "mean", "SUM": "sum", "REDUCE_MAX": "max", "REDUCE_MIN": "min",
           "REDUCE_PROD": "prod"}
_FUSED = {0: None, 1: "relu", 3: "relu6", 4: "tanh"}


class Lowerer:
    def __init__(self, g: Graph, sh: dict[int, tuple[int, ...]], mode: str, seq_inputs: list[int],
                 length: int, names: dict[int, str], out_names: dict[int, str],
                 positions: str, namer: Namer, notes: Notes) -> None:
        self.g, self.sh, self.mode = g, sh, mode
        self.seq_inputs, self.length = seq_inputs, length
        self.names, self.out_names, self.positions = names, out_names, positions
        self.namer, self.notes = namer, notes
        self.nodes: list[dict[str, Any]] = []
        self.meta: list[dict[str, Any]] = []
        self.weights: list[Weight] = []
        self.weight_of: dict[tuple[int, str], str] = {}
        self.vals: dict[int, Value] = {}
        self.states: list[dict[str, Any]] = []
        self.pos: str | None = None
        self.errors: list[str] = []
        self.prod = g.producer()
        self.current: Op | None = None

    # Emitting.

    def node(self, op: str, inputs: list[str], out_shape: list[int], seq: int | None,
             dtype: str = "float32", hint: str | None = None, rowwise: int | None = None,
             **attrs: Any) -> Value:
        base = hint or self._hint() or op
        name = self.namer.unique(f"{base}.{op}" if not base.endswith(op) else base)
        self.nodes.append({"op": op, "inputs": inputs, "output": name, **attrs})
        self.meta.append({"shape": list(out_shape), "seq": seq, "rowwise": rowwise})
        return Value(name, list(out_shape), seq, dtype)

    def _hint(self) -> str | None:
        if self.current is None:
            return None
        for t in [*[v for k, v in self.current.options.items() if k in ("kernel", "gamma")],
                  *self.current.outputs, *self.current.inputs]:
            if isinstance(t, int) and t >= 0:
                layer = self.g.hints.get(t) or self.namer.layer(self.g.tensors[t].name)
                if layer:
                    return layer
        return None

    def weight(self, t: int, role: str, data: np.ndarray | None = None) -> str:
        key = (t, role)
        if key in self.weight_of and data is None:
            return self.weight_of[key]
        tensor = self.g.tensors[t]
        layer = self.g.hints.get(t) or self.namer.layer(tensor.name)
        if not layer and self.current is not None:
            layer = self._hint()
        name = self.namer.unique(f"{layer}.{role}" if layer else role)
        if data is not None:
            w = Weight(name, np.asarray(data, np.float32))
        elif tensor.dtype == "int8" and tensor.quant is not None:
            q = tensor.quant
            zp = np.unique(q.zero_points)
            if zp.size > 1:
                raise ConvertError(f"{tensor.name}: int8 zero points differ by channel")
            axis = q.axis if q.scales.size > 1 else None
            w = Weight(name, tensor.data.astype(np.int8), q.scales.astype(np.float32).reshape(-1),
                       axis, int(zp[0]) if zp.size else 0)
        else:
            if tensor.data is None or tensor.data.dtype.kind not in "fiub":
                raise ConvertError(f"{tensor.name}: not a constant rayo can hold")
            w = Weight(name, tensor.data.astype(np.float32))
        self.weights.append(w)
        if data is None:
            self.weight_of[key] = name
        return name

    def const_weight(self, value: np.ndarray, role: str, near: int) -> str:
        """A weight of the converter's own (a reshaped constant), named after tensor ``near``."""
        return self.weight(near, role, np.asarray(value, np.float32))

    # Values.

    def val(self, t: int) -> Value:
        if t in self.vals:
            return self.vals[t]
        tensor = self.g.tensors[t]
        if tensor.const:
            raise ConvertError(f"{tensor.name}: a constant where rayo needs an activation")
        raise ConvertError(f"{tensor.name}: no value (an op before it was not converted)")

    def mix(self, why: str) -> None:
        if self.mode == "decode":
            raise SequenceError(f"{self.g.describe(self.current)} {why}, so it cannot run one "
                                "position at a time" if self.current else why)

    def sym(self, shape: list[int], seq: int | None) -> list[Any]:
        return [("T" if i == seq else d) for i, d in enumerate(shape)]

    def fit(self, v: Value, want: list[int]) -> Value:
        """``v`` reshaped to ``want`` (rayo's shape of the source tensor), if it differs."""
        if v.shape == want:
            return v
        if int(np.prod(v.shape)) != int(np.prod(want)):
            raise ConvertError(f"{self.g.describe(self.current)}: rayo's shape {v.shape} does "
                               f"not fit {want}")
        seq = None
        if v.seq is not None:
            a, L = v.seq, v.shape[v.seq]
            pre = int(np.prod(v.shape[:a]))
            for b, d in enumerate(want):
                if d == L and int(np.prod(want[:b])) == pre:
                    seq = b
                    break
            if seq is None:
                raise SequenceError(f"{self.g.describe(self.current)} reshapes the sequence "
                                    f"axis ({v.shape} to {want})")
        return self.node("reshape", [v.name], want, seq, v.dtype, shape=self.sym(want, seq),
                         rowwise=0)

    def positions_value(self) -> str:
        if self.pos is None:
            start: Any = "P" if self.mode == "decode" else 0
            self.pos = self.namer.unique("positions")
            self.nodes.append({"op": "range", "inputs": [], "output": self.pos, "start": start,
                               "size": "T"})
            self.meta.append({"shape": [self.length], "seq": 0, "rowwise": None})
        return self.pos

    # The lowering.

    def run(self) -> Lowered:
        g = self.g
        inputs = []
        for i in g.inputs:
            t = g.tensors[i]
            shape = rshape(self.sh[i])
            dtype = {"int32": "int32", "int64": "int32", "float32": "float32"}.get(t.dtype)
            if dtype is None:
                raise ConvertError(f"input {self.names.get(i, t.name)}: dtype {t.dtype}; rayo "
                                   "takes int32 or float32 inputs")
            if t.dtype == "int64":
                self.notes.add(f"input {self.names.get(i, t.name)}: int64 in the source, int32 "
                               "in rayo")
            seq = 0 if (self.mode != "fixed" and i in self.seq_inputs) else None
            name = self.namer.unique(self.names.get(i, t.name))
            self.vals[i] = Value(name, shape, seq, dtype)
            inputs.append({"name": name, "dtype": dtype, "shape": self.sym(shape, seq)})
        for o in g.ops:
            self.current = o
            try:
                self.lower(o)
            except SequenceError:
                raise
            except ConvertError as e:
                self.errors.append(str(e) if str(e).startswith("op ") else
                                   f"{g.describe(o)}: {e}")
                for t in o.outputs:
                    self.vals[t] = Value(f"?{t}", rshape(self.sh[t]))
        self.current = None
        if self.errors:
            raise ConvertError("\n".join(self.errors))
        outputs = []
        for t in g.outputs:
            v = self.vals.get(t)
            if v is None:
                raise ConvertError(f"output {self.g.tensors[t].name}: a constant; rayo's outputs "
                                   "are computed")
            if not any(n["output"] == v.name for n in self.nodes):
                v = self.node("reshape", [v.name], v.shape, v.seq, v.dtype, hint="output",
                              shape=self.sym(v.shape, v.seq))
            if self.mode == "decode" and v.seq is not None and self.positions == "last":
                if v.seq != 0 or len(v.shape) != 2:
                    self.notes.add(f"output {self.out_names.get(t, t)}: every new position "
                                   f"({self.sym(v.shape, v.seq)}); only [T, C] outputs are cut "
                                   "to the last")
                else:
                    v = self.last_row(v)
            outputs.append({"name": self.out_names.get(t) or self.g.tensors[t].name,
                            "node": v.name})
        graph: dict[str, Any] = {"version": 1}
        if self.mode == "sequence":
            graph["symbols"] = {"T": self.length}
        elif self.mode == "decode":
            graph["symbols"] = {"T": self.length, "P": self.length - 1}
            graph["plan"] = {"P": 0}
        graph["inputs"] = inputs
        if self.states:
            graph["states"] = self.states
        graph["nodes"] = self.nodes
        graph["outputs"] = outputs
        used = {i for n in self.nodes for i in n["inputs"]}
        weights = [w for w in self.weights if w.name in used]
        return Lowered(graph, weights)

    def last_row(self, v: Value) -> Value:
        """last_row of ``v`` [T, C], moved before the ops that work on each row alone (so a
        decoder's final norm and logits run on one row)."""

        def at(name: str) -> int | None:
            return next((k for k, n in enumerate(self.nodes) if n["output"] == name), None)

        def users(name: str) -> list[int]:
            return [k for k, n in enumerate(self.nodes) if name in n["inputs"]]

        def seq_of(name: str) -> int | None:
            k = at(name)
            if k is not None:
                return self.meta[k]["seq"]
            return next((x.seq for x in self.vals.values() if x.name == name), None)

        chain: list[int] = []
        cur = v.name
        while True:
            i = at(cur)
            if i is None or (chain and users(cur) != [chain[-1]]) or (not chain and users(cur)):
                break
            n, m = self.nodes[i], self.meta[i]
            src = m["rowwise"]
            if src is None or n["op"] == "reshape":
                break
            x = n["inputs"][src]
            j = at(x)
            if j is None or self.meta[j]["seq"] != 0 or len(self.meta[j]["shape"]) != 2:
                break
            if any(seq_of(inp) is not None for k, inp in enumerate(n["inputs"]) if k != src):
                break
            chain.append(i)
            cur = x
        last = self.namer.unique(f"{cur}.last")
        if not chain:
            k = at(v.name)
            self.nodes.insert(k + 1, {"op": "last_row", "inputs": [v.name], "output": last})
            self.meta.insert(k + 1, {"shape": v.shape[1:], "seq": None, "rowwise": None})
            return Value(last, v.shape[1:], None)
        width = self.meta[at(cur)]["shape"][1]
        moved = [(self.nodes[i], self.meta[i]) for i in sorted(chain)]
        for i in sorted(chain, reverse=True):
            self.nodes.pop(i)
            self.meta.pop(i)
        # Where the output's node was: every other input of the moved ops is made before it.
        k = chain[0] - (len(chain) - 1)
        insert = [({"op": "last_row", "inputs": [cur], "output": last},
                   {"shape": [width], "seq": None, "rowwise": None})]
        prev = last
        for n, m in moved:
            n = dict(n)
            n["inputs"] = [prev if j == m["rowwise"] else inp for j, inp in enumerate(n["inputs"])]
            insert.append((n, {"shape": m["shape"][1:], "seq": None, "rowwise": None}))
            prev = n["output"]
        for off, (n, m) in enumerate(insert):
            self.nodes.insert(k + off, n)
            self.meta.insert(k + off, m)
        return Value(v.name, v.shape[1:], None)

    # Ops.

    def lower(self, o: Op) -> None:
        k = o.kind
        if k == "rayo:dense":
            self.dense(o)
        elif k == "rayo:conv1d":
            self.conv1d(o)
        elif k == "rayo:layer_norm":
            self.layer_norm(o)
        elif k == "rayo:attention":
            self.attention(o)
        elif k == "LAYER_NORM":
            self.onnx_layer_norm(o)
        elif k in _BINARY:
            self.binary(o, _BINARY[k])
        elif k in _UNARY or k == "GELU":
            name = _UNARY.get(k) or ("gelu_tanh" if o.options.get("approximate") else "gelu_erf")
            x = self.val(o.inputs[0])
            self.out(o, self.node(name, [x.name], x.shape, x.seq, rowwise=0))
        elif k == "SOFTMAX":
            self.softmax(o)
        elif k in ("GATHER", "EMBEDDING_LOOKUP"):
            self.gather(o)
        elif k == "DEQUANTIZE":
            src = self.prod.get(o.inputs[0])
            if src is None or src.kind not in ("GATHER", "EMBEDDING_LOOKUP"):
                raise ConvertError("DEQUANTIZE of an int8 activation: rayo's activations are "
                                   "float32 (dynamic-range models only)")
            self.vals[o.outputs[0]] = self.val(o.inputs[0])
        elif k == "CONCATENATION":
            self.concat(o)
        elif k in ("RESHAPE", "EXPAND_DIMS", "SQUEEZE", "ONNX_RESHAPE", "ONNX_FLATTEN",
                   "ONNX_SQUEEZE", "ONNX_UNSQUEEZE", "IDENTITY"):
            x = self.val(o.inputs[0])
            self.vals[o.outputs[0]] = self.fit(x, rshape(self.sh[o.outputs[0]]))
        elif k == "TRANSPOSE":
            self.transpose(o)
        elif k in ("STRIDED_SLICE", "SLICE"):
            self.slice(o)
        elif k == "ONNX_SLICE":
            x = self.val(o.inputs[0])
            src = self.sh[o.inputs[0]]
            args = [None, *[self.g.tensors[i].data if i >= 0 else None for i in o.inputs[1:]]]
            if any(v is None for v in args[1:3]):
                raise ConvertError("a slice whose bounds are not constants")
            v = self._slice_value(x, src, ref.onnx_slice_spec(src, args))
            self.vals[o.outputs[0]] = self.fit(v, rshape(self.sh[o.outputs[0]]))
        elif k in ("SPLIT", "SPLIT_V", "ONNX_SPLIT"):
            self.split(o)
        elif k in _REDUCE:
            self.reduce(o)
        elif k == "BATCH_MATMUL":
            self.matmul(o)
        elif k == "FULLY_CONNECTED":
            self.fc_matmul(o)
        elif k in _COMPARE:
            self.compare(o)
        elif k == "CAST":
            raise ConvertError(f"CAST from {self.g.tensors[o.inputs[0]].dtype} to "
                               f"{self.g.tensors[o.outputs[0]].dtype}: rayo has no cast")
        elif k == "CONV_2D":
            self.conv2d_refusal(o)
        elif k in ("SELECT", "SELECT_V2"):
            self.select(o)
        else:
            raise ConvertError(why_not(k))

    def out(self, o: Op, v: Value, fused: Any = 0) -> None:
        act = _FUSED.get(int(fused or 0), "?")
        if act == "?":
            raise ConvertError(f"fused activation {fused}")
        if act:
            v = self.node(act, [v.name], v.shape, v.seq, rowwise=0)
        self.vals[o.outputs[0]] = self.fit(v, rshape(self.sh[o.outputs[0]]))

    def dense(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        if x.dtype != "float32":
            raise ConvertError("a dense layer over int32 values")
        if x.seq is not None and x.seq == len(x.shape) - 1:
            raise SequenceError(f"{self.g.describe(o)} mixes the sequence axis")
        kt = self.g.tensors[o.options["kernel"]]
        if kt.dtype == "int8" and kt.quant is not None and kt.quant.scales.size > 1 and \
                kt.quant.axis != 0:
            raise ConvertError("an int8 kernel quantised along its inputs; rayo's dense takes "
                               "one scale per output")
        out_n, in_n = kt.shape
        if x.shape[-1] != in_n:
            if len(x.shape) == 1 or int(np.prod(x.shape)) % in_n:
                raise ConvertError(f"input {x.shape} for a kernel of {in_n}")
            x = self.fit(x, [int(np.prod(x.shape)) // in_n, in_n])
        ins = [x.name, self.weight(o.options["kernel"], "kernel")]
        if o.options.get("bias", -1) >= 0:
            ins.append(self.weight(o.options["bias"], "bias"))
        attrs: dict[str, Any] = {}
        act = o.options.get("activation", "none")
        fused = int(o.options.get("fused", 0) or 0)
        if fused == 1 and act == "none":
            act, fused = "relu", 0
        if act != "none":
            attrs["activation"] = act
        if kt.dtype == "int8":
            attrs["input_quantization"] = "rows"
        v = self.node("dense", ins, [*x.shape[:-1], out_n], x.seq, rowwise=0, **attrs)
        self.out(o, v, fused)

    def conv1d(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        if len(x.shape) != 2:
            raise ConvertError(f"a 1-D convolution over {x.shape}, not [T, C]")
        if x.seq not in (None, 0):
            raise SequenceError(f"{self.g.describe(o)}: the sequence is not the convolved axis")
        if x.seq == 0:
            self.mix("convolves along the sequence")
        kt = self.g.tensors[o.options["kernel"]]
        ins = [x.name, self.weight(o.options["kernel"], "kernel")]
        if o.options.get("bias", -1) >= 0:
            ins.append(self.weight(o.options["bias"], "bias"))
        attrs: dict[str, Any] = {"size": int(o.options["size"]),
                                 "dilation": int(o.options["dilation"])}
        act = o.options.get("activation", "none")
        fused = int(o.options.get("fused", 0) or 0)
        if fused == 1 and act == "none":
            act, fused = "relu", 0
        if act != "none":
            attrs["activation"] = act
        if kt.dtype == "int8":
            attrs["input_quantization"] = "tensor_phases"
        v = self.node("conv1d", ins, [x.shape[0], kt.shape[0]], x.seq, **attrs)
        self.out(o, v, fused)

    def layer_norm(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        if x.seq is not None and x.seq == len(x.shape) - 1:
            raise SequenceError(f"{self.g.describe(o)} normalises over the sequence axis")
        ins = [x.name, self.weight(o.options["gamma"], "gamma"),
               self.weight(o.options["beta"], "beta")]
        v = self.node("layer_norm", ins, x.shape, x.seq, rowwise=0,
                      epsilon=_short(o.options.get("epsilon", 1e-5)))
        self.out(o, v)

    def onnx_layer_norm(self, o: Op) -> None:
        x_t, g_t, b_t = [*o.inputs, -1, -1][:3]
        c = self.sh[x_t][-1]
        gamma = self.g.tensors[g_t].data if g_t >= 0 else np.ones(c, np.float32)
        beta = self.g.tensors[b_t].data if b_t >= 0 else np.zeros(c, np.float32)
        if gamma is None or beta is None:
            raise ConvertError("LayerNormalization with a scale or bias that is not a constant")
        x = self.val(x_t)
        if x.seq is not None and x.seq == len(x.shape) - 1:
            raise SequenceError(f"{self.g.describe(o)} normalises over the sequence axis")
        near = g_t if g_t >= 0 else o.outputs[0]
        gw = (self.weight(g_t, "gamma") if g_t >= 0 and gamma.ndim == 1 else
              self.const_weight(np.broadcast_to(gamma, (c,)), "gamma", near))
        bw = (self.weight(b_t, "beta") if b_t >= 0 and beta.ndim == 1 else
              self.const_weight(np.broadcast_to(beta, (c,)), "beta", b_t if b_t >= 0 else near))
        v = self.node("layer_norm", [x.name, gw, bw], x.shape, x.seq, rowwise=0,
                      epsilon=_short(o.options.get("epsilon", 1e-5)))
        self.out(o, v)

    def attention(self, o: Op) -> None:
        q, k, v = (self.val(t) for t in o.inputs)
        for x, what in ((q, "query"), (k, "key"), (v, "value")):
            if len(x.shape) != 2:
                raise ConvertError(f"the attention's {what} is {x.shape}, not [T, D]")
        attrs: dict[str, Any] = {"heads": int(o.options["heads"]),
                                 "causal": int(bool(o.options["causal"]))}
        hd = q.shape[1] // attrs["heads"]
        default = np.float32(1.0) / np.float32(np.sqrt(np.float32(hd)))
        scale = np.float32(o.options["scale"])
        if scale != default:
            attrs["scale"] = _short(scale)
        if self.mode == "decode":
            if not attrs["causal"]:
                raise SequenceError(f"{self.g.describe(o)}: attention that is not causal sees "
                                    "later positions")
            if q.seq != 0 or k.seq != 0 or v.seq != 0:
                raise SequenceError(f"{self.g.describe(o)}: the keys or values are not of the "
                                    "same sequence (not self-attention)")
            n = len(self.states) // 2
            ks, vs = f"keys_{n}", f"values_{n}"
            self.states += [{"name": ks, "dtype": "float32", "shape": [self.length, k.shape[1]]},
                            {"name": vs, "dtype": "float32", "shape": [self.length, v.shape[1]]}]
            kc = self.node("cache_write", [ks, k.name], k.shape, 0, at="P")
            vc = self.node("cache_write", [vs, v.name], v.shape, 0, at="P")
            k, v = kc, vc
        elif q.seq is not None:
            self.mix("attends across positions")
        res = self.node("attention", [q.name, k.name, v.name], q.shape, q.seq, **attrs)
        self.out(o, res)

    def binary(self, o: Op, name: str) -> None:
        a_t, b_t = o.inputs
        ca, cb = const_value(self.g, a_t), const_value(self.g, b_t)
        if ca is not None and cb is not None:
            raise ConvertError("both inputs constant (not folded)")
        vals: list[Value | None] = [None if c is not None else self.val(t)
                                    for t, c in ((a_t, ca), (b_t, cb))]
        for v in vals:
            if v is not None and v.dtype != "float32":
                raise ConvertError(f"{_BINARY_NAMES.get(o.kind, o.kind)} of int32 values: rayo's "
                                   "elementwise ops are float32")
        act = next(v for v in vals if v is not None)
        out_rank = max(len(rshape(self.sh[a_t])), len(rshape(self.sh[b_t])))
        seq_out = None
        for v in vals:
            if v is not None and v.seq is not None:
                at = v.seq + out_rank - len(v.shape)
                if seq_out is not None and seq_out != at:
                    raise SequenceError(f"{self.g.describe(o)}: two sequence axes")
                seq_out = at
        names = []
        for t, c, v in ((a_t, ca, vals[0]), (b_t, cb, vals[1])):
            if v is not None:
                if seq_out is not None and v.seq is None:
                    at = seq_out - (out_rank - len(v.shape))
                    if 0 <= at < len(v.shape) and v.shape[at] != 1:
                        raise SequenceError(f"{self.g.describe(o)}: an input of the sequence's "
                                            "length that is not the sequence")
                names.append(v.name)
                continue
            assert c is not None
            if c.dtype.kind not in "fb":
                raise ConvertError(f"an {c.dtype} constant in a float op")
            cs = rshape(c.shape)
            value = c.reshape(cs).astype(np.float32)
            at = None if seq_out is None else seq_out - (out_rank - len(cs))
            if at is not None and 0 <= at < len(cs) and cs[at] != 1:
                names.append(self.position_constant(t, value, at))
                continue
            role = "bias" if o.kind in ("ADD", "SUB") and c.size > 1 else "const"
            names.append(self.const_weight(value, role, t) if list(c.shape) != cs
                         else self.weight(t, role))
        shape = list(np.broadcast_shapes(*[tuple(rshape(self.sh[t])) for t in o.inputs]))
        rowwise = 0 if vals[0] is not None and vals[0].seq is not None else (
            1 if vals[1] is not None and vals[1].seq is not None else None)
        v = self.node(name, names, shape, seq_out, rowwise=rowwise)
        self.out(o, v, o.options.get("fused_activation_function", 0))
        del act

    def position_constant(self, t: int, value: np.ndarray, axis: int) -> str:
        """A constant with a row per position (a position embedding): in the sequence modes,
        the rows of the positions run, looked up by position."""
        if axis != 0 or value.shape[0] != self.length:
            raise SequenceError(f"{self.g.describe(self.current)}: a constant along the sequence "
                                f"of shape {list(value.shape)}")
        table = value.reshape(value.shape[0], -1)
        w = self.weight(t, "table") if list(self.g.tensors[t].data.shape) == list(table.shape) \
            else self.const_weight(table, "table", t)
        rows = self.node("gather", [w, self.positions_value()], [self.length, table.shape[1]], 0)
        if list(value.shape) != list(table.shape):
            rows = self.node("reshape", [rows.name], list(value.shape), 0,
                             shape=self.sym(list(value.shape), 0))
        return rows.name

    def softmax(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        if x.seq is not None and x.seq == len(x.shape) - 1:
            self.mix("takes a softmax over positions")
        beta = float(o.options.get("beta", 1.0))
        if beta != 1.0:
            w = self.const_weight(np.asarray([beta], np.float32), "beta", o.outputs[0])
            x = self.node("mul", [x.name, w], x.shape, x.seq, rowwise=0)
        self.out(o, self.node("softmax", [x.name], x.shape, x.seq,
                              rowwise=None if x.seq == len(x.shape) - 1 else 0))

    def gather(self, o: Op) -> None:
        if o.kind == "EMBEDDING_LOOKUP":
            ids_t, table_t = o.inputs
        else:
            table_t, ids_t = o.inputs
            if int(o.options.get("axis", 0)) != 0:
                raise ConvertError("GATHER along an axis other than 0: rayo's gather looks up "
                                   "rows")
        table = self.g.tensors[table_t]
        if not table.const:
            raise ConvertError("GATHER from an activation: rayo's gather looks rows up in a "
                               "weight")
        ids = self.val(ids_t)
        if ids.dtype != "int32" or len(ids.shape) != 1:
            raise ConvertError(f"ids {ids.shape} {ids.dtype}: rayo's gather takes int32 [n]")
        rows = table.shape[0]
        width = int(np.prod(table.shape[1:])) if len(table.shape) > 1 else 1
        if table.dtype == "int8" and table.quant is not None and table.quant.scales.size == 1 \
                and len(table.shape) == 2:
            w = self.weight(table_t, "table")
        else:
            real = table.real() if table.dtype == "int8" else table.data.astype(np.float32)
            if table.dtype == "int8":
                self.notes.add(f"{table.name}: an int8 table with a scale per row is stored as "
                               "float32 (rayo's gather takes one scale)")
            if real.dtype.kind != "f":
                raise ConvertError("GATHER of an integer table: rayo looks up float rows")
            w = (self.weight(table_t, "table") if table.dtype != "int8" and len(table.shape) == 2
                 else self.const_weight(real.reshape(rows, width), "table", table_t))
        v = self.node("gather", [w, ids.name], [ids.shape[0], width], ids.seq, rowwise=1)
        self.out(o, v)

    def concat(self, o: Op) -> None:
        out_shape = rshape(self.sh[o.outputs[0]])
        rank_src = len(self.sh[o.outputs[0]])
        axis = int(o.options.get("axis", 0)) % rank_src
        if axis != rank_src - 1:
            raise ConvertError("CONCATENATION along an axis other than the innermost: rayo's "
                               "concat joins innermost dimensions")
        vals = []
        for t in o.inputs:
            if self.g.tensors[t].const:
                raise ConvertError("CONCATENATION with a constant: rayo's concat joins "
                                   "activations")
            v = self.val(t)
            want = [*out_shape[:-1], rshape(self.sh[t])[-1]] if len(out_shape) > 1 else \
                [rshape(self.sh[t])[-1]]
            vals.append(self.fit(v, want))
        seqs = {v.seq for v in vals}
        if len(seqs) > 1:
            raise SequenceError(f"{self.g.describe(o)} joins a sequence with a non-sequence")
        seq = seqs.pop()
        if seq is not None and seq == len(out_shape) - 1:
            raise SequenceError(f"{self.g.describe(o)} joins along the sequence")
        cur = vals[0]
        for v in vals[1:]:
            shape = [*cur.shape[:-1], cur.shape[-1] + v.shape[-1]]
            cur = self.node("concat", [cur.name, v.name], shape, seq)
        self.out(o, cur, o.options.get("fused_activation_function", 0))

    def transpose(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        perm = [int(p) for p in self.g.tensors[o.inputs[1]].data.reshape(-1)]
        src = self.sh[o.inputs[0]]
        lead = len(src) - len(x.shape)
        keep = [p - lead for p in perm if p >= lead]
        if len(keep) != len(x.shape):
            raise ConvertError("TRANSPOSE that rayo's shapes do not follow")
        shape = [x.shape[p] for p in keep]
        seq = keep.index(x.seq) if x.seq is not None else None
        if len(shape) > 4:
            raise ConvertError("rank above 4")
        if keep == list(range(len(keep))):
            v = x
        else:
            v = self.node("transpose", [x.name], shape, seq, x.dtype, perm=keep)
        want = rshape(self.sh[o.outputs[0]])
        if len(want) > 4:
            raise ConvertError(f"a tensor of rank {len(want)} {want}: rayo holds at most 4 "
                               "dimensions (RAYO_MAX_RANK)")
        self.vals[o.outputs[0]] = self.fit(v, want)

    def _slice_value(self, x: Value, src: tuple[int, ...], spec: list[slice]) -> Value:
        lead = len(src) - len(x.shape)
        begin, size = [], []
        for i, s in enumerate(spec):
            b, e, st = s.indices(src[i])
            if st != 1:
                raise ConvertError("a strided slice with a step: rayo's slice takes every element")
            n = max(e - b, 0)
            if i < lead:
                continue
            ax = i - lead
            if ax == x.seq:
                if b != 0 or n != src[i]:
                    raise SequenceError(f"{self.g.describe(self.current)} cuts the sequence")
                begin.append(0)
                size.append(-1)
            else:
                begin.append(b)
                size.append(n)
        shape = [x.shape[i] if size[i] == -1 else size[i] for i in range(len(size))]
        if begin == [0] * len(begin) and shape == x.shape:
            return x
        return self.node("slice", [x.name], shape, x.seq, x.dtype, begin=begin, size=size,
                         rowwise=0)

    def slice(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        src = self.sh[o.inputs[0]]
        vals = [self.g.tensors[i].data for i in o.inputs[1:]]
        if any(v is None for v in vals):
            raise ConvertError("a slice whose bounds are not constants")
        if o.kind == "SLICE":
            b = [int(v) for v in vals[0].reshape(-1)]
            s = [int(v) for v in vals[1].reshape(-1)]
            spec = [slice(bi, None if si == -1 else bi + si) for bi, si in zip(b, s, strict=True)]
        else:
            spec, _ = ref.strided_slice_spec(src, *[[int(v) for v in a.reshape(-1)] for a in vals],
                                             o.options)
        v = self._slice_value(x, src, spec)
        self.vals[o.outputs[0]] = self.fit(v, rshape(self.sh[o.outputs[0]]))

    def split(self, o: Op) -> None:
        if o.kind == "SPLIT":
            axis_t, x_t = o.inputs
        else:
            x_t = o.inputs[0]
            axis_t = o.inputs[2] if o.kind == "SPLIT_V" else -1
        x = self.val(x_t)
        src = self.sh[x_t]
        axis = (int(o.options.get("axis", 0)) if axis_t < 0 else
                int(self.g.tensors[axis_t].data.reshape(-1)[0])) % len(src)
        at = 0
        for t in o.outputs:
            n = self.sh[t][axis]
            spec = [slice(None)] * len(src)
            spec[axis] = slice(at, at + n)
            at += n
            v = self._slice_value(x, src, spec)
            self.vals[t] = self.fit(v, rshape(self.sh[t]))

    def reduce(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        src = self.sh[o.inputs[0]]
        axes = sorted({int(a) % len(src) for a in self.g.tensors[o.inputs[1]].data.reshape(-1)})
        lead = len(src) - len(x.shape)
        cur = x
        how = _REDUCE[o.kind]
        for a in reversed(axes):
            if a < lead:
                continue
            ax = a - lead
            if ax == x.seq:
                self.mix("reduces over positions")
            shape = list(cur.shape)
            shape[ax] = 1
            seq = None if ax == cur.seq else cur.seq
            cur = self.node("reduce", [cur.name], shape, seq, reduction=how, axis=ax, keepdims=1,
                            rowwise=None if ax == cur.seq else 0)
        self.vals[o.outputs[0]] = self.fit(cur, rshape(self.sh[o.outputs[0]]))

    def matmul(self, o: Op) -> None:
        a_t, b_t = o.inputs
        a = self.val(a_t)
        attrs: dict[str, Any] = {}
        if o.options.get("adj_x"):
            attrs["transpose_a"] = 1
        if o.options.get("adj_y"):
            attrs["transpose_b"] = 1
        if len(a.shape) < 2:
            a = self.fit(a, [1, *a.shape])
        cb = const_value(self.g, b_t)
        if cb is not None:
            bt = self.g.tensors[b_t]
            if bt.dtype == "int8":
                raise ConvertError("BATCH_MATMUL with an int8 weight: TFLite's hybrid kernel "
                                   "for it is not rayo's; write the layer as a dense")
            real = cb.astype(np.float32)
            b_name = self.const_weight(real.reshape(rshape(real.shape)), "kernel", b_t)
            b = Value(b_name, rshape(real.shape))
        else:
            b = self.val(b_t)
        if len(b.shape) < 2:
            b = self.fit(b, [*b.shape, 1])
        ar = [a.shape[-1], a.shape[-2]] if attrs.get("transpose_a") else a.shape[-2:]
        br = [b.shape[-1], b.shape[-2]] if attrs.get("transpose_b") else b.shape[-2:]
        batch = list(np.broadcast_shapes(tuple(a.shape[:-2]), tuple(b.shape[:-2])))
        shape = [*batch, ar[0], br[1]]
        seq = None
        rank = len(shape)
        for v, is_a in ((a, True), (b, False)):
            if v.seq is None:
                continue
            pos = v.seq + rank - len(v.shape)
            rows = pos == rank - 2 and is_a and not attrs.get("transpose_a")
            cols = pos == rank - 1 and not is_a and not attrs.get("transpose_b")
            if pos < rank - 2 or rows or cols:
                if seq is not None and seq != pos:
                    raise SequenceError(f"{self.g.describe(o)}: scores between positions")
                seq = pos
            else:
                self.mix("sums over positions")
                raise SequenceError(f"{self.g.describe(o)}: a product over the sequence axis")
        v = self.node("matmul", [a.name, b.name], shape, seq, **attrs)
        self.out(o, v)

    def fc_matmul(self, o: Op) -> None:
        x = self.val(o.inputs[0])
        w = self.val(o.inputs[1])
        v = self.node("matmul", [x.name, w.name], [*x.shape[:-1], w.shape[0]], x.seq,
                      transpose_b=1)
        if len(o.inputs) > 2 and o.inputs[2] >= 0:
            bias = self.weight(o.inputs[2], "bias")
            v = self.node("add", [v.name, bias], v.shape, v.seq, rowwise=0)
        self.out(o, v, o.options.get("fused_activation_function", 0))

    def compare(self, o: Op) -> None:
        a_t, b_t = o.inputs
        how = _COMPARE[o.kind]
        ca, cb = const_value(self.g, a_t), const_value(self.g, b_t)
        if cb is not None and uniform(cb) is not None:
            x, value = self.val(a_t), uniform(cb)
        elif ca is not None and uniform(ca) is not None:
            x, value, how = self.val(b_t), uniform(ca), _FLIP[how]
        else:
            raise ConvertError(f"{o.kind} between two tensors: rayo's compare tests against a "
                               "number")
        v = self.node("compare", [x.name], x.shape, x.seq, comparison=how, value=value,
                      rowwise=0)
        self.vals[o.outputs[0]] = self.fit(v, rshape(self.sh[o.outputs[0]]))

    def select(self, o: Op) -> None:
        """SELECT with a constant condition and one constant branch: x * m + c, with m the
        condition as 0/1 (1 where x is taken) and c the constant where x is not."""
        cond, a_t, b_t = o.inputs
        cc = const_value(self.g, cond)
        ca, cb = const_value(self.g, a_t), const_value(self.g, b_t)
        if cc is None or (ca is None) == (cb is None):
            raise ConvertError("SELECT of two activations, or by a computed condition: rayo has "
                               "no select")
        take = np.asarray(cc, bool)
        x_t, c = (a_t, cb) if cb is not None else (b_t, ca)
        m = take if x_t == a_t else ~take
        c = np.asarray(c, np.float32)
        if not np.all(np.isfinite(c)):
            self.notes.add(f"{self.g.describe(o)}: infinite constants stored as the largest "
                           "float")
            c = np.nan_to_num(c, posinf=3.4028235e38, neginf=-3.4028235e38)
        out = rshape(self.sh[o.outputs[0]])
        mw = np.broadcast_to(m, self.sh[o.outputs[0]]).astype(np.float32)
        cw = np.where(np.broadcast_to(m, self.sh[o.outputs[0]]), 0.0, np.broadcast_to(
            c, self.sh[o.outputs[0]])).astype(np.float32)
        x = self.val(x_t)
        lead = len(self.sh[o.outputs[0]]) - len(out)
        mw = mw.reshape(mw.shape[lead:]) if lead else mw
        cw = cw.reshape(cw.shape[lead:]) if lead else cw
        mw, cw = _squeeze_broadcast(mw, x.shape), _squeeze_broadcast(cw, x.shape)
        if x.seq is not None and (mw.shape[x.seq - (len(out) - mw.ndim)] != 1 if
                                  x.seq - (len(out) - mw.ndim) >= 0 else False):
            raise SequenceError(f"{self.g.describe(o)}: a mask along the sequence")
        y = self.node("mul", [x.name, self.const_weight(mw, "mask", cond)], out, x.seq,
                      rowwise=0)
        y = self.node("add", [y.name, self.const_weight(cw, "fill", cond)], out, x.seq,
                      rowwise=0)
        self.vals[o.outputs[0]] = y

    def conv2d_refusal(self, o: Op) -> None:
        x = self.sh[o.inputs[0]]
        k = self.g.tensors[o.inputs[1]]
        opt = o.options
        if len(x) == 4 and x[1] != 1:
            raise ConvertError(f"CONV_2D over a {x[1]}x{x[2]} image: rayo convolves sequences "
                               "only (conv1d)")
        if int(opt.get("stride_w", 1)) != 1:
            raise ConvertError("CONV_2D with a stride: rayo's conv1d has none")
        if int(opt.get("padding", 0)) == 1:
            raise ConvertError("CONV_2D with valid padding: rayo's conv1d pads the same")
        if k.dtype == "int8" and int(opt.get("dilation_w_factor", 1)) > 1:
            raise ConvertError("an int8 CONV_2D with dilation: TFLite quantises its whole input, "
                               "rayo's dilated conv1d each phase (as TFLite does for the "
                               "SPACE_TO_BATCH_ND form)")
        raise ConvertError("CONV_2D that is not a same-padded 1-D convolution with an odd reach")


_BINARY_NAMES = {k: k for k in _BINARY}


def why_not(kind: str) -> str:
    if kind.startswith("CUSTOM:"):
        return (f"the custom op {kind[7:]}: write it as a rayo op of your own "
                "(include/rayo/op.h) and pass it to rayo_model_create_with")
    return _WHY.get(kind, f"{kind}: rayo has no op for it")


def _squeeze_broadcast(v: np.ndarray, like: list[int]) -> np.ndarray:
    """``v`` with the dimensions it repeats along taken back to 1 (a smaller weight)."""
    for ax in range(v.ndim):
        first = np.take(v, [0], axis=ax)
        if v.shape[ax] > 1 and np.all(v == first):
            v = first
    return v


def _short(x: Any) -> float:
    """A float32 as the shortest decimal that reads back as it."""
    return float(repr(np.float32(x)).removeprefix("np.float32(").removesuffix(")"))

_WHY = {
    "SPACE_TO_BATCH_ND": "SPACE_TO_BATCH_ND outside a dilated convolution: rayo has no op for it",
    "BATCH_TO_SPACE_ND": "BATCH_TO_SPACE_ND outside a dilated convolution: rayo has no op for it",
    "DEPTHWISE_CONV_2D": "DEPTHWISE_CONV_2D: rayo has no depthwise convolution",
    "AVERAGE_POOL_2D": "AVERAGE_POOL_2D: rayo has no 2-D pooling",
    "MAX_POOL_2D": "MAX_POOL_2D: rayo has no 2-D pooling",
    "QUANTIZE": "QUANTIZE: an int8 activation; rayo runs float32 activations with int8 weights "
                "(dynamic-range quantisation), not fully quantised models",
    "SELECT": "SELECT: rayo has no select",
    "SELECT_V2": "SELECT_V2: rayo has no select",
    "PAD": "PAD: rayo has no pad",
    "PADV2": "PADV2: rayo has no pad",
    "UNIDIRECTIONAL_SEQUENCE_LSTM": "an LSTM: rayo has no recurrent ops",
    "UNIDIRECTIONAL_SEQUENCE_RNN": "an RNN: rayo has no recurrent ops",
    "WHILE": "WHILE: rayo runs one graph of ops in order, without loops",
}
