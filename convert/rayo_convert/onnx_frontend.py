"""Reads an ONNX model (float32; opset 13 or later) into the converter's graph, with the
``onnx`` package. ONNX ops become their TFLite equivalents, so the same passes and lowering
serve both; a few with no TFLite twin (ONNX's Reshape, Slice, Squeeze, ...) keep kinds of
their own (``ONNX_*``), which the reference and the lowering know.

Layouts: ONNX's 1-D Conv is channels first ([N, C, T]); it becomes a transpose, TFLite's
CONV_2D on one row, and a transpose back, so rayo's conv1d ([T, C]) runs it.

Names: a weight is named after its layer, from the initializer's name when it is a PyTorch
parameter's (``conv_1.weight`` is layer ``conv_1``), otherwise from the scope PyTorch's exporter
gives the node reading it (``/hidden/MatMul`` is layer ``hidden``). So a torch module whose
layers are named as the Keras model's were gives a file with the same weight names
(``conv_1.kernel``, ``hidden.bias``, ``token_embedding.table``, ``ln.gamma``).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from .ir import ConvertError, Graph, Op, Tensor

_BINARY = {"Add": "ADD", "Sub": "SUB", "Mul": "MUL", "Div": "DIV", "Max": "MAXIMUM",
           "Min": "MINIMUM", "Pow": "POW"}
_UNARY = {"Relu": "RELU", "Tanh": "TANH", "Sigmoid": "LOGISTIC", "Exp": "EXP", "Log": "LOG",
          "Sqrt": "SQRT", "Neg": "NEG", "Abs": "ABS", "Erf": "ONNX_ERF", "Floor": "FLOOR",
          "Ceil": "CEIL", "Not": "LOGICAL_NOT"}
_COMPARE = {"Equal": "EQUAL", "Greater": "GREATER", "Less": "LESS",
            "GreaterOrEqual": "GREATER_EQUAL", "LessOrEqual": "LESS_EQUAL"}
_REDUCE = {"ReduceMean": "MEAN", "ReduceSum": "SUM", "ReduceMax": "REDUCE_MAX",
           "ReduceMin": "REDUCE_MIN", "ReduceProd": "REDUCE_PROD"}
_SHAPELIKE = {"Reshape": "ONNX_RESHAPE", "Flatten": "ONNX_FLATTEN", "Squeeze": "ONNX_SQUEEZE",
              "Unsqueeze": "ONNX_UNSQUEEZE", "Identity": "IDENTITY", "Dropout": "IDENTITY"}
_ONNX_DTYPES = {1: "float32", 6: "int32", 7: "int64", 9: "bool", 3: "int8", 11: "float64",
                10: "float16"}


def read(raw: bytes) -> Graph:
    try:
        import onnx
        from onnx import numpy_helper, shape_inference
    except ImportError as e:
        raise ConvertError("ONNX models need the onnx package: pip install "
                           "'rayo-convert[onnx]'") from e
    try:
        model = onnx.load_from_string(raw)
    except Exception as e:
        raise ConvertError(f"not an ONNX model: {e}") from e
    with contextlib.suppress(Exception):  # shapes are also found by running the graph later
        model = shape_inference.infer_shapes(model)
    og = model.graph
    tensors: list[Tensor] = []
    by_name: dict[str, int] = {}
    ranks: dict[str, int] = {}

    def type_of(vi: Any) -> tuple[str, tuple[int, ...] | None]:
        tt = vi.type.tensor_type
        dt = _ONNX_DTYPES.get(tt.elem_type, f"onnx type {tt.elem_type}")
        if not tt.HasField("shape"):
            return dt, None
        dims = tuple(int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else 1
                     for d in tt.shape.dim)
        return dt, dims

    for vi in [*og.input, *og.value_info, *og.output]:
        _, shape = type_of(vi)
        if shape is not None:
            ranks[vi.name] = len(shape)

    def tensor(name: str, dtype: str = "float32", shape: tuple[int, ...] = (),
               data: np.ndarray | None = None) -> int:
        t = Tensor(len(tensors), name, dtype, tuple(shape), data)
        tensors.append(t)
        if name:
            by_name[name] = t.index
        return t.index

    def const(name: str, value: Any) -> int:
        v = np.asarray(value)
        if v.dtype == np.float64:
            v = v.astype(np.float32)
        return tensor(name, v.dtype.name, v.shape, v)

    hints: dict[int, str] = {}
    for init in og.initializer:
        v = numpy_helper.to_array(init)
        if v.dtype == np.float64:
            v = v.astype(np.float32)
        i = tensor(init.name, v.dtype.name, v.shape, v)
        ranks[init.name] = v.ndim
        if _param_layer(init.name):
            hints[i] = _param_layer(init.name)
    inputs = []
    in_names: dict[int, str] = {}
    for vi in og.input:
        if vi.name in by_name:
            continue  # an initializer listed as an input
        dt, shape = type_of(vi)
        if dt.startswith("onnx type"):
            raise ConvertError(f"input {vi.name}: {dt} is not supported")
        i = tensor(vi.name, dt, shape or ())
        inputs.append(i)
        in_names[i] = vi.name
    ops: list[Op] = []

    def attr(node: Any, name: str, default: Any = None) -> Any:
        for a in node.attribute:
            if a.name == name:
                return onnx.helper.get_attribute_value(a)
        return default

    def out(node: Any, k: int = 0, dtype: str = "float32") -> int:
        name = node.output[k]
        return tensor(name, dtype)

    def inp(node: Any, k: int) -> int:
        if k >= len(node.input) or not node.input[k]:
            return -1
        name = node.input[k]
        if name not in by_name:
            raise ConvertError(f"node {node.name or node.op_type}: no input {name}")
        return by_name[name]

    def rank(name: str) -> int | None:
        return ranks.get(name)

    for idx, node in enumerate(og.node):
        t = node.op_type
        ins = [inp(node, k) for k in range(len(node.input))]
        dt_in = tensors[ins[0]].dtype if ins and ins[0] >= 0 else "float32"
        layer = _scope(node.name)
        for i in ins:
            if i >= 0 and tensors[i].const and i not in hints and layer:
                hints[i] = layer
        made = len(tensors)  # constants made from this node's inputs are named as its layer

        def add(kind: str, inputs: list[int], outputs: list[int], at: int = idx,
                **options: Any) -> None:
            ops.append(Op(at, kind, inputs, outputs, options))

        if t == "Constant":
            v = attr(node, "value")
            if v is None:
                for key in ("value_float", "value_int", "value_floats", "value_ints"):
                    if attr(node, key) is not None:
                        v = np.asarray(attr(node, key))
                        break
            else:
                v = numpy_helper.to_array(v)
            const(node.output[0], v)
        elif t in _BINARY:
            add(_BINARY[t], ins[:2], [out(node, 0, dt_in)])
        elif t in _UNARY:
            add(_UNARY[t], ins, [out(node, 0, "bool" if t == "Not" else dt_in)])
        elif t == "Gelu":
            add("GELU", ins, [out(node)], approximate=attr(node, "approximate", b"none")
                in (b"tanh", "tanh"))
        elif t in _COMPARE:
            add(_COMPARE[t], ins, [out(node, 0, "bool")])
        elif t in ("And", "Or"):
            add("LOGICAL_AND" if t == "And" else "LOGICAL_OR", ins, [out(node, 0, "bool")])
        elif t == "Where":
            add("SELECT_V2", ins, [out(node, 0, tensors[ins[1]].dtype)])
        elif t == "Clip":  # PyTorch's clamp: a maximum and a minimum
            x = ins[0]
            for k, kind in ((1, "MAXIMUM"), (2, "MINIMUM")):
                if k < len(ins) and ins[k] >= 0:
                    last = not any(j < len(ins) and ins[j] >= 0 for j in range(k + 1, 3))
                    y = out(node, 0, dt_in) if last else tensor(f"{node.output[0]}/{kind}",
                                                                dt_in)
                    add(kind, [x, ins[k]], [y])
                    x = y
            if x == ins[0]:
                add("IDENTITY", [x], [out(node, 0, dt_in)])
        elif t == "Softmax":
            axis = int(attr(node, "axis", -1))
            r = rank(node.input[0])
            if axis != -1 and (r is None or axis != r - 1):
                raise ConvertError(f"node {idx} (Softmax): over axis {axis}; rayo's softmax is "
                                   "over the innermost")
            add("SOFTMAX", ins, [out(node)], beta=1.0)
        elif t == "MatMul":
            b = tensors[ins[1]]
            if b.const and b.data.ndim == 2:
                k = const(node.input[1] + "/kernel", b.data.T.copy())
                add("FULLY_CONNECTED", [ins[0], k, -1], [out(node)], keep_num_dims=True,
                    asymmetric_quantize_inputs=True)
            else:
                add("BATCH_MATMUL", ins, [out(node)])
        elif t == "Gemm":
            a, b = tensors[ins[0]], tensors[ins[1]]
            alpha, beta = float(attr(node, "alpha", 1.0)), float(attr(node, "beta", 1.0))
            if attr(node, "transA", 0) or not b.const:
                raise ConvertError(f"node {idx} (Gemm): only x · W with W a constant")
            w = b.data if attr(node, "transB", 0) else b.data.T
            k = const(node.input[1] + "/kernel", (w * alpha).astype(np.float32))
            bias = -1
            if len(ins) > 2 and ins[2] >= 0:
                c = tensors[ins[2]]
                if not c.const:
                    raise ConvertError(f"node {idx} (Gemm): a bias that is not a constant")
                bias = const(node.input[2] + "/bias",
                             np.broadcast_to(c.data * beta, (w.shape[0],)).astype(np.float32))
            del a
            add("FULLY_CONNECTED", [ins[0], k, bias], [out(node)], keep_num_dims=True,
                asymmetric_quantize_inputs=True)
        elif t == "LayerNormalization":
            if int(attr(node, "axis", -1)) not in (-1, (rank(node.input[0]) or 0) - 1):
                raise ConvertError(f"node {idx} (LayerNormalization): over more than the "
                                   "innermost axis")
            add("LAYER_NORM", ins[:3] + [-1] * (3 - len(ins[:3])), [out(node)],
                epsilon=float(attr(node, "epsilon", 1e-5)))
        elif t == "Gather":
            add("GATHER", ins, [out(node, 0, tensors[ins[0]].dtype)], axis=int(attr(node, "axis",
                                                                                     0)))
        elif t == "Concat":
            add("CONCATENATION", ins, [out(node, 0, dt_in)], axis=int(attr(node, "axis")))
        elif t in _SHAPELIKE:
            kind = _SHAPELIKE[t]
            opts: dict[str, Any] = {}
            if t in ("Squeeze", "Unsqueeze") and attr(node, "axes") is not None:
                opts["axes"] = [int(a) for a in attr(node, "axes")]
            if t == "Flatten":
                opts["axis"] = int(attr(node, "axis", 1))
            if t == "Reshape":
                opts["allowzero"] = int(attr(node, "allowzero", 0))
            ins_used = ins[:1] if t in ("Identity", "Dropout") else ins
            add(kind, ins_used, [out(node, 0, dt_in)], **opts)
        elif t == "Transpose":
            perm = attr(node, "perm")
            if perm is None:
                r = rank(node.input[0])
                if r is None:
                    raise ConvertError(f"node {idx} (Transpose): no perm and no known rank")
                perm = list(reversed(range(r)))
            p = const(f"{node.output[0]}/perm", np.asarray(perm, np.int32))
            add("TRANSPOSE", [ins[0], p], [out(node, 0, dt_in)])
        elif t == "Slice":
            add("ONNX_SLICE", ins, [out(node, 0, dt_in)])
        elif t == "Split":
            opts = {"axis": int(attr(node, "axis", 0))}
            if attr(node, "split") is not None:
                opts["split"] = [int(s) for s in attr(node, "split")]
            if attr(node, "num_outputs") is not None:
                opts["num_outputs"] = int(attr(node, "num_outputs"))
            add("ONNX_SPLIT", ins, [out(node, k, dt_in) for k in range(len(node.output))],
                **opts)
        elif t in _REDUCE:
            axes = attr(node, "axes")
            keep = int(attr(node, "keepdims", 1))
            if axes is None and len(ins) > 1 and ins[1] >= 0:
                if not tensors[ins[1]].const:
                    raise ConvertError(f"node {idx} ({t}): axes that are not constants")
                ax = ins[1]
            else:
                if axes is None:
                    r = rank(node.input[0])
                    if r is None:
                        raise ConvertError(f"node {idx} ({t}): over every axis of an unknown "
                                           "rank")
                    axes = list(range(r))
                ax = const(f"{node.output[0]}/axes", np.asarray(axes, np.int32))
            add(_REDUCE[t], [ins[0], ax], [out(node)], keep_dims=bool(keep))
        elif t == "Cast":
            to = _ONNX_DTYPES.get(int(attr(node, "to")), None)
            if to is None:
                raise ConvertError(f"node {idx} (Cast): to an unsupported type")
            add("CAST", ins, [out(node, 0, to)])
        elif t == "Shape":
            add("SHAPE", ins, [out(node, 0, "int64")])
        elif t == "ConstantOfShape":
            v = attr(node, "value")
            v = numpy_helper.to_array(v) if v is not None else np.zeros(1, np.float32)
            add("ONNX_CONSTANT_OF_SHAPE", ins, [out(node, 0, v.dtype.name)],
                value=v.reshape(-1)[0].item())
        elif t == "Range":
            add("RANGE", ins, [out(node, 0, dt_in)])
        elif t == "Expand":
            add("ONNX_EXPAND", ins, [out(node, 0, dt_in)])
        elif t == "Trilu":
            add("ONNX_TRILU", ins, [out(node, 0, dt_in)], upper=int(attr(node, "upper", 1)))
        elif t == "Tile":
            add("TILE", ins, [out(node, 0, dt_in)])
        elif t == "Conv":
            _conv(node, idx, ins, tensors, const, out, add, attr)
        elif t == "BatchNormalization":
            x, scale, bias, mean, var = (tensors[i] for i in ins[:5])
            if not all(v.const for v in (scale, bias, mean, var)):
                raise ConvertError(f"node {idx} (BatchNormalization): training statistics")
            r = rank(node.input[0]) or 3
            eps = float(attr(node, "epsilon", 1e-5))
            a = scale.data / np.sqrt(var.data + eps)
            c = bias.data - mean.data * a
            shape = (-1,) + (1,) * (r - 2)
            ka = const(node.output[0] + "/scale", a.reshape(shape).astype(np.float32))
            kc = const(node.output[0] + "/shift", c.reshape(shape).astype(np.float32))
            mid = tensor(node.output[0] + "/scaled")
            add("MUL", [ins[0], ka], [mid])
            add("ADD", [mid, kc], [out(node)])
        else:
            raise ConvertError(f"node {idx} ({t}, output {node.output[0]!r}): the ONNX op {t} "
                               "is not supported")
        source = next((hints[i] for i in ins[1:] if i in hints), None) or layer
        for i in range(made, len(tensors)):
            if tensors[i].const and source:
                hints.setdefault(i, source)
    outputs, out_names = [], {}
    for vo in og.output:
        if vo.name not in by_name:
            raise ConvertError(f"output {vo.name}: not computed by the graph")
        outputs.append(by_name[vo.name])
        out_names[by_name[vo.name]] = vo.name
    g = Graph(tensors, ops, inputs, outputs, in_names, out_names, "onnx", hints)
    _fuse_erf_gelu(g)
    return g


def _param_layer(name: str) -> str | None:
    """The layer of a PyTorch parameter's name (``attention_0.query.weight``:
    ``attention_0.query``), or None for a name PyTorch's exporter made (``onnx::MatMul_7``)."""
    if "::" in name or "/" in name or "." not in name:
        return None
    return name.rsplit(".", 1)[0] or None


def _scope(node_name: str) -> str | None:
    """The layer of a node PyTorch's exporter named by its module path (``/hidden/MatMul``:
    ``hidden``; ``/attention_0/query/MatMul``: ``attention_0.query``)."""
    if not node_name.startswith("/"):
        return None
    parts = [p for p in node_name.split("/")[1:-1] if p]
    return ".".join(parts) or None


def _conv(node: Any, idx: int, ins: list[int], tensors: list[Tensor], const: Any, out: Any,
          add: Any, attr: Any) -> None:
    """A 1-D Conv ([N, C, T], weight [O, C, k]) as TFLite's form on [N, T, C]."""
    w = tensors[ins[1]]
    if not w.const or w.data.ndim != 3:
        raise ConvertError(f"node {idx} (Conv): only 1-D convolutions with constant weights")
    if int(attr(node, "group", 1)) != 1:
        raise ConvertError(f"node {idx} (Conv): grouped convolutions are not supported")
    k = w.data.shape[2]
    dil = int((attr(node, "dilations") or [1])[0])
    if list(attr(node, "strides") or [1]) != [1]:
        raise ConvertError(f"node {idx} (Conv): a stride; rayo's conv1d has none")
    pads = list(attr(node, "pads") or [0, 0])
    if attr(node, "auto_pad", b"NOTSET") in (b"SAME_UPPER", b"SAME_LOWER"):
        pads = [(k - 1) * dil // 2] * 2
    if pads != [(k - 1) * dil // 2] * 2 or (k - 1) * dil % 2:
        raise ConvertError(f"node {idx} (Conv): pads {pads}; rayo's conv1d pads the same, "
                           f"{(k - 1) * dil // 2} on each side")
    name = node.output[0]
    perm = const(name + "/to_tc", np.asarray([0, 2, 1], np.int32))
    xt = len(tensors)
    tensors.append(Tensor(xt, name + "/tc", "float32", ()))
    add("TRANSPOSE", [ins[0], perm], [xt])
    ax = const(name + "/dim", np.asarray(1, np.int32))
    x4 = len(tensors)
    tensors.append(Tensor(x4, name + "/row", "float32", ()))
    add("EXPAND_DIMS", [xt, ax], [x4])
    kern = const(name + "/kernel", np.transpose(w.data, (0, 2, 1))[:, None, :, :].astype(
        np.float32))
    bias = -1
    if len(ins) > 2 and ins[2] >= 0:
        bias = ins[2]
    y4 = len(tensors)
    tensors.append(Tensor(y4, name + "/conv", "float32", ()))
    add("CONV_2D", [x4, kern, bias], [y4], padding=0, stride_w=1, stride_h=1,
        dilation_w_factor=dil, dilation_h_factor=1)
    y3 = len(tensors)
    tensors.append(Tensor(y3, name + "/to", "float32", ()))
    add("ONNX_SQUEEZE", [y4], [y3], axes=[1])
    perm2 = const(name + "/to_ct", np.asarray([0, 2, 1], np.int32))
    add("TRANSPOSE", [y3, perm2], [out(node)])


def _fuse_erf_gelu(g: Graph) -> None:
    """GELU as PyTorch exports it: x * 0.5 * (1 + erf(x / sqrt(2))) (in any order of the
    multiplications), to one GELU."""
    prod = g.producer()
    users = g.consumers()

    def c(t: int) -> float | None:
        v = g.tensors[t].data if t >= 0 else None
        if v is None or v.size != 1:
            return None
        return float(v.reshape(-1)[0])

    def other(o: Op, t: int) -> int:
        return o.inputs[1] if o.inputs[0] == t else o.inputs[0]

    for erf in [o for o in g.ops if o.kind == "ONNX_ERF"]:
        pre = prod.get(erf.inputs[0])
        if pre is None:
            continue
        if pre.kind == "DIV" and c(pre.inputs[1]) is not None and abs(
                c(pre.inputs[1]) - 2 ** 0.5) < 1e-4:
            x = pre.inputs[0]
        elif pre.kind == "MUL" and c(other(pre, pre.inputs[0])) is not None and abs(
                c(other(pre, pre.inputs[0]) if g.tensors[pre.inputs[1]].const else pre.inputs[0])
                - 2 ** -0.5) < 1e-4:
            x = pre.inputs[0] if g.tensors[pre.inputs[1]].const else pre.inputs[1]
        else:
            continue
        chain = [pre, erf]
        nxt = users.get(erf.outputs[0], [])
        if len(nxt) != 1 or nxt[0].kind != "ADD" or c(other(nxt[0], erf.outputs[0])) != 1.0:
            continue
        add1 = nxt[0]
        chain.append(add1)
        t = add1.outputs[0]
        half = False
        mulx = False
        for _ in range(2):
            u = users.get(t, [])
            if len(u) != 1 or u[0].kind != "MUL":
                break
            o2 = other(u[0], t)
            if o2 == x and not mulx:
                mulx = True
            elif c(o2) == 0.5 and not half:
                half = True
            else:
                # x * 0.5 computed first: (x * 0.5) * (1 + erf)
                p2 = prod.get(o2)
                if p2 and p2.kind == "MUL" and x in p2.inputs and c(other(p2, x)) == 0.5 \
                        and not mulx and not half:
                    mulx = half = True
                    chain.append(p2)
                else:
                    break
            chain.append(u[0])
            t = u[0].outputs[0]
            if mulx and half:
                break
        if not (mulx and half):
            continue
        last = chain[-1]
        gelu = Op(erf.index, "GELU", [x], list(last.outputs), {"approximate": False})
        at = g.ops.index(last)
        g.ops[at] = gelu
        for o in chain[:-1]:
            if o in g.ops and all(u in chain for u in users.get(o.outputs[0], [])):
                g.ops.remove(o)
        prod = g.producer()
        users = g.consumers()
