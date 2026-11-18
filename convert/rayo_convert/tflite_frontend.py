"""Reads a TFLite flatbuffer into the converter's graph (rayo_convert.ir), with the ``tflite``
package (the flatbuffers code generated from TFLite's schema); no TensorFlow."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import tflite

from .ir import ConvertError, Graph, Op, Quant, Tensor

_DTYPES = {
    tflite.TensorType.FLOAT32: "float32",
    tflite.TensorType.FLOAT16: "float16",
    tflite.TensorType.INT32: "int32",
    tflite.TensorType.UINT8: "uint8",
    tflite.TensorType.INT64: "int64",
    tflite.TensorType.BOOL: "bool",
    tflite.TensorType.INT16: "int16",
    tflite.TensorType.INT8: "int8",
    tflite.TensorType.FLOAT64: "float64",
}

_OPS = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
_OPTIONS = {v: k for k, v in vars(tflite.BuiltinOptions).items() if not k.startswith("_")}
_SKIP = {"Init", "GetRootAs"}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _options(op: Any) -> dict[str, Any]:
    """An operator's builtin options as a dict of snake_case names."""
    kind = _OPTIONS.get(op.BuiltinOptionsType(), "NONE")
    table = op.BuiltinOptions()
    if kind == "NONE" or table is None:
        return {}
    cls = getattr(tflite, kind, None)
    if cls is None:
        return {}
    obj = cls()
    obj.Init(table.Bytes, table.Pos)
    out: dict[str, Any] = {}
    names = [n for n in dir(cls) if not n.startswith("_") and n[0].isupper()]
    for n in names:
        if n in _SKIP or n.startswith("GetRootAs") or n.endswith("BufferHasIdentifier"):
            continue
        if n.endswith(("Length", "IsNone", "AsNumpy")):
            continue
        if n + "AsNumpy" in names:
            v = getattr(obj, n + "AsNumpy")()
            out[_snake(n)] = [] if isinstance(v, int) else [int(x) for x in v]
            continue
        try:
            v = getattr(obj, n)()
        except TypeError:
            continue
        out[_snake(n)] = v
    return out


def _buffer(model: Any, raw: bytes, index: int) -> bytes | None:
    b = model.Buffers(index)
    if b is None:
        return None
    if b.DataLength() > 0:
        return b.DataAsNumpy().tobytes()
    off, size = b.Offset(), b.Size()
    if off > 1 and size > 0:
        return raw[off : off + size]
    return None


def read(raw: bytes) -> Graph:
    """The first subgraph of the TFLite model ``raw``."""
    try:
        model = tflite.Model.GetRootAsModel(raw, 0)
        n_sub = model.SubgraphsLength()
    except Exception as e:  # a damaged flatbuffer raises anything
        raise ConvertError(f"not a TFLite model: {e}") from e
    if n_sub < 1:
        raise ConvertError("not a TFLite model: no subgraph")
    g = model.Subgraphs(0)
    tensors: list[Tensor] = []
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        dtype = _DTYPES.get(t.Type())
        if dtype is None:
            raise ConvertError(f"tensor {i}: TFLite type {t.Type()} is not supported")
        shape = tuple(int(d) for d in t.ShapeAsNumpy()) if t.ShapeLength() else ()
        if t.ShapeSignatureLength():
            sig = [int(d) for d in t.ShapeSignatureAsNumpy()]
            shape = tuple(1 if d < 0 else d for d in sig) if len(sig) == len(shape) else shape
        data = None
        buf = _buffer(model, raw, t.Buffer())
        if buf is not None:
            data = np.frombuffer(buf, dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)
            data = data.reshape(shape) if data.size == int(np.prod(shape)) else data
        quant = None
        q = t.Quantization()
        if q is not None and q.ScaleLength() > 0:
            scales = q.ScaleAsNumpy().astype(np.float32)
            zps = (q.ZeroPointAsNumpy().astype(np.int64) if q.ZeroPointLength()
                   else np.zeros(scales.shape, np.int64))
            quant = Quant(scales, zps, int(q.QuantizedDimension()) if scales.size > 1 else None)
        name = t.Name().decode("utf-8", "replace") if t.Name() else f"tensor_{i}"
        tensors.append(Tensor(i, name, dtype, shape, data, quant))
    ops: list[Op] = []
    for k in range(g.OperatorsLength()):
        o = g.Operators(k)
        code = model.OperatorCodes(o.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        kind = _OPS.get(builtin, f"BUILTIN_{builtin}")
        if kind == "CUSTOM":
            custom = code.CustomCode()
            kind = "CUSTOM:" + (custom.decode() if custom else "?")
        ins = [int(x) for x in o.InputsAsNumpy()] if o.InputsLength() else []
        outs = [int(x) for x in o.OutputsAsNumpy()] if o.OutputsLength() else []
        ops.append(Op(k, kind, ins, outs, _options(o)))
    inputs = [int(x) for x in g.InputsAsNumpy()] if g.InputsLength() else []
    outputs = [int(x) for x in g.OutputsAsNumpy()] if g.OutputsLength() else []
    in_names: dict[int, str] = {}
    out_names: dict[int, str] = {}
    if model.SignatureDefsLength() > 0:
        sig = model.SignatureDefs(0)
        for j in range(sig.InputsLength()):
            in_names[sig.Inputs(j).TensorIndex()] = sig.Inputs(j).Name().decode()
        for j in range(sig.OutputsLength()):
            out_names[sig.Outputs(j).TensorIndex()] = sig.Outputs(j).Name().decode()
    if n_sub > 1:
        used = {op.kind for op in ops}
        if used & {"WHILE", "IF", "CALL_ONCE", "CALL"}:
            raise ConvertError("control flow (WHILE, IF, CALL) is not supported: rayo runs one "
                               "graph of ops in order")
    return Graph(tensors, ops, inputs, outputs, in_names, out_names, "tflite")
