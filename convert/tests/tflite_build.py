"""Small TFLite models built in the tests, with the flatbuffers code of the ``tflite`` package:
no TensorFlow. Ops are written in the forms TFLite's converter emits for Keras layers."""

from __future__ import annotations

import importlib
from typing import Any

import flatbuffers
import numpy as np
import tflite

_TYPES = {"float32": tflite.TensorType.FLOAT32, "int32": tflite.TensorType.INT32,
          "int8": tflite.TensorType.INT8, "bool": tflite.TensorType.BOOL,
          "int64": tflite.TensorType.INT64}
_OPTIONS = {  # op: (options table, BuiltinOptions member)
    "ADD": "AddOptions", "SUB": "SubOptions", "MUL": "MulOptions", "DIV": "DivOptions",
    "FULLY_CONNECTED": "FullyConnectedOptions", "CONV_2D": "Conv2DOptions",
    "SOFTMAX": "SoftmaxOptions", "RESHAPE": "ReshapeOptions",
    "STRIDED_SLICE": "StridedSliceOptions", "BATCH_MATMUL": "BatchMatMulOptions",
    "MEAN": "ReducerOptions", "SUM": "ReducerOptions", "REDUCE_MAX": "ReducerOptions",
    "GATHER": "GatherOptions", "CONCATENATION": "ConcatenationOptions", "GELU": "GeluOptions",
    "TRANSPOSE": "TransposeOptions", "SQUEEZE": "SqueezeOptions",
    "EXPAND_DIMS": "ExpandDimsOptions", "SPACE_TO_BATCH_ND": "SpaceToBatchNDOptions",
    "BATCH_TO_SPACE_ND": "BatchToSpaceNDOptions", "SQUARED_DIFFERENCE": "SquaredDifferenceOptions",
    "RSQRT": None, "NEG": "NegOptions", "GREATER": "GreaterOptions", "LESS": "LessOptions",
    "CAST": "CastOptions", "SELECT_V2": "SelectV2Options", "MAXIMUM": "MaximumMinimumOptions",
    "AVERAGE_POOL_2D": "Pool2DOptions", "RELU": None, "LOGISTIC": None, "TANH": None,
}


def _mod(name: str) -> Any:
    return importlib.import_module(f"tflite.{name}")


class Builder:
    def __init__(self) -> None:
        self.tensors: list[dict[str, Any]] = []
        self.ops: list[tuple[str, list[int], list[int], dict[str, Any]]] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.sig_in: dict[int, str] = {}
        self.sig_out: dict[int, str] = {}

    def tensor(self, name: str, shape: list[int] | tuple[int, ...], dtype: str = "float32",
               data: np.ndarray | None = None, scales: np.ndarray | None = None,
               axis: int = 0) -> int:
        self.tensors.append({"name": name, "shape": list(shape), "dtype": dtype, "data": data,
                             "scales": scales, "axis": axis})
        return len(self.tensors) - 1

    def const(self, name: str, value: Any, dtype: str | None = None) -> int:
        v = np.asarray(value, dtype=dtype)
        if v.dtype == np.float64:
            v = v.astype(np.float32)
        if v.dtype == np.int64 and dtype is None:
            v = v.astype(np.int32)
        return self.tensor(name, v.shape, v.dtype.name, v)

    def int8(self, name: str, real: np.ndarray, axis: int | None = 0) -> int:
        """Symmetric int8, one scale per index along ``axis`` (None: one in all)."""
        if axis is None:
            s = np.asarray([max(float(np.abs(real).max()) / 127.0, 1e-8)], np.float32)
            q = np.clip(np.round(real / s[0]), -127, 127).astype(np.int8)
        else:
            moved = np.moveaxis(real, axis, 0).reshape(real.shape[axis], -1)
            s = np.maximum(np.abs(moved).max(axis=1) / 127.0, 1e-8).astype(np.float32)
            shape = [1] * real.ndim
            shape[axis] = -1
            q = np.clip(np.round(real / s.reshape(shape)), -127, 127).astype(np.int8)
        return self.tensor(name, real.shape, "int8", q, s, axis or 0)

    def op(self, kind: str, inputs: list[int], outputs: list[int], **options: Any) -> int:
        self.ops.append((kind, inputs, outputs, options))
        return outputs[0] if outputs else -1

    def add(self, kind: str, inputs: list[int], name: str, shape: list[int] | tuple[int, ...],
            dtype: str = "float32", **options: Any) -> int:
        out = self.tensor(name, shape, dtype)
        return self.op(kind, inputs, [out], **options)

    def input(self, name: str, shape: list[int], dtype: str = "float32") -> int:
        t = self.tensor(f"serving_default_{name}:0", shape, dtype)
        self.inputs.append(t)
        self.sig_in[t] = name
        return t

    def output(self, t: int, name: str) -> None:
        self.outputs.append(t)
        self.sig_out[t] = name

    def build(self) -> bytes:
        b = flatbuffers.Builder(1024)
        # Buffers: 0 is the empty buffer.
        buf_index = []
        datas = []
        for t in self.tensors:
            if t["data"] is not None:
                datas.append(np.ascontiguousarray(t["data"]).tobytes())
                buf_index.append(len(datas))
            else:
                buf_index.append(0)
        M = _mod

        def vec_i32(values: list[int]) -> int:
            b.StartVector(4, len(values), 4)
            for v in reversed(values):
                b.PrependInt32(int(v))
            return b.EndVector()

        def vec_f32(values: list[float]) -> int:
            b.StartVector(4, len(values), 4)
            for v in reversed(values):
                b.PrependFloat32(float(v))
            return b.EndVector()

        def vec_i64(values: list[int]) -> int:
            b.StartVector(8, len(values), 8)
            for v in reversed(values):
                b.PrependInt64(int(v))
            return b.EndVector()

        def vec_off(offs: list[int]) -> int:
            b.StartVector(4, len(offs), 4)
            for o in reversed(offs):
                b.PrependUOffsetTRelative(o)
            return b.EndVector()

        buffers = []
        empty = M("Buffer")
        empty.BufferStart(b)
        buffers.append(empty.BufferEnd(b))
        for d in datas:
            b.StartVector(1, len(d), 16)
            b.head = b.head - len(d)
            b.Bytes[b.head : b.head + len(d)] = d
            v = b.EndVector()
            mb = M("Buffer")
            mb.BufferStart(b)
            mb.BufferAddData(b, v)
            buffers.append(mb.BufferEnd(b))
        tensors = []
        for i, t in enumerate(self.tensors):
            name = b.CreateString(t["name"])
            shape = vec_i32(t["shape"])
            q = None
            if t["scales"] is not None:
                sc = vec_f32(list(np.asarray(t["scales"]).reshape(-1)))
                zp = vec_i64([0] * np.asarray(t["scales"]).size)
                mq = M("QuantizationParameters")
                mq.QuantizationParametersStart(b)
                mq.QuantizationParametersAddScale(b, sc)
                mq.QuantizationParametersAddZeroPoint(b, zp)
                mq.QuantizationParametersAddQuantizedDimension(b, int(t["axis"]))
                q = mq.QuantizationParametersEnd(b)
            mt = M("Tensor")
            mt.TensorStart(b)
            mt.TensorAddShape(b, shape)
            mt.TensorAddType(b, _TYPES[t["dtype"]])
            mt.TensorAddBuffer(b, buf_index[i])
            mt.TensorAddName(b, name)
            if q is not None:
                mt.TensorAddQuantization(b, q)
            tensors.append(mt.TensorEnd(b))
        kinds = sorted({k for k, *_ in self.ops})
        codes = []
        for k in kinds:
            mc = M("OperatorCode")
            mc.OperatorCodeStart(b)
            code = getattr(tflite.BuiltinOperator, k)
            mc.OperatorCodeAddDeprecatedBuiltinCode(b, min(code, 127))
            mc.OperatorCodeAddBuiltinCode(b, code)
            mc.OperatorCodeAddVersion(b, 1)
            codes.append(mc.OperatorCodeEnd(b))
        ops = []
        for kind, ins, outs, opt in self.ops:
            table = _OPTIONS.get(kind)
            opt_off = None
            if table:
                mo = M(table)
                vectors = {k: vec_i32(v) for k, v in opt.items() if isinstance(v, list)}
                getattr(mo, f"{table}Start")(b)
                for k, v in opt.items():
                    field = "".join(p.capitalize() for p in k.split("_"))
                    getattr(mo, f"{table}Add{field}")(b, vectors.get(k, v))
                opt_off = getattr(mo, f"{table}End")(b)
            iv, ov = vec_i32(ins), vec_i32(outs)
            mo2 = M("Operator")
            mo2.OperatorStart(b)
            mo2.OperatorAddOpcodeIndex(b, kinds.index(kind))
            mo2.OperatorAddInputs(b, iv)
            mo2.OperatorAddOutputs(b, ov)
            if opt_off is not None:
                mo2.OperatorAddBuiltinOptionsType(b, getattr(tflite.BuiltinOptions, table))
                mo2.OperatorAddBuiltinOptions(b, opt_off)
            ops.append(mo2.OperatorEnd(b))
        tv, opv = vec_off(tensors), vec_off(ops)
        inv, outv = vec_i32(self.inputs), vec_i32(self.outputs)
        ms = M("SubGraph")
        ms.SubGraphStart(b)
        ms.SubGraphAddTensors(b, tv)
        ms.SubGraphAddInputs(b, inv)
        ms.SubGraphAddOutputs(b, outv)
        ms.SubGraphAddOperators(b, opv)
        sub = ms.SubGraphEnd(b)

        def tensor_maps(d: dict[int, str]) -> int:
            offs = []
            for t, n in d.items():
                nm = b.CreateString(n)
                mm = M("TensorMap")
                mm.TensorMapStart(b)
                mm.TensorMapAddName(b, nm)
                mm.TensorMapAddTensorIndex(b, t)
                offs.append(mm.TensorMapEnd(b))
            return vec_off(offs)

        si, so = tensor_maps(self.sig_in), tensor_maps(self.sig_out)
        key = b.CreateString("serving_default")
        msd = M("SignatureDef")
        msd.SignatureDefStart(b)
        msd.SignatureDefAddInputs(b, si)
        msd.SignatureDefAddOutputs(b, so)
        msd.SignatureDefAddSignatureKey(b, key)
        msd.SignatureDefAddSubgraphIndex(b, 0)
        sig = msd.SignatureDefEnd(b)
        subs, codesv, bufs, sigs = vec_off([sub]), vec_off(codes), vec_off(buffers), vec_off([sig])
        desc = b.CreateString("rayo-convert test")
        mm = M("Model")
        mm.ModelStart(b)
        mm.ModelAddVersion(b, 3)
        mm.ModelAddOperatorCodes(b, codesv)
        mm.ModelAddSubgraphs(b, subs)
        mm.ModelAddDescription(b, desc)
        mm.ModelAddBuffers(b, bufs)
        mm.ModelAddSignatureDefs(b, sigs)
        b.Finish(mm.ModelEnd(b), file_identifier=b"TFL3")
        return bytes(b.Output())
