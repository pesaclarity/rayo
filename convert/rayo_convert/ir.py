"""The converter's model of a source graph: tensors and ops with TFLite's meaning.

Both front ends (TFLite, ONNX) produce this. Op kinds are TFLite's builtin operator names
(``FULLY_CONNECTED``, ``CONV_2D``, ...) with their options as a dict of snake_case names, so one
set of passes and one lowering serve both. Shapes include the batch dimension, as in the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ConvertError(Exception):
    """A model rayo-convert cannot convert; the message names the op and why."""


@dataclass
class Quant:
    """int8 quantisation: real = scales[c] * (q - zero_points[c]), c along ``axis``."""

    scales: np.ndarray
    zero_points: np.ndarray
    axis: int | None  # None: one scale for the whole tensor


@dataclass
class Tensor:
    index: int
    name: str
    dtype: str  # float32, int32, int64, int8, bool, ...
    shape: tuple[int, ...]
    data: np.ndarray | None = None  # set for constants
    quant: Quant | None = None

    @property
    def const(self) -> bool:
        return self.data is not None

    def real(self) -> np.ndarray:
        """A constant's values as float32 (dequantised when int8)."""
        assert self.data is not None
        if self.quant is None or self.dtype != "int8":
            return self.data.astype(np.float32)
        q = self.data.astype(np.float32)
        s, z = self.quant.scales.astype(np.float32), self.quant.zero_points.astype(np.float32)
        if self.quant.axis is None or s.size == 1:
            return ((q - z.reshape(-1)[0]) * s.reshape(-1)[0]).astype(np.float32)
        shape = [1] * q.ndim
        shape[self.quant.axis] = -1
        return ((q - z.reshape(shape)) * s.reshape(shape)).astype(np.float32)


@dataclass
class Op:
    index: int  # the op's index in the source graph, for messages
    kind: str
    inputs: list[int]  # tensor indices; -1 for an absent optional input
    outputs: list[int]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    tensors: list[Tensor]
    ops: list[Op]
    inputs: list[int]
    outputs: list[int]
    input_names: dict[int, str]  # signature names, when the file has them
    output_names: dict[int, str]
    source: str = "tflite"
    hints: dict[int, str] = field(default_factory=dict)  # tensor: the layer to name it after

    def producer(self) -> dict[int, Op]:
        return {t: op for op in self.ops for t in op.outputs}

    def consumers(self) -> dict[int, list[Op]]:
        out: dict[int, list[Op]] = {}
        for op in self.ops:
            for t in op.inputs:
                if t >= 0:
                    out.setdefault(t, []).append(op)
        return out

    def add_tensor(self, name: str, dtype: str, shape: tuple[int, ...],
                   data: np.ndarray | None = None, quant: Quant | None = None) -> int:
        t = Tensor(len(self.tensors), name, dtype, tuple(int(d) for d in shape), data, quant)
        self.tensors.append(t)
        return t.index

    def add_const(self, name: str, value: np.ndarray) -> int:
        value = np.asarray(value)
        return self.add_tensor(name, value.dtype.name, value.shape, value)

    def describe(self, op: Op) -> str:
        out = self.tensors[op.outputs[0]].name if op.outputs else "?"
        return f"op {op.index} ({op.kind}, output {out!r})"
