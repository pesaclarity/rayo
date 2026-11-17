"""GGUF files for rayo (needs only NumPy). A copy of pesaclarity-ml's akili/export/gguf.py.

GGUF is ggml's model file: a header, typed metadata, tensor descriptions, then the tensors,
each aligned to 32 bytes. rayo reads its graph from the metadata key ``rayo.graph`` and its
weights from the tensors (docs/FORMAT.md). This module writes and reads the subset rayo uses:
metadata of strings, integers, floats, booleans and arrays of them, and tensors in F32, F16 or
Q8_0 (blocks of 32 int8 values with one float16 scale).

ggml lists a tensor's dimensions innermost first, the reverse of NumPy: a NumPy array of shape
``(out, in)`` is a ggml tensor ``[in, out]``. This module takes and returns NumPy shapes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32

# ggml tensor types.
F32, F16, Q8_0 = 0, 1, 8
TYPE_NAMES = {F32: "f32", F16: "f16", Q8_0: "q8_0"}
QK8_0 = 32

# GGUF metadata value types.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)


def quantize_q8_0(x: npt.NDArray[np.float32]) -> bytes:
    """``x`` (innermost axis a multiple of 32) as Q8_0 blocks, as ggml quantises it."""
    flat = np.asarray(x, dtype=np.float32).reshape(-1, QK8_0)
    amax = np.abs(flat).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.clip(np.round(flat * inv[:, None]), -128, 127).astype(np.int8)
    blocks = np.empty(flat.shape[0], dtype=[("d", "<f2"), ("q", "i1", QK8_0)])
    blocks["d"] = d.astype(np.float16)
    blocks["q"] = q
    return blocks.tobytes()


def dequantize_q8_0(data: bytes, shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    blocks = np.frombuffer(data, dtype=[("d", "<f2"), ("q", "i1", QK8_0)])
    values = blocks["q"].astype(np.float32) * blocks["d"].astype(np.float32)[:, None]
    return values.reshape(shape)


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # NumPy order
    type: int
    data: bytes

    def values(self) -> npt.NDArray[np.float32]:
        if self.type == F32:
            return np.frombuffer(self.data, "<f4").reshape(self.shape).copy()
        if self.type == F16:
            return np.frombuffer(self.data, "<f2").astype(np.float32).reshape(self.shape)
        if self.type == Q8_0:
            return dequantize_q8_0(self.data, self.shape)
        raise ValueError(f"tensor type {self.type}")


@dataclass(slots=True)
class GgufFile:
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, GgufTensor] = field(default_factory=dict)

    def add(self, name: str, value: npt.ArrayLike, type_: int = F32) -> None:
        """Adds ``value`` stored as ``type_`` (F32, F16 or Q8_0)."""
        if name in self.tensors:
            raise ValueError(f"tensor {name} twice")
        a = np.ascontiguousarray(value, dtype=np.float32)
        if type_ == F32:
            data = a.astype("<f4").tobytes()
        elif type_ == F16:
            data = a.astype("<f2").tobytes()
        elif type_ == Q8_0:
            if a.shape[-1] % QK8_0:
                raise ValueError(f"{name}: Q8_0 needs the innermost size to be a multiple of 32")
            data = quantize_q8_0(a)
        else:
            raise ValueError(f"tensor type {type_}")
        self.tensors[name] = GgufTensor(name, tuple(a.shape), type_, data)

    def to_bytes(self) -> bytes:
        out = bytearray(MAGIC)
        out += struct.pack("<IQQ", VERSION, len(self.tensors), len(self.metadata))
        for key, value in self.metadata.items():
            out += _string(key)
            out += _value(value)
        offset = 0
        for t in self.tensors.values():
            out += _string(t.name)
            out += struct.pack("<I", len(t.shape))
            out += b"".join(struct.pack("<Q", d) for d in reversed(t.shape))
            out += struct.pack("<IQ", t.type, offset)
            offset += _aligned(len(t.data))
        out += b"\0" * (_aligned(len(out)) - len(out))
        for t in self.tensors.values():
            out += t.data + b"\0" * (_aligned(len(t.data)) - len(t.data))
        return bytes(out)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return p

    @classmethod
    def from_bytes(cls, data: bytes) -> GgufFile:
        r = _Reader(data)
        if r.take(4) != MAGIC:
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", r.take(20))
        if version != VERSION:
            raise ValueError(f"GGUF version {version}")
        f = cls()
        for _ in range(n_kv):
            key = r.string()
            f.metadata[key] = r.value(r.u32())
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            type_, offset = struct.unpack("<IQ", r.take(12))
            infos.append((name, tuple(reversed(dims)), type_, offset))
        start = _aligned(r.at)
        for name, shape, type_, offset in infos:
            n = int(np.prod(shape))
            size = n * 4 if type_ == F32 else n * 2 if type_ == F16 else n // QK8_0 * 34
            at = start + offset
            f.tensors[name] = GgufTensor(name, shape, type_, data[at : at + size])
        return f


def _aligned(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _scalar(value: Any) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return _BOOL, struct.pack("<b", value)
    if isinstance(value, int):
        if 0 <= value < 2**32:
            return _U32, struct.pack("<I", value)
        return _I64, struct.pack("<q", value)
    if isinstance(value, float):
        return _F32, struct.pack("<f", value)
    if isinstance(value, str):
        return _STR, _string(value)
    raise TypeError(f"metadata value {value!r}")


def _value(value: Any) -> bytes:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty metadata array")
        items = [_scalar(v) for v in value]
        kinds = {k for k, _ in items}
        if len(kinds) != 1:
            raise ValueError(f"mixed metadata array {value!r}")
        return struct.pack("<IIQ", _ARR, items[0][0], len(items)) + b"".join(b for _, b in items)
    kind, data = _scalar(value)
    return struct.pack("<I", kind) + data


class _Reader:
    _FIXED = {
        _U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i", _F32: "<f",
        _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d",
    }

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.at = 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise ValueError("GGUF file ends early")
        b = self.data[self.at : self.at + n]
        self.at += n
        return b

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.take(8))[0])

    def string(self) -> str:
        return self.take(self.u64()).decode()

    def value(self, kind: int) -> Any:
        if kind == _STR:
            return self.string()
        if kind == _ARR:
            inner, n = self.u32(), self.u64()
            return [self.value(inner) for _ in range(n)]
        fmt = self._FIXED[kind]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
