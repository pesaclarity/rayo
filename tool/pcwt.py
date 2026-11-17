"""Plain-weights (PCWT) files with a rayo graph, in NumPy only.

The format is pesaclarity-ml's docs/WEIGHTS.md (version 1) with the optional "graph" and
"metadata" header fields of rayo's docs/FORMAT.md. This writer is for rayo's own test
models; pesaclarity-ml writes the real ones.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

MAGIC = b"PCWT"
VERSION = 1


def int8_per_channel(real: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 with one scale per index along ``axis``, as TFLite quantises weights."""
    moved = np.moveaxis(real, axis, 0).reshape(real.shape[axis], -1)
    scales = (np.abs(moved).max(axis=1) / 127.0).astype(np.float32)
    scales[scales == 0] = 1.0
    shape = [1] * real.ndim
    shape[axis] = -1
    q = np.clip(np.round(real / scales.reshape(shape)), -127, 127).astype(np.int8)
    return q, scales


def write(model: str, tensors: dict[str, Any], graph: dict[str, Any],
          metadata: dict[str, Any] | None = None) -> bytes:
    """``tensors``: name to a float32 array, or to (int8 array, scales, axis or None)."""
    entries, blobs, offset = [], [], 0

    def put(data: bytes) -> tuple[int, int]:
        nonlocal offset
        at = offset
        blobs.append(data + b"\0" * (-len(data) % 4))
        offset += len(blobs[-1])
        return at, len(data)

    for name, t in tensors.items():
        if isinstance(t, tuple):
            q, scales, axis = t
            at, size = put(np.asarray(q, "<i1").tobytes())
            s_at, s_size = put(np.asarray(scales, "<f4").tobytes())
            entries.append({"name": name, "dtype": "int8", "shape": list(q.shape), "offset": at,
                            "bytes": size, "quantization": {"axis": axis, "zero_point": 0,
                                                            "scales_offset": s_at,
                                                            "scales_bytes": s_size}})
        else:
            a = np.asarray(t, "<f4")
            at, size = put(a.tobytes())
            entries.append({"name": name, "dtype": "float32", "shape": list(a.shape),
                            "offset": at, "bytes": size})
    header: dict[str, Any] = {"format_version": VERSION, "model": model,
                              "source": {"file": "synthetic", "sha256": "0" * 64, "size_bytes": 0},
                              "layers": [], "tensors": entries, "graph": graph}
    if metadata is not None:
        header["metadata"] = metadata
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-(12 + len(h)) % 4)
    return MAGIC + struct.pack("<II", VERSION, len(h)) + h + b"".join(blobs)


def round_half_away(x: Any) -> Any:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantised(x: np.ndarray) -> np.ndarray:
    """``x``'s int8 round trip as TFLite's hybrid kernels quantise a float input: one scale and
    zero point for all of ``x`` (pesaclarity-ml akili.ask.reader_weights._quantised)."""
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return x.astype(np.float64)
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if lo == hi:
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
