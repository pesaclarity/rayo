"""Writing a plain-weights file (PCWT version 1, pesaclarity-ml docs/WEIGHTS.md) with a rayo
graph (rayo docs/FORMAT.md)."""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from .lower import Weight

MAGIC = b"PCWT"
VERSION = 1


def tensor_entries(weights: list[Weight]) -> tuple[list[dict[str, Any]], bytes]:
    """The header's tensor list and the data after the header."""
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    offset = 0

    def put(data: bytes) -> tuple[int, int]:
        nonlocal offset
        at = offset
        blobs.append(data + b"\0" * (-len(data) % 4))
        offset += len(blobs[-1])
        return at, len(data)

    for w in weights:
        if w.scales is not None:
            at, size = put(np.ascontiguousarray(w.data, "<i1").tobytes())
            s_at, s_size = put(np.ascontiguousarray(w.scales, "<f4").tobytes())
            entries.append({"name": w.name, "dtype": "int8", "shape": list(w.data.shape),
                            "offset": at, "bytes": size,
                            "quantization": {"axis": w.axis, "zero_point": int(w.zero_point),
                                             "scales_offset": s_at, "scales_bytes": s_size}})
        else:
            at, size = put(np.ascontiguousarray(w.data, "<f4").tobytes())
            entries.append({"name": w.name, "dtype": "float32", "shape": list(w.data.shape),
                            "offset": at, "bytes": size})
    return entries, b"".join(blobs)


def write(header: dict[str, Any], weights: list[Weight]) -> bytes:
    """The file: ``header`` (without ``tensors``) with the tensor list of ``weights``."""
    entries, data = tensor_entries(weights)
    full = {**header, "tensors": entries}
    order = ["format_version", "model", "source", "layers", "tensors", "graph", "metadata"]
    full = {k: full[k] for k in order if k in full} | {k: v for k, v in full.items()
                                                       if k not in order}
    h = json.dumps(full, separators=(",", ":")).encode()
    h += b" " * (-(12 + len(h)) % 4)
    return MAGIC + struct.pack("<II", VERSION, len(h)) + h + data
