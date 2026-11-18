"""rayo's C library from Python (ctypes), for checking a converted model against its source.

``Library.build(rayo_dir)`` compiles rayo's ``src/`` into a shared library with the system's C
compiler (C89, as rayo is always built) once per process; ``Model`` and ``Session`` wrap the C
API (include/rayo/*.h).
"""

from __future__ import annotations

import ctypes as C
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

MAX_RANK = 4
_DTYPES = {0: np.float32, 1: np.int32, 2: np.int8}


class _Shape(C.Structure):
    _fields_ = [("rank", C.c_int32), ("dims", C.c_int32 * MAX_RANK)]


class _View(C.Structure):
    _fields_ = [("name", C.c_char_p), ("dtype", C.c_int), ("shape", _Shape),
                ("data", C.c_void_p), ("bytes", C.c_size_t)]


class _Message(C.Structure):
    _fields_ = [("text", C.c_char * 256)]


class _ModelOptions(C.Structure):
    _fields_ = [("ops", C.c_void_p), ("op_count", C.c_int), ("read_file", C.c_int),
                ("access", C.c_int), ("integrity", C.c_void_p)]


class _SessionOptions(C.Structure):
    _fields_ = [("threads", C.c_int), ("cache", C.c_int), ("state_rows", C.c_int32),
                ("limits", C.c_void_p), ("limit_count", C.c_int)]


CACHE_FLOAT32, CACHE_INT8 = 0, 1


class RayoError(RuntimeError):
    pass


def default_rayo_dir() -> Path | None:
    """The rayo checkout this package sits in (convert/ is in rayo), if it is one."""
    here = Path(__file__).resolve().parent.parent.parent
    return here if (here / "src" / "model.c").exists() else None


class Library:
    _built: ClassVar[dict[str, Library]] = {}

    def __init__(self, path: Path) -> None:
        self.lib = C.CDLL(str(path))
        L = self.lib
        L.rayo_model_create.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
                                        C.POINTER(_Message)]
        L.rayo_session_create.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(_Message)]
        L.rayo_session_bind.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(_Message)]
        L.rayo_session_input.argtypes = [C.c_void_p, C.c_int, C.POINTER(_View)]
        L.rayo_session_output.argtypes = [C.c_void_p, C.c_int, C.POINTER(_View)]
        L.rayo_session_run.argtypes = [C.c_void_p, C.POINTER(_Message)]
        L.rayo_session_reset.argtypes = [C.c_void_p]
        L.rayo_session_destroy.argtypes = [C.c_void_p]
        L.rayo_model_destroy.argtypes = [C.c_void_p]
        L.rayo_model_input_count.argtypes = [C.c_void_p]
        L.rayo_model_output_count.argtypes = [C.c_void_p]
        L.rayo_model_input_info.argtypes = [C.c_void_p, C.c_int, C.POINTER(_View)]
        L.rayo_model_output_info.argtypes = [C.c_void_p, C.c_int, C.POINTER(_View)]
        L.rayo_session_kernels.argtypes = [C.c_void_p]
        L.rayo_session_kernels.restype = C.c_char_p
        L.rayo_model_load.argtypes = [C.c_char_p, C.POINTER(_ModelOptions), C.POINTER(C.c_void_p),
                                      C.POINTER(_Message)]
        L.rayo_session_create_with.argtypes = [C.c_void_p, C.POINTER(_SessionOptions),
                                               C.POINTER(C.c_void_p), C.POINTER(_Message)]
        L.rayo_session_threads.argtypes = [C.c_void_p]
        L.rayo_tokenizer_create.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p),
                                            C.POINTER(_Message)]
        L.rayo_tokenizer_encode.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t,
                                            C.POINTER(C.c_int32), C.c_size_t,
                                            C.POINTER(C.c_size_t), C.POINTER(_Message)]
        L.rayo_tokenizer_piece.argtypes = [C.c_void_p, C.c_int32, C.POINTER(C.c_size_t)]
        L.rayo_tokenizer_piece.restype = C.c_void_p
        L.rayo_tokenizer_destroy.argtypes = [C.c_void_p]
        L.rayo_tokenizer_create_table.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
                                                  C.POINTER(_Message)]
        L.rayo_tokenizer_table.argtypes = [C.c_void_p, C.POINTER(C.c_void_p),
                                           C.POINTER(C.c_size_t)]
        L.rayo_tokenizer_vocab_size.argtypes = [C.c_void_p]

    @classmethod
    def build(cls, rayo_dir: str | Path | None = None, cc: str | None = None,
              flags: list[str] | None = None) -> Library:
        root = Path(rayo_dir) if rayo_dir else default_rayo_dir()
        if root is None:
            raise RayoError("no rayo source tree: pass its directory")
        key = f"{root}|{cc}|{flags}"
        if key in cls._built:
            return cls._built[key]
        out = Path(tempfile.mkdtemp(prefix="rayo-lib-")) / "librayo.so"
        srcs = sorted(str(p) for p in (root / "src").rglob("*.c"))
        cmd = [cc or os.environ.get("CC", "cc"), "-std=c89", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
               f"-I{root / 'include'}", f"-I{root / 'src'}",
               *(flags or []), *srcs, "-lm", "-lpthread", "-o", str(out)]
        subprocess.run(cmd, check=True)
        lib = cls(out)
        cls._built[key] = lib
        return lib


class Model:
    def __init__(self, lib: Library, data: bytes | None = None, path: str | Path | None = None,
                 read_file: bool = False) -> None:
        """A model from its bytes, or loaded (mapped) from ``path``."""
        self.lib = lib
        self.ptr = C.c_void_p()
        msg = _Message()
        if path is not None:
            opts = _ModelOptions(None, 0, int(read_file), 0, None)
            s = lib.lib.rayo_model_load(str(path).encode(), C.byref(opts), C.byref(self.ptr),
                                        C.byref(msg))
        else:
            assert data is not None
            self._bytes = C.create_string_buffer(data, len(data))
            s = lib.lib.rayo_model_create(self._bytes, len(data), C.byref(self.ptr),
                                          C.byref(msg))
        if s != 0:
            raise RayoError(msg.text.decode())

    def close(self) -> None:
        if self.ptr:
            self.lib.lib.rayo_model_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class Session:
    def __init__(self, model: Model, threads: int = 1, cache: int = CACHE_FLOAT32) -> None:
        self.model = model
        self.L = model.lib.lib
        self.ptr = C.c_void_p()
        msg = _Message()
        opts = _SessionOptions(threads, cache, 0, None, 0)
        if self.L.rayo_session_create_with(model.ptr, C.byref(opts), C.byref(self.ptr),
                                           C.byref(msg)) != 0:
            raise RayoError(msg.text.decode())

    def threads(self) -> int:
        return int(self.L.rayo_session_threads(self.ptr))

    def bind(self, name: str, value: int) -> None:
        msg = _Message()
        if self.L.rayo_session_bind(self.ptr, name.encode(), int(value), C.byref(msg)) != 0:
            raise RayoError(msg.text.decode())

    def _view(self, fn: Any, i: int) -> np.ndarray:
        v = _View()
        if fn(self.ptr, i, C.byref(v)) != 0:
            raise RayoError(f"no tensor {i}")
        shape = [v.shape.dims[k] for k in range(v.shape.rank)]
        dt = _DTYPES[v.dtype]
        n = int(np.prod(shape)) if shape else 1
        buf = (C.c_char * (n * np.dtype(dt).itemsize)).from_address(v.data) if n else b""
        return np.frombuffer(buf, dtype=dt, count=n).reshape(shape)

    def input(self, i: int) -> np.ndarray:
        return self._view(self.L.rayo_session_input, i)

    def output(self, i: int) -> np.ndarray:
        return self._view(self.L.rayo_session_output, i).copy()

    def run(self, feeds: list[np.ndarray] | None = None) -> None:
        for i, x in enumerate(feeds or []):
            self.input(i)[...] = np.asarray(x).reshape(self.input(i).shape)
        msg = _Message()
        if self.L.rayo_session_run(self.ptr, C.byref(msg)) != 0:
            raise RayoError(msg.text.decode())

    def reset(self) -> None:
        self.L.rayo_session_reset(self.ptr)

    def kernels(self) -> str:
        return self.L.rayo_session_kernels(self.ptr).decode()

    def close(self) -> None:
        if self.ptr:
            self.L.rayo_session_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class Tokenizer:
    """rayo's tokeniser of a tokenizer.json, or (with ``table``) of a tokeniser table
    (rayo_convert.tokenizer_table), read in place."""

    def __init__(self, lib: Library, json_bytes: bytes | None = None,
                 table: bytes | None = None) -> None:
        self.L = lib.lib
        self.ptr = C.c_void_p()
        msg = _Message()
        if table is not None:
            # The C side reads the table where it is: keep it, 4-byte aligned.
            self._table = (C.c_uint32 * ((len(table) + 3) // 4))()
            C.memmove(self._table, table, len(table))
            s = self.L.rayo_tokenizer_create_table(self._table, len(table), C.byref(self.ptr),
                                                   C.byref(msg))
        else:
            assert json_bytes is not None
            s = self.L.rayo_tokenizer_create(json_bytes, len(json_bytes), C.byref(self.ptr),
                                             C.byref(msg))
        if s != 0:
            raise RayoError(msg.text.decode())

    def table(self) -> bytes:
        """The tokeniser's table, as rayo holds it."""
        p, n = C.c_void_p(), C.c_size_t()
        if self.L.rayo_tokenizer_table(self.ptr, C.byref(p), C.byref(n)) != 0:
            raise RayoError("no table")
        return C.string_at(p, n.value)

    def encode(self, text: str) -> list[int]:
        raw = text.encode()
        cap = 4 * len(raw) + 16
        ids = (C.c_int32 * cap)()
        n = C.c_size_t()
        msg = _Message()
        if self.L.rayo_tokenizer_encode(self.ptr, raw, len(raw), ids, cap, C.byref(n),
                                        C.byref(msg)) != 0:
            raise RayoError(msg.text.decode())
        return list(ids[: n.value])

    def piece(self, i: int) -> bytes:
        n = C.c_size_t()
        p = self.L.rayo_tokenizer_piece(self.ptr, i, C.byref(n))
        return C.string_at(p, n.value) if p else b""

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.piece(i) for i in ids).decode("utf-8", "replace")

    def close(self) -> None:
        if self.ptr:
            self.L.rayo_tokenizer_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __del__(self) -> None:
        self.close()
