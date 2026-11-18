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
    _fields_ = [("ops", C.c_void_p), ("op_count", C.c_int), ("read_file", C.c_int)]


class _SessionOptions(C.Structure):
    _fields_ = [("threads", C.c_int)]


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
        cmd = [cc or os.environ.get("CC", "cc"), "-std=c89", "-O2", "-shared", "-fPIC",
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
            opts = _ModelOptions(None, 0, int(read_file))
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
    def __init__(self, model: Model, threads: int = 1) -> None:
        self.model = model
        self.L = model.lib.lib
        self.ptr = C.c_void_p()
        msg = _Message()
        opts = _SessionOptions(threads)
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
