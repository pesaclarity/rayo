# Changelog

Written from the conventional commit titles on `main` at each release.

## Unreleased

- erf is fdlibm's (`LICENSES/fdlibm.txt`) in place of fitted polynomials, checked on every
  float in [0, 4] against the C library's erf (`tool/ci/erf.sh`, in CI and `tool/check`).

- rayo is C89 (ANSI C): every file builds with `-std=c89 -pedantic -Werror` and gcc and clang.
  Dimensions are 32-bit (`rayo_i32`) in the API; the fixed-size types are in
  `include/rayo/types.h`. erf, exp and sqrt come from `src/core/mathx.c`.
- The C tests run the real Ask reader and writer against TFLite (tests/ask) with each kernel
  set; GitHub Actions CI on Linux x86_64 and arm64, armv7, macOS and the Android NDK.

- Decoders: graph states (a key/value cache) with `rayo_session_reset`, a graph `plan` for
  symbols that trade off, and the ops `layer_norm`, `attention`, `cache_write`, `range` and
  `last_row`, and the `gelu_erf` activation. The Ask writer runs on them.

- rayo is its own engine in plain C11: ggml is gone. Models are plain-weights (PCWT) files
  with a graph in the header (docs/FORMAT.md); int8 layers run as TFLite's dynamic-range
  kernels do. Kernels: generic C, SSE4.1/AVX2, AVX-VNNI, NEON and SDOT.
- A session plans its memory once; a run allocates nothing.
- The Dart API is split into bindings, model, session and an isolate runner; inputs and outputs
  are views of native memory.
