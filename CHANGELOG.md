# Changelog

Written from the conventional commit titles on `main` at each release.

## Unreleased

- rayo is its own engine in plain C11: ggml is gone. Models are plain-weights (PCWT) files
  with a graph in the header (docs/FORMAT.md); int8 layers run as TFLite's dynamic-range
  kernels do. Kernels: generic C, SSE4.1/AVX2, AVX-VNNI, NEON and SDOT.
- A session plans its memory once; a run allocates nothing.
- The Dart API is split into bindings, model, session and an isolate runner; inputs and outputs
  are views of native memory.
