# rayo

[![CI](https://github.com/pesaclarity/rayo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pesaclarity/rayo/actions/workflows/ci.yml)

On-device inference in plain C89 (ANSI C), for C, Dart and Flutter. A general runtime for
small models: a graph of ops (docs/OPS.md) over a plain-weights file, with ops of your own when
those are not enough. Pesa Clarity's question reader and reply writer run on it, on the phone,
offline, in the Play and the F-Droid builds alike; nothing in the library is specific to them.

- **Our own engine, no Google code**: no TFLite/LiteRT, no XNNPACK, and no other library.
  Portable C first, SIMD where it pays.
- **Built from source, in C89**: a Dart build hook compiles `src/` with the platform's C
  compiler (the NDK, Xcode, clang/gcc), `-std=c89 -pedantic -Werror`. Nothing is downloaded,
  nothing is prebuilt, so F-Droid builds it offline like the rest of the app. No C++ anywhere.
  Compiler extensions (SIMD intrinsics, target attributes, CPU checks) live only in
  `src/kernels/x86*.c` and `src/kernels/neon*.c`, behind `#if`, and those files are C89 too.
- **Small**: the Android library is about 63 KB (arm64-v8a), 49 KB (armeabi-v7a) and 66 KB
  (x86_64), stripped.
- **Memory planned once**: a session plans every activation into one arena when it is made
  (buffers whose lifetimes do not overlap share memory); a run allocates nothing.
- **TFLite's numbers**: int8 layers run exactly as TFLite's dynamic-range kernels do, so a
  model exported from TFLite gives the same outputs (the Ask reader: every label, probabilities
  within 4e-5; the Ask writer: every recorded reply).
- **Decoders**: states kept across runs (a key/value cache) and size symbols bound per run, so
  a transformer decodes word by word without recomputing the past.

## Use

```yaml
dependencies:
  rayo:
    git:
      url: https://github.com/pesaclarity/rayo.git
      ref: v0.1.0
```

```dart
import 'package:rayo/rayo.dart';

final model = RayoModel.load(bytes, sha256: expectedSha256);
final session = RayoSession(model); // keep it for many runs
session.bind('T', 96);
session.inputInt32(0).setAll(0, tokens); // written in place
session.run();
final intent = session.outputFloat32(1); // read in place, until the next run
session.close();
model.close();
```

- `RayoModel.load` / `loadFile` read a model, check its SHA-256 if given, and keep their own
  copy of the bytes. `name`, `metadata`, `inputs` and `outputs` describe it.
- `RayoSession` holds the model's planned memory. Inputs and outputs are views of it: no copy
  in, none out. `bind` sets a size symbol (such as `T`) for the next runs.
- `runInIsolate(session, task)` runs a task on the session in another isolate, so a UI never
  waits. Runs on one session must not overlap.
- `rayoAvailable()` says whether the native library loaded, so an app can fall back.
- Errors (a damaged file, a size that does not fit, an id out of range) throw
  `RayoException`; the C library never aborts on bad input.

From C, include `rayo/rayo.h`: `rayo_model_create` / `rayo_model_destroy`,
`rayo_session_create` / `rayo_session_destroy`, `rayo_session_bind`, `rayo_session_input`,
`rayo_session_run`, `rayo_session_output`, `rayo_session_memory`. Every function documents who
owns what; destroy functions accept NULL.

## Models

A model is a plain-weights file (PCWT) with its graph in the header:
[docs/FORMAT.md](docs/FORMAT.md); its ops are in [docs/OPS.md](docs/OPS.md). Pesa Clarity's
are written by [pesaclarity-ml](https://github.com/pesaclarity/pesaclarity-ml).
`tool/make_test_models.py` (NumPy only) writes the small random models the tests use, with the
outputs NumPy computes.

## Using rayo with your own model

`examples/classifier` is a complete one: a bag-of-words classifier over token ids, with
made-up weights.

1. **Write the file.** `make_model.py` writes `classifier.weights` with `tool/pcwt.py`
   (NumPy only): the tensors, then a graph that names them.

   ```json
   "symbols": {"T": 32},
   "inputs": [{"name": "ids", "dtype": "int32", "shape": ["T"]}],
   "nodes": [
     {"op": "gather", "inputs": ["embedding", "ids"], "output": "vectors"},
     {"op": "compare", "inputs": ["ids"], "output": "real", "comparison": "greater", "value": 0},
     {"op": "masked_reduce", "inputs": ["vectors", "real"], "output": "mean", "reduction": "mean"},
     {"op": "dense", "inputs": ["mean", "hidden.kernel", "hidden.bias"], "output": "hidden",
      "activation": "relu"},
     {"op": "dense", "inputs": ["hidden", "out.kernel", "out.bias"], "output": "logits"},
     {"op": "softmax", "inputs": ["logits"], "output": "probabilities"}
   ],
   "outputs": [{"name": "probabilities", "node": "probabilities"}]
   ```

   `hidden.kernel` is int8 (one scale per output), so that layer runs as TFLite's
   dynamic-range kernels do; the rest are float32.

2. **Run it from C.** `classify.c` loads the file, binds `T` to the number of ids, writes them
   into the input in place, runs, and reads the probabilities in place:
   `classify classifier.weights 4 17 9`. `tool/ci/examples.sh` checks it against NumPy in CI.

3. **Or from Dart**, as in [Use](#use): `RayoModel.load(bytes)`, `RayoSession(model)`,
   `bind('T', ids.length)`, `inputInt32(0).setAll(0, ids)`, `run()`, `outputFloat32(0)`.

**Ops of your own.** If your graph needs an op rayo does not have, write it in C (`prepare`,
`infer` and `run`; `include/rayo/op.h`) and pass it to `rayo_model_create_with`. See the end
of docs/OPS.md.

**Limits.** These are read from the file:
- the number of nodes, inputs, outputs, states and size symbols;
- a node's inputs, a kernel's size and a convolution's dilation.

Each count is bounded only by overflow checks. The compile-time options are in
`include/rayo/config.h`, with their defaults:
- `RAYO_MAX_RANK`: 4;
- `RAYO_MAX_DIM`: 2^24, the largest dimension;
- `RAYO_ALIGN`: 64;
- `RAYO_JSON_MAX_DEPTH`: 32;
- `RAYO_OP_MAX_INPUTS`: 16, for your own ops.

## Layout

```
include/rayo/     public headers: rayo.h (all of them), config.h (compile-time options),
                  types.h, error.h, tensor.h, model.h, session.h, op.h (your own ops),
                  sha256.h, version.h
src/core/         allocation, arena, JSON, tensors, messages and formatting, maths, C89 shims
src/format/       the PCWT reader and its checks
src/graph/        the graph, shapes and the memory plan
src/ops/          one file per op family, and the registry
src/kernels/      generic.c, x86.c (SSE4.1/AVX2), x86_vnni.c, neon.c, neon_dotprod.c,
                  quantize.c, select.c (which kernels this CPU runs)
src/model.c, src/session.c, src/sha256/
lib/              the Dart API: bindings/, model.dart, session.dart, isolate_runner.dart, ...
tests/            the C tests (tests/test_*.c), test models, and the real Ask models (tests/ask)
tools/            bench.c: timing helpers for programs that time a model, op by op
examples/         classifier/: a small model of its own, end to end;
                  ask/: Pesa Clarity's reader and writer against TFLite's recordings, timed
```

## Kernels

The int8 products have a generic C version and SIMD ones, each checked against the generic one
in the tests:

| Build | Baseline | Chosen at run time when the CPU has it |
| --- | --- | --- |
| Android arm64-v8a | NEON (ARMv8.0) | SDOT dot product (ARMv8.2) |
| Android armeabi-v7a | NEON | |
| Android x86_64 | SSE4.1 | AVX-VNNI |
| Linux x86_64 | AVX2 | AVX-VNNI |

## Speed

The Ask reader (96 positions, int8, one thread) on one core of a 2.1 GHz Xeon, per question,
with the same machine's TFLite (Python, `num_threads=1`) for comparison:

| | median |
| --- | --- |
| rayo, AVX-VNNI (chosen at run time) | 0.18 ms |
| rayo, AVX2 | 0.34 ms |
| rayo, SSE4.1 (the Android x86_64 baseline) | 0.66 ms |
| rayo, generic C (compiler-vectorised) | 0.80 ms |
| TFLite with XNNPACK | 0.08 ms |
| TFLite built-in kernels | 0.28 ms |

The Ask writer (a 3-layer transformer, int8, word by word with a key/value cache), per reply of
about 13 steps on the 172 recorded replies, the same machine:

| | median |
| --- | --- |
| rayo, AVX-VNNI | 0.89 ms (0.07 ms a step) |
| TFLite with XNNPACK, 1 thread (the whole 64 positions every step, as the app ran it) | 4.1 ms |
| TFLite with XNNPACK, 2 threads | 6.9 ms |
| TFLite built-in kernels | 39 ms |

Every one of the 265 decodes the app's tests make matches the replies TFLite recorded, token
for token, and the next-token choice agrees at every step.

For the reader, loading the model takes about 0.2 ms and making a session about 0.15 ms. Memory: the file's
copy (251 KB), the session's arena (241 KB) and 82 KB of tables. Measured with
`examples/ask/ask_reader_check.c`, which also prints the time per op. Not yet measured on a phone.

## C89

Every C file, the tests' and tools' too, builds with gcc and clang as
`-std=c89 -pedantic -Werror -Wall -Wextra -Wconversion -Wshadow -Wdeclaration-after-statement -Wvla`.

- No `//`, declarations at the top of blocks, no designated initialisers or compound literals,
  no `inline`, `restrict` only as `RAYO_RESTRICT` (`__restrict__` where the compiler has it),
  no `stdint.h` or `stdbool.h`: `include/rayo/types.h` defines `rayo_i8` to `rayo_u32`,
  `rayo_f32` and `rayo_bool`, with compile-time size checks.
- No 64-bit integers: dimensions are `rayo_i32`, byte counts `size_t`, and every tensor stays
  below 2^31 bytes (checked when a model loads and at every bind). SHA-256 counts its length in
  two 32-bit words.
- C89 has no `snprintf`, `expf`, `sqrtf` or `erf`: `src/core/format.c` formats messages
  (bounded), and `src/core/mathx.c` computes exp and sqrt in double and erf from fitted
  polynomials (correctly rounded on every float in [-6, 6] but where pieces meet; glibc's
  `erff` is one ulp off on 5% of them).

## Develop

```sh
dart pub get
tool/check   # what CI runs and this computer can: lint, C tests, sanitisers, Android, Dart
```

`tool/ci/*.sh` are the steps; `tool/check` and `.github/workflows/ci.yml` both call them.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request and push to
`main`: lint (format, C89 with gcc and clang), the C tests on Linux x86_64 (gcc and clang;
generic, SSE4.1 and AVX2 builds, each kernel set, AVX-VNNI where the runner has it), under
ASan, LSan and UBSan, natively on Linux arm64 (NEON, and SDOT where the CPU has it), on
armv7 under QEMU, on macOS arm64, the Android NDK builds with a size budget, and the Dart
package on Linux and macOS. On `main` a bench job reports the timings in the job summary.

To protect `main`, require these checks: `lint`, `linux-x86_64 (gcc)`, `linux-x86_64 (clang)`,
`sanitizers`, `linux-arm64`, `linux-armv7`, `macos`, `android`, `dart (ubuntu-24.04)`,
`dart (macos-15)`.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); the changelog is
written from them.

## Licence

GNU Affero General Public License v3.0 (`LICENSE`), like the Pesa Clarity app.
