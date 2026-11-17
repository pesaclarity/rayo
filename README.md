# rayo

On-device inference in plain C11, for Dart and Flutter. Built for
[Pesa Clarity](https://github.com/pesaclarity), whose small models (a question reader, a reply
writer) run on the phone, offline, in the Play and the F-Droid builds alike.

- **Our own engine, no Google code**: no TFLite/LiteRT, no XNNPACK, and no other library. The
  kernels for exactly the ops our models use, portable C first, SIMD where it pays.
- **Built from source**: a Dart build hook compiles `src/` with the platform's C compiler
  (the NDK, Xcode, clang/gcc). Nothing is downloaded, nothing is prebuilt, so F-Droid builds
  it offline like the rest of the app. No C++ anywhere.
- **Small**: the Android library is about 63 KB (arm64-v8a), 49 KB (armeabi-v7a) and 64 KB
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
[docs/FORMAT.md](docs/FORMAT.md). Pesa Clarity's are written by
[pesaclarity-ml](https://github.com/pesaclarity/pesaclarity-ml). `tool/make_test_models.py`
(NumPy only) writes the small random models the tests use, with the outputs NumPy computes.

## Layout

```
include/rayo/     public headers: rayo.h (all of them), types.h, error.h, tensor.h,
                  model.h, session.h, sha256.h, version.h
src/core/         allocation, arena, JSON, tensors, messages
src/format/       the PCWT reader and its checks
src/graph/        the graph, shapes and the memory plan
src/ops/          one file per op family, and the registry
src/kernels/      generic.c, x86.c (SSE4.1/AVX2), x86_vnni.c, neon.c, neon_dotprod.c,
                  quantize.c, select.c (which kernels this CPU runs)
src/model.c, src/session.c, src/sha256/
lib/              the Dart API: bindings/, model.dart, session.dart, isolate_runner.dart, ...
tests/            the C tests (tests/test_*.c) and test models
tools/            ask_reader_check.c: the reader against TFLite's recorded outputs, timed;
                  ask_writer_bench.c: the writer decoding the recorded replies, timed
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
`tools/ask_reader_check.c`, which also prints the time per op. Not yet measured on a phone.

## Develop

```sh
dart pub get
tool/check   # C tests (generic, and AVX2 under ASan/UBSan), formats, Dart analyze and tests
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); the changelog is
written from them.

## Licence

GNU Affero General Public License v3.0 (`LICENSE`), like the Pesa Clarity app.
