# rayo

On-device inference for Dart and Flutter, on [ggml](https://github.com/ggml-org/ggml)
compiled from source. Built for [Pesa Clarity](https://github.com/pesaclarity), whose small
models (a question reader, a reply writer) run on the phone, offline, in the Play and the
F-Droid builds alike.

- **No Google ML code**: no TFLite/LiteRT, no XNNPACK. The engine is ggml (MIT), the C
  library behind llama.cpp, with NEON and SSE/AVX kernels, int8 and float16 weights, a
  thread pool, and the ops transformers need, so it grows to larger models.
- **No prebuilt binaries**: a Dart build hook compiles rayo and the vendored ggml with the
  platform's C/C++ compiler (the Android NDK, Xcode, clang/gcc). Nothing is downloaded at
  build time, so F-Droid builds it offline like the rest of the app.
- **Small**: the Android library is about 1.4 MB (arm64-v8a, stripped), 1.2 MB
  (armeabi-v7a) and 1.6 MB (x86_64).

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
final session = model.session(threads: 1);
session
  ..bind('T', tokens.length)
  ..setInt32('tokens', tokens)
  ..run();
final logits = session.output('logits'); // Float32List data, shape innermost first
session.close();
model.close();
```

- `RayoModel.load` / `loadFile` read a GGUF model and, given `sha256`, refuse bytes that
  do not hash to it. `metadata`, `inputs`, `states` and `outputs` describe it.
- A `RayoSession` holds a model's threads, scratch memory and state (such as a key/value
  cache; `reset()` zeroes it). `run()` computes on the calling thread; to keep a UI free,
  pass `session.address` to another isolate and use `RayoSession.fromAddress` there.
- `rayoAvailable()` says whether the native library loaded, so an app can fall back to
  something else on a platform rayo was not built for.
- Errors (a damaged model, a wrong input size, a shape that does not fit) throw
  `RayoException`.

## Models

A model is a GGUF file with its graph as text in the metadata: a list of ops (embeddings,
dense layers, 1-D convolutions, layer norm, GELU, attention with a key/value cache, masked
pooling and a few more) over named tensors. [docs/FORMAT.md](docs/FORMAT.md) specifies it.
Graphs take size symbols, so one model runs on inputs of any length, and a reader can run
on a question's real tokens instead of its padding.

Pesa Clarity's models are exported for rayo by
[pesaclarity-ml](https://github.com/pesaclarity/pesaclarity-ml) (`akili.export.rayo`).
`tool/gguf.py` (NumPy only) writes and reads the files; `tool/make_test_models.py` writes
the small random models the tests use.

## Platforms

| Target | Built by the hook | Tested |
| --- | --- | --- |
| Android arm64-v8a, armeabi-v7a, x86_64 | NDK clang | compiled and linked; not yet run on a phone |
| Linux x86_64 | clang/gcc | the test suite (assumes AVX2) |
| iOS, macOS | Xcode clang | not yet built |

arm64 is built for the ARMv8.0 baseline (NEON), so it runs on every 64-bit Android phone.
Newer cores' dot-product and float16 instructions are not used yet; dispatching to them at
run time is future work.

## Build

`dart test` or a Flutter build runs `hook/build.dart`, which compiles ggml's C and C++ halves
into two static libraries and links them with `src/rayo.cpp` into `librayo`. The first build
takes about a minute per target; later builds are cached.

The ggml sources in `third_party/ggml` are the CPU part of ggml **v1.0.0**, commit
`417a024b4fb6bc3116d41af7c067166a039bdf12`, copied unchanged by `tool/vendor_ggml.sh`,
which records the commit in `third_party/ggml/PIN` and every file's SHA-256 in
`third_party/ggml/SHA256SUMS`. `tool/check` verifies them. To move to another ggml
release, change the tag and commit in `tool/vendor_ggml.sh`, run it, and review the diff.

## Develop

```sh
dart pub get
tool/check     # vendored sources, test models, format, analyze, tests
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/); the changelog
is written from them.

## Licence

rayo is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`), like the
Pesa Clarity app. The vendored ggml, including the llamafile sgemm in
`third_party/ggml/src/ggml-cpu/llamafile` (Mozilla Foundation), is under the MIT licence in
`third_party/ggml/LICENSE`.
