# WebAssembly

rayo builds to WebAssembly with [wasi-sdk](https://github.com/WebAssembly/wasi-sdk) without
changes to its core: the platform layer falls back to the C library when it is neither POSIX
nor Windows, and a build without threads (`-DRAYO_NO_THREADS`) needs nothing else. The
modules import nothing (no WASI call is reached), so one file runs in Node, Deno, Bun,
Cloudflare-style Workers and browsers with `WebAssembly.instantiate(module, {})`.

## Building

```sh
# wasi-sdk 25 or later (clang 19), unpacked anywhere
WASI_SDK=/opt/wasi-sdk wasm/build.sh            # writes wasm/out/
```

| File | What it is |
| --- | --- |
| `rayo.wasm` | the library and the ABI below, generic C kernels (about 130 KB, 57 KB gzipped) |
| `rayo-simd.wasm` | the same with `-msimd128`: the SIMD128 int8 kernel, the rest autovectorised |
| `tests.wasm`, `tests-simd.wasm` | the C tests as WASI programs, for wasmtime |

Both modules are built `-std=c89 -O2 -ffp-contract=off` like every other build, with 16 MB
of initial memory (growing to at most 256 MB) and a 1 MB stack. SIMD128 runs in every
current browser, Node 16.4 and later, Deno and Workers; ship `rayo.wasm` as the fallback
where it may not (`WebAssembly.validate` on the SIMD module tells you).

## Kernels

| Build | int8 products | 4-bit, 3-bit, float32 products |
| --- | --- | --- |
| `rayo.wasm` | generic C | generic C |
| `rayo-simd.wasm` | `src/kernels/wasm_simd.c` (SIMD128, `i32x4.dot_i16x8_s`) | generic C, autovectorised |

WebAssembly has no way to ask what the CPU has, so the build decides: `rayo_kernels_simd`
and `rayo_kernels_best` both give `wasm-simd128` in the SIMD build. The kernel sums integers,
so its products are the generic kernel's exactly; `tests/test_kernels.c` checks it against
the generic kernel bit for bit, as it does the NEON and x86 kernels, and a model gives the
same floats from both modules and from native builds ([Same floats from every
build](../README.md#same-floats-from-every-build)).

On the Ask reader (`tests/ask`, 96 tokens, one thread, Node 22 on a 2.1 GHz Xeon) a run takes
about 6.1 ms with `rayo.wasm` and 1.0 ms with `rayo-simd.wasm`; native AVX-VNNI takes 0.5 ms.

## The C ABI

`wasm/rayo_wasm.c` exports a few functions whose arguments and results are all i32
(pointers, sizes and statuses), so a host needs no bindings generator:

| Export | Does |
| --- | --- |
| `rayo_wasm_alloc(n)`, `rayo_wasm_free(p)` | memory in the module, for the model's bytes and names |
| `rayo_wasm_open(bytes, size)` | a model and a session; NULL on failure. rayo copies what it needs, so free the bytes after |
| `rayo_wasm_inputs(h)`, `rayo_wasm_outputs(h)` | how many inputs and outputs |
| `rayo_wasm_bind(h, name, value)` | `rayo_session_bind`; 0 or a `rayo_status` |
| `rayo_wasm_input(h, i, &bytes)` | where input `i` lives and its size: write it in place |
| `rayo_wasm_run(h)` | `rayo_session_run`; 0 or a `rayo_status` |
| `rayo_wasm_output(h, i, &bytes)` | where output `i` lives after a run and its size |
| `rayo_wasm_kernels(h)` | the kernel set's name |
| `rayo_wasm_error()` | the last failure's text |
| `rayo_wasm_close(h)` | frees the session and the model |

The module also exports `memory` and `_initialize` (a reactor: call it once before anything
else). Pointers into memory stay valid until the memory grows, which only `rayo_wasm_alloc`
and `rayo_wasm_open` can make it do; read and write through fresh views after those.

## From JavaScript

`wasm/rayo.mjs` is a host for the ABI (about a hundred lines, no dependencies):

```js
import { loadRayo } from "./rayo.mjs";

const wasm = await fetch("rayo-simd.wasm").then((r) => r.arrayBuffer());
const rayo = await loadRayo(wasm);
const model = rayo.open(await fetch("ask-reader.weights").then((r) => r.arrayBuffer()));
// One typed array per input, each exactly its size at the bound dimensions.
const [slots, intent] = model.run([tokens, shapes], { binds: {} });
console.log(model.kernels(), intent); // "wasm-simd128", Float32Array
model.close();
```

`run` copies each output out of the module's memory (as `Float32Array` unless `as` names
other views, such as `{ as: ["i32"] }`). In Node:

```sh
node examples/wasm/ask_reader.mjs wasm/out/rayo.wasm wasm/out/rayo-simd.wasm
```

runs the Ask reader on the 30 questions TFLite's outputs were recorded for, checks every label
and every probability within 1e-4 (as `tests/test_ask.c` does natively), checks that the two
modules give the same floats, bit for bit, and times a run.

## Testing

`tool/ci/wasm.sh` (the `wasm` CI job, and `tool/check` when `WASI_SDK` is set and wasmtime
is installed) builds both variants, runs the whole C suite under wasmtime with each kernel set
(`RAYO_TEST_KERNELS=generic`, `simd`, `best`), which includes the golden models of
`tests/models` and the Ask models of `tests/ask`, then runs `examples/wasm/ask_reader.mjs` in
Node on both modules.

```sh
WASI_SDK=/opt/wasi-sdk WASMTIME=~/.wasmtime/bin/wasmtime tool/ci/wasm.sh
```

## Limits

- One thread: WebAssembly threads need shared memory and a host that allows it; rayo's
  threads are off in these builds.
- No mapped files: the host copies a model's bytes into the module, and rayo copies what it
  keeps. The Ask models are 75 to 270 KB; a language model of hundreds of megabytes needs a
  module built with a larger `--max-memory`.
- `wasm/rayo_wasm.c` is not part of the public C API (`include/rayo/`), and may change with
  the hosts that use it.
