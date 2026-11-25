# rayo developer guide

Everything about using rayo, from a first run to shipping it in an app. Start with
**Getting started**; the rest is reference to come back to.

## Start here

- [Getting started](GETTING-STARTED.md): a first run from C, from Dart and Flutter, and
  converting a model with rayo-convert.
- [Concepts](CONCEPTS.md): models, sessions, size symbols, states, the weights format,
  quantisation, kernels, threads and memory.

## Reference

- [C API](API-C.md): every public function and type, with its declaration.
- [Dart API](API-DART.md): every class, constructor and member of `package:rayo`.
- [Ops](OPS.md): the 41 built-in ops, their inputs and attributes, and ops of your own.
- [Model format](FORMAT.md): the plain-weights file, its header and graph, int4 and int3
  tensors, the tokeniser's table.

## Guides

- [Converting a model](CONVERTING.md): rayo-convert for TFLite, ONNX and Hugging Face models.
- [Running small language models](RUNNING-SLMS.md): from a Hugging Face model to a chat on
  an Android phone and a computer, and what to expect.
- [Performance and benchmarks](PERFORMANCE.md): where every number comes from, and measuring
  a phone.
- [Security](SECURITY.md): the threat model, what rayo guarantees, and signed models.
- [WebAssembly](WASM.md): rayo in Node, Workers and browsers, built with wasi-sdk, and its
  JavaScript host.
- [Licensing and embedding](LICENSING.md): AGPL-3.0 with the linking exception, for apps on
  every platform.

## Help

- [Troubleshooting](TROUBLESHOOTING.md): errors, their messages, and what to do.
- [FAQ](FAQ.md)
- [Roadmap](ROADMAP.md): native mobile packages, phone results and what comes next.
- [Releasing](RELEASING.md): how a maintainer makes a release.
