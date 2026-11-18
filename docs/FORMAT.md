# rayo model format

A rayo model is a **plain-weights file** (PCWT, version 1), the format pesaclarity-ml writes
for the app's models ([pesaclarity-ml docs/WEIGHTS.md](https://github.com/pesaclarity/pesaclarity-ml/blob/dev/docs/WEIGHTS.md)),
whose JSON header also holds a **`graph`**: the ops that run the tensors. The graph is an
optional header field, which version 1 allows ("unknown header fields are ignored"), so one
file serves rayo and any other reader of version 1 alike. `rayo-convert` writes one from a
TFLite or ONNX model ([CONVERTING.md](CONVERTING.md)).

```
"PCWT" | uint32 version (1) | uint32 header length H | UTF-8 JSON header (H bytes) | tensors
```

All little-endian; the tensor data starts on a 4-byte boundary. rayo checks every size and
offset against the file before it uses them, and refuses what does not fit rather than
guessing.

## Header fields rayo reads

| Field | |
| --- | --- |
| `format_version` | `1`, as bytes 4-7. |
| `model` | the model's name, such as `ask-reader`. |
| `tensors` | as WEIGHTS.md: `name`, `dtype` (`float32`, `int8` or `int4`), row-major `shape`, `offset`, `bytes`, and for int8 a `quantization` (`axis` or null, `zero_point`, `scales_offset`, `scales_bytes`); for int4 see below. |
| `graph` | the graph, below. Required by rayo. |
| `metadata` | optional object, passed to the caller as JSON (labels, sizes, provenance). |

## int4 tensors

Weight-only 4-bit tensors, for language models, in groups along the innermost dimension:

```json
{"name": "l0.q", "dtype": "int4", "shape": [576, 576], "offset": 0, "bytes": 165888,
 "quantization": {"group_size": 32, "zero_point": 8, "scales_offset": 165888,
                  "scales_bytes": 41472}}
```

- **Values:** two a byte, `bytes = count / 2`. Each row is cut into groups of `group_size` (32:
  the only size rayo reads; the innermost dimension must be a multiple of it), stored one after
  the other in 16 bytes each. **Byte j of a group holds value j in its low four bits and value
  j + 16 in its high four**, each stored plus 8 (0 to 15 for -8 to 7).
- **Scales:** float32, one per group, row by row (`count / group_size` of them,
  `scales_bytes = count / group_size * 4`), at `scales_offset`.
- **Real value:** `scale[row][group] * (stored - 8)`.

`rayo-convert` maps the value of largest magnitude in each group to -8, so the scale is that
value / -8 (as llama.cpp's Q4_0 does, with float32 scales instead of float16 ones: 5 bits a
weight in all, not 4.5).

A reader of version 1 that does not know `int4` refuses the file, naming the dtype; it does
not misread it. So the version stays 1.

## Alignment

Tensor offsets need only be multiples of 4. `rayo-convert` writes language models with the
data, and every tensor and scale table, starting at multiples of 64 bytes from the start of
the file (the header is padded with spaces), so a file mapped with `rayo_model_load` gives
the kernels aligned weights. rayo reads unaligned files as well: every SIMD load is an
unaligned one.

**Size limits.** Offsets and sizes are 32-bit (C89 has no 64-bit integers), so a file holds
at most 2 GiB of data: a 0.5B-parameter model in float32 just fits, a 1.5B one only in int8
or int4. A later version may write offsets as two numbers.

## The graph (version 1)

```json
"graph": {
  "version": 1,
  "symbols": {"T": 96},
  "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]}],
  "nodes": [
    {"op": "gather", "inputs": ["token_embedding.table", "tokens"], "output": "emb"},
    {"op": "conv1d", "inputs": ["emb", "conv_1.kernel", "conv_1.bias"], "output": "c1",
     "size": 3, "dilation": 1, "activation": "relu", "input_quantization": "tensor_phases"}
  ],
  "outputs": [{"name": "slots", "node": "slots"}]
}
```

- `symbols`: size symbols and the largest value of each. The caller binds each (0 to its
  largest) before a run; an input may not come out empty.
- `plan` (optional): the symbol values a session plans its memory for, each symbol's largest by
  default. Set it when symbols trade off: a decoder's past `P` and new `T` add up to at most its
  cache's length, so it plans with `P` 0 and `T` the length. A bind whose sizes would need more
  memory than planned is refused.
- `inputs`: `dtype` `int32` or `float32`; `shape` lists whole numbers or symbols.
- `states` (optional): float32 tensors with whole-number shapes that a session keeps across
  runs, such as a key/value cache. `rayo_session_reset` zeroes them.
- `nodes`: run in order. `inputs` name graph inputs, states, earlier nodes' outputs, or
  tensors of the file; the other fields are the op's attributes. An attribute that is a size
  may be a whole number or a symbol's name.
- `outputs`: which nodes the caller reads, as float32.

Shapes are row-major, outermost first: a sequence of `T` vectors of size `C` is `[T, C]`.

## Ops

Each node's `op` is one of rayo's built-in ops, documented with its attributes, shapes and
int8 behaviour in [docs/OPS.md](OPS.md), or one the program gave rayo when it loaded the model
(`include/rayo/op.h`). A graph that names any other op fails to load, naming it.

## A decoder

A transformer decoder (the Ask writer's graph, for example) decodes token by token with a
key/value cache per layer:

```json
"symbols": {"T": 64, "P": 63}, "plan": {"P": 0},
"inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]}],
"states": [{"name": "keys_0", "dtype": "float32", "shape": [64, 128]}, ...],
"nodes": [
  {"op": "range", "inputs": [], "output": "pos", "start": "P", "size": "T"},
  ...
  {"op": "cache_write", "inputs": ["keys_0", "k0"], "output": "ks0", "at": "P"},
  {"op": "attention", "inputs": ["q0", "ks0", "vs0"], "output": "h0", "heads": 4, "causal": 1},
  ...
]
```

Reset, run the prompt with `P = 0, T = n`, then each new token with `T = 1` at `P = n, n+1, ...`.

## Versions

A change that older rayo versions would misread bumps `graph.version`; rayo refuses a version
it does not read, and a graph with an op it does not know, naming it. New ops and attributes
with defaults that keep the old meaning do not bump it.
