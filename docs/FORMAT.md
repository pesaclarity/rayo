# rayo model format

A rayo model is a **plain-weights file** (PCWT, version 1), the format pesaclarity-ml writes
for the app's models ([pesaclarity-ml docs/WEIGHTS.md](https://github.com/pesaclarity/pesaclarity-ml/blob/dev/docs/WEIGHTS.md)),
whose JSON header also holds a **`graph`**: the ops that run the tensors. The graph is an
optional header field, which version 1 allows ("unknown header fields are ignored"), so one
file serves rayo and any other reader of version 1 alike.

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
| `tensors` | as WEIGHTS.md: `name`, `dtype` (`float32` or `int8`), row-major `shape`, `offset`, `bytes`, and for int8 a `quantization` (`axis` or null, `zero_point`, `scales_offset`, `scales_bytes`). |
| `graph` | the graph, below. Required by rayo. |
| `metadata` | optional object, passed to the caller as JSON (labels, sizes, provenance). |

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

- `symbols`: size symbols and the largest value of each. A session plans its memory for the
  largest sizes once; the caller binds smaller ones before a run.
- `inputs`: `dtype` `int32` or `float32`; `shape` lists whole numbers or symbols.
- `nodes`: run in order. `inputs` name graph inputs, earlier nodes' outputs, or tensors of the
  file; the other fields are the op's attributes.
- `outputs`: which nodes the caller reads, as float32.

Shapes are row-major, outermost first: a sequence of `T` vectors of size `C` is `[T, C]`.

## Ops

| Op | Inputs | Attributes | Output |
| --- | --- | --- | --- |
| `gather` | table `[rows, width]` (float32, or int8 with one scale), ids int32 `[n]` | | float32 `[n, width]`; an id outside the table fails the run |
| `concat` | a `[..., p]`, b `[..., q]` | | `[..., p + q]` |
| `mask` | ids int32 | | 1 where the id is above 0, else 0 |
| `add` | a, b of one shape | | a + b |
| `softmax` | x | | over the innermost dimension |
| `dense` | x `[..., in]`, kernel `[out, in]`, bias `[out]` (optional) | `activation` (`none`, `relu`), `input_quantization` | `[..., out]` |
| `conv1d` | x `[T, C]`, kernel `[out, 1, size, C]` (TFLite's) or `[out, size, C]`, bias | `size`, `dilation` (default 1), `activation`, `input_quantization` | `[T, out]`, same padding: tap j of position t reads `t + (j - (size-1)/2) * dilation`, zeros outside |
| `masked_max` | x `[T, C]`, mask `[T]` | `fill` (default 10000) | `[C]`: max over t of `x * m - (1 - m) * fill` |
| `masked_mean` | x `[T, C]`, mask `[T]` | | `[C]`: sum over t of `x * m`, divided by max(sum of m, 1) |

### int8 kernels

A `dense` or `conv1d` with an int8 kernel (symmetric, one scale per output or one in all) runs
as TFLite's dynamic-range ("hybrid") kernels do, so a model exported from TFLite gives
TFLite's numbers:

1. the float input is quantised to int8 on the fly, asymmetrically, its range widened to
   include 0, halves rounded away from zero (TFLite's `AsymmetricQuantizeFloats`):
   - `input_quantization: "rows"` (dense, the default): one scale and zero point per row;
   - `input_quantization: "tensor_phases"` (conv1d, the default): one for the whole input,
     except that a dilated convolution (TFLite's `SPACE_TO_BATCH_ND`) quantises each of its
     `dilation` phases (positions t with `t % dilation == p`) on its own;
2. the product is exact in int32;
3. `y = bias + (acc - zero_point * rowsum(kernel)) * input_scale * kernel_scale`, then the
   activation.

Positions outside the sequence count as float zeros (the zero point in int8).

## Versions

A change that older rayo versions would misread bumps `graph.version`; rayo refuses a version
it does not read, and a graph with an op it does not know, naming it. New ops and attributes
with defaults that keep the old meaning do not bump it.
