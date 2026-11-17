# rayo's ops

Every op a graph can name (docs/FORMAT.md), with its inputs, attributes, output, and what it
does with int8 weights. A program can add its own ops through `include/rayo/op.h` (see the
end of this page). `rayo_op_builtin(i)` lists the built-in ones at run time.

Notes for every op:

- Shapes are row-major, outermost first, and have at most `RAYO_MAX_RANK` dimensions (4 by
  default; `include/rayo/config.h`).
- Activations are float32 unless an op says otherwise.
- An input is either an activation (a graph input, a state, or an earlier node) or a weight
  (a tensor of the file). Each op says which inputs must be weights.
- A *size* attribute is a whole number from 0 to `RAYO_MAX_DIM`, or the name of a size
  symbol, which is read at each run.
- Every op checks its inputs when the model loads (`prepare`) and at every bind (`infer`),
  and fails with a message that names the node. A run allocates nothing.

## Elementwise

### `add`

| | |
| --- | --- |
| Inputs | a, b: float32, of one shape |
| Output | a + b |

### `compare`

| | |
| --- | --- |
| Inputs | x: float32 or int32 |
| Attributes | `comparison`: `greater`, `greater_equal`, `less`, `less_equal`, `equal` or `not_equal` (required). `value`: a number (default 0). `dtype`: of the output, `float32` (default) or `int32` |
| Output | x's shape: 1 where `x <comparison> value` holds, else 0 |

The comparison is made in double. A padding mask over token ids, where id 0 is padding, is
`{"op": "compare", "comparison": "greater", "value": 0}`.

### `softmax`

| | |
| --- | --- |
| Inputs | x: float32, rank 1 or more |
| Output | x's shape: the softmax over the innermost dimension (the maximum subtracted first) |

## Lookups and joins

### `gather`

| | |
| --- | --- |
| Inputs | table: a weight `[rows, width]`, float32, or int8 with one scale. ids: int32 `[n]` |
| Output | float32 `[n, width]`: row `ids[i]` of the table for each i |

An int8 table is dequantised as each row is read (`q * scale`). An id outside the table fails
the run with `RAYO_ERR_SHAPE`.

### `concat`

| | |
| --- | --- |
| Inputs | a `[..., p]`, b `[..., q]`: float32, equal outside the innermost dimension |
| Output | `[..., p + q]`: a's values, then b's |

## Reductions

### `masked_reduce`

| | |
| --- | --- |
| Inputs | x: float32. mask: float32, one value per position of the reduced axis |
| Attributes | `reduction`: `max`, `sum` or `mean` (required). `axis`: the axis reduced (default 0; negative counts from the end). `fill`: optional, see below. `empty`: the result where nothing counts (default 0) |
| Output | x's shape without the axis |

- **Which positions count:** a position is kept where its mask value is not 0. Without
  `fill`, the positions the mask drops are left out. With `fill`, each dropped position
  counts as the value `fill`.
- **`mean`:** the sum of the positions that count, divided by how many there are.
- **Nothing counts:** the result is `empty`.

Keras models often mask a max by subtracting a large number from the padding,
`max(x * m - (1 - m) * 1e4)`. For a 0/1 mask that is `"reduction": "max", "fill": -10000`,
bit for bit. A masked average, `sum(x * m) / max(sum(m), 1)`, is `"reduction": "mean"`.

### `last_row`

| | |
| --- | --- |
| Inputs | x `[T, C]` |
| Output | `[C]`: row T - 1 |

## Linear layers

`dense` and `conv1d` take float32 or int8 kernels.

**Float32 kernels:** the product is plain float32.

**int8 kernels** are symmetric, with one scale per output or one scale in all. They run as
TFLite's dynamic-range ("hybrid") kernels do, so a model exported from TFLite gives TFLite's
numbers:

1. **Quantise the input.** The float input is quantised to int8 on the fly, asymmetrically,
   as TFLite's `AsymmetricQuantizeFloats` does. The range is widened to include 0, and
   halves round away from zero. A range that is not finite quantises to zeros. The parts
   quantised together depend on the op's `input_quantization`.
2. **Multiply.** The products are summed exactly in int32. That is why a kernel's inner
   dimension is limited to `RAYO_I32_MAX / (256 * 128)`.
3. **Dequantise.** `y = bias + (acc - zero_point * rowsum(kernel)) * input_scale *
   kernel_scale`, then the activation.

`activation` is `none` (the default), `relu`, or `gelu_erf` (`0.5 x (1 + erf(x / sqrt 2))`,
Keras' `gelu`). It is fused into the layer.

### `dense`

| | |
| --- | --- |
| Inputs | x `[..., in]`; kernel: a weight `[out, in]`; bias: a weight `[out]` (optional) |
| Attributes | `activation`; `input_quantization`: `rows` (default), one scale and zero point per row of x |
| Output | `[..., out]`: `act(x · kernelᵀ + bias)` |

### `conv1d`

| | |
| --- | --- |
| Inputs | x `[T, C]`; kernel: a weight `[out, 1, size, C]` (TFLite's layout) or `[out, size, C]`; bias `[out]` (optional) |
| Attributes | `size`: the kernel's taps, which must match the kernel. `dilation`: default 1. `activation`. `input_quantization`: `tensor_phases` (default) |
| Output | `[T, out]`, same padding: tap j of position t reads position `t + (j - (size - 1) / 2) * dilation`, and positions outside the sequence read zeros |

- **Limits:** `(size - 1) * dilation` must be even, to keep the output centred, and below
  2^30.
- **`tensor_phases`:** the input is quantised with one scale for all of it. A dilated
  convolution quantises each of its `dilation` phases (the positions with `t % dilation ==
  p`) on its own, as TFLite does when it runs one as `SPACE_TO_BATCH_ND` and a convolution.
- **Padding:** positions outside the sequence count as float zeros, which is the zero point
  in int8.

### `layer_norm`

| | |
| --- | --- |
| Inputs | x `[..., C]`; gamma, beta: float32 weights `[C]` |
| Attributes | `epsilon` (default 1e-5) |
| Output | x's shape, normalised over the innermost dimension |

It follows TFLite's arithmetic for Keras' `LayerNormalization`: the mean and the variance,
then `m = rsqrt(var + eps) * gamma`, then `x * m + (beta - mean * m)`.

## Sequences and decoding

### `range`

| | |
| --- | --- |
| Inputs | none |
| Attributes | `start`, `size`: sizes |
| Output | int32 `[size]`: start, start + 1, ... |

### `cache_write`

| | |
| --- | --- |
| Inputs | a state `[N, C]`; x `[T, C]` |
| Attributes | `at`: a size |
| Output | rows `0 .. at + T - 1` of the state, a view with no copy |

It writes x into rows `at .. at + T - 1` of the state. It is how a decoder keeps each
layer's keys and values across runs. `at + T` must be at most N.

### `attention`

| | |
| --- | --- |
| Inputs | q `[T, D]`; k, v `[S, D]` with S ≥ T |
| Attributes | `heads` (default 1; D must divide by it). `causal`: 0 or 1. `scale`: default `1 / sqrt(D / heads)` |
| Output | `[T, D]` |

- **Per head:** `softmax(q · kᵀ * scale) · v`. Each head uses its own D / heads of the
  dimensions.
- **Query positions:** the T queries are at positions `S - T .. S - 1`, so the earlier keys
  can come from a cache.
- **`causal`:** a query sees only the keys at or before its own position. A key it may not
  see gets weight 0, which is exactly what TFLite's `-1e9` mask gives in float32.

## Ops of your own

`include/rayo/op.h` lets a program give rayo ops when it loads a model:

```c
static rayo_status scale_prepare(void *user, const rayo_op_node *node, const void **params,
                                 rayo_message *msg);
static rayo_status scale_infer(const void *params, const rayo_tensor_view *in, int n,
                               rayo_tensor_view *out, size_t *scratch, rayo_message *msg);
static rayo_status scale_run(const void *params, const rayo_tensor_view *in, int n,
                             rayo_tensor_view *out, void *scratch);

rayo_custom_op scale;
const rayo_custom_op *ops[1];
rayo_model_options options;
memset(&scale, 0, sizeof scale);
scale.api_version = RAYO_OP_API_VERSION;
scale.name = "scale";
scale.min_inputs = scale.max_inputs = 1;
scale.prepare = scale_prepare;
scale.infer = scale_infer;
scale.run = scale_run;
ops[0] = &scale;
memset(&options, 0, sizeof options);
options.ops = ops;
options.op_count = 1;
status = rayo_model_create_with(bytes, size, &options, &model, &msg);
```

- **prepare** reads the node's attributes (`rayo_op_node_int`, `_number`, `_string`) and
  its weights (`rayo_op_node_input`), and allocates its parameters with
  `rayo_op_node_alloc`, which lives as long as the model.
- **infer** sets the output's dtype and shape, and the scratch bytes a run needs.
- **run** computes the output. It must not allocate. The parameters are shared by every
  session, so anything a run changes belongs in `scratch`.
- **Rules:** an op's name may not be a built-in op's. It may take at most
  `RAYO_OP_MAX_INPUTS` inputs (16 by default). rayo refuses an op written for another
  `RAYO_OP_API_VERSION`.
- **Unknown ops:** a graph that names an op neither rayo nor the program has fails to load,
  with the op's name in the message.
- **Example:** `tests/test_ops.c` has a complete one.
