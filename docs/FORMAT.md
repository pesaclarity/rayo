# rayo model format, version 1

A rayo model is a [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file
(version 3): ggml's own model file, holding typed metadata and tensors. rayo reads:

| Key | Type | |
| --- | --- | --- |
| `rayo.format_version` | uint32 | `1`. rayo refuses any other value. |
| `rayo.graph` | string | The graph, below. |
| `general.architecture` | string | `rayo` (informational). |
| `general.name` | string | The model's name (informational). |

Any other metadata is the model's own (labels, sizes, provenance) and is passed to the
caller as `RayoModel.metadata`. Writers should record where the weights came from, for
example `rayo.source.file` and `rayo.source.sha256`.

Tensors may be `F32`, `F16` or `Q8_0` (blocks of 32 int8 values with one float16 scale; the
innermost size must be a multiple of 32). Biases, norms and small tables are best kept `F32`.

## Shapes

As in ggml, shapes list the **innermost dimension first**. A sequence of `T` vectors of size
`C` is `[C, T]`, stored as `T` rows of `C` values; a dense layer's weight with `in` inputs
and `out` outputs is `[in, out]` (NumPy shape `(out, in)`).

Sizes in `input` and `state` lines, and in attributes, are integers, **size symbols**, or
sums and differences of them: `T`, `P+T`, `T-1`. The caller binds symbols before each run
(`RayoSession.bind`), so one graph runs on sequences of any length.

## The graph

Plain text, one statement per line; `#` starts a comment.

```
input  <name> <i32|f32> <size>...     # a tensor the caller sets before each run
state  <name> f32 <size>...           # kept across runs, zeroed by reset(); integer sizes
output <name> <node>                  # read with output(name) after a run, as float32
<node> = <op> <arg>... [<key>=<value>]...
```

Arguments name inputs, states, earlier nodes or tensors in the file. Nodes run in order.

## Ops

All results are float32 unless said otherwise.

| Op | Arguments | Attributes | Result |
| --- | --- | --- | --- |
| `get_rows` | table `[C, V]`, ids i32 `[T]` | | `[C, T]`: row `ids[i]` for each i |
| `range` | | `start`, `len` | i32 `[len]`: `start, start + 1, ...` |
| `mask_pos` | ids i32 `[T]` | | `[T]`: 1 where the id is above 0, else 0 |
| `concat` | a, b | `dim` (default 0) | a and b joined along `dim` |
| `dense` | x `[in, ...]`, w `[in, out]`, b `[out]` (optional) | `act` | `w x + b` |
| `conv1d` | x `[C, T]`, w `[k*C, out]`, b `[out]` (optional) | `k`, `d` (dilation, default 1), `act` | same-padded 1-D convolution over positions: tap j of position i reads position `i + (j - (k-1)/2) * d`, zeros outside; w's element (j, c) at `j*C + c` |
| `add` | a, b (broadcast to a) | `act` | a + b |
| `mul` | a, b (broadcast to a) | | a * b |
| `scale` | x | `s` | x * s |
| `relu`, `gelu`, `gelu_erf` | x | | activations (`gelu` is ggml's tanh approximation) |
| `softmax` | x | | softmax over the innermost dimension |
| `layer_norm` | x, g (optional), b (optional) | `eps` (default 1e-5) | normalised over the innermost dimension, `* g + b` |
| `masked_max`, `masked_mean` | x `[C, T]`, mask `[T]` | `empty` (masked_max; default 0) | `[C]`: max or mean over the positions where mask > 0; with none, `empty` (max) or 0 (mean) |
| `cols` | x `[C, T]` | `start`, `len` | columns (positions) `start .. start+len-1` |
| `write` | state `[C, N]`, x `[C, T]` | `at` | copies x into the state's columns `at .. at+T-1`, then gives columns `0 .. at+T-1`; runs before every later node |
| `attention` | q `[D, T]`, k `[D, S]`, v `[D, S]` | `heads`, `causal` (0/1), `scale` (default 1/sqrt(D/heads)) | multi-head scaled dot-product attention; the T queries sit at positions `S-T .. S-1`, and with `causal=1` each sees only keys at or before its own position |

`act` is `none` (default), `relu`, `gelu` or `gelu_erf`.

An unknown op fails the load. An unknown tensor or symbol, or shapes that do not fit, fail
the run with a message instead of crashing.

## A decoder with a key/value cache

```
input tokens i32 T
state kc f32 64 128          # keys for up to 128 positions
state vc f32 64 128
pos = range start=P len=T
h = add e pe                 # token and position embeddings, ...
k = dense h wk
v = dense h wv
ks = write kc k at=P
vs = write vc v at=P
a = attention q ks vs heads=4 causal=1
```

Run the prompt once with `P=0, T=n`, then each new token with `P=n, T=1`, `P=n+1, T=1`,
and so on; `reset()` before the next sequence.

## Versioning

A change that older rayo versions would misread (a changed meaning, a new statement kind)
bumps `rayo.format_version`. New ops do not: an older rayo refuses to load a graph with an
op it does not know, naming the op.
