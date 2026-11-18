"""Synthetic models for the tests, in the forms TFLite emits for Keras layers: a dense
classifier, a reader of token ids (embeddings, same and dilated 1-D convolutions, masked
pooling), and transformer decoders with attention written two ways (one fused qkv projection
as a custom Keras layer exports it; Keras MultiHeadAttention's separate projections)."""

from __future__ import annotations

from typing import Any

import numpy as np

from tflite_build import Builder

RNG = np.random.default_rng(7)


def rnd(*shape: int, scale: float = 0.5) -> np.ndarray:
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def dense(b: Builder, x: int, rows: list[int], n_in: int, n_out: int, name: str,
          int8: bool = True, act: int = 0, bias_after: bool = False) -> int:
    """FULLY_CONNECTED as TFLite writes a Keras Dense: kernel [out, in], keep_num_dims."""
    w = rnd(n_out, n_in)
    k = b.int8(f"m/{name}_1/MatMul", w) if int8 else b.const(f"m/{name}_1/MatMul", w)
    bias = b.const(f"m/{name}_1/BiasAdd", rnd(n_out, scale=0.1))
    shape = [*rows, n_out]
    if bias_after:  # a Tensordot: the bias in an ADD after it, the activation fused there
        y = b.add("FULLY_CONNECTED", [x, k, -1], f"m/{name}_1/Tensordot", shape,
                  keep_num_dims=True, asymmetric_quantize_inputs=True)
        return b.add("ADD", [y, bias], f"m/{name}_1/BiasAdd;Relu", shape,
                     fused_activation_function=act)
    return b.add("FULLY_CONNECTED", [x, k, bias], f"m/{name}_1/MatMul;BiasAdd", shape,
                 keep_num_dims=True, asymmetric_quantize_inputs=True,
                 fused_activation_function=act)


def layer_norm(b: Builder, x: int, shape: list[int], name: str) -> int:
    """Keras' LayerNormalization as TFLite decomposes it."""
    c = shape[-1]
    p = f"m/{name}_1/"
    one = [*shape[:-1], 1]
    axis = b.const(p + "axis", [-1])
    mean = b.add("MEAN", [x, axis], p + "moments/mean", one, keep_dims=True)
    neg = b.add("NEG", [mean], p + "Neg", one)
    sq = b.add("SQUARED_DIFFERENCE", [x, mean], p + "moments/SquaredDifference", shape)
    var = b.add("MEAN", [sq, axis], p + "moments/variance", one, keep_dims=True)
    ve = b.add("ADD", [var, b.const(p + "eps", np.float32(1e-5))], p + "add", one)
    r = b.add("RSQRT", [ve], p + "Rsqrt", one)
    gamma = b.const(p + "Reshape", (1 + rnd(1, 1, c, scale=0.1)))
    beta = b.const(p + "Reshape_1", rnd(1, 1, c, scale=0.1))
    m = b.add("MUL", [r, gamma], p + "mul", shape)
    t = b.add("MUL", [neg, m], p + "mul_1", shape)
    sh = b.add("ADD", [t, beta], p + "add_1", shape)
    xm = b.add("MUL", [x, m], p + "mul_2", shape)
    return b.add("ADD", [xm, sh], p + "add_2", shape)


def causal_mask(t: int, value: float = -1e9) -> np.ndarray:
    return np.where(np.tril(np.ones((t, t))) > 0, 0.0, value).astype(np.float32)


def attention_fused(b: Builder, x: int, t: int, d: int, heads: int, name: str,
                    mask: Any = "causal", scale: Any = None, int8: bool = True) -> int:
    """A custom layer's attention: one qkv dense, RESHAPE [1, T, 3, H, hd], TRANSPOSE
    [2, 0, 3, 1, 4], a STRIDED_SLICE each, the scores scaled and masked, the heads back."""
    hd = d // heads
    p = f"m/{name}_1/"
    if isinstance(mask, str):
        mask = causal_mask(t)
    qkv = dense(b, x, [1, t], d, 3 * d, f"{name}_1/qkv", int8)
    r = b.add("RESHAPE", [qkv, b.const(p + "c_shape", [1, t, 3, heads, hd])], p + "Reshape",
              [1, t, 3, heads, hd])
    tr = b.add("TRANSPOSE", [r, b.const(p + "c_perm", [2, 0, 3, 1, 4])], p + "transpose",
               [3, 1, heads, t, hd])
    parts = []
    for i, nm in enumerate(("strided_slice", "strided_slice_1", "strided_slice_2")):
        parts.append(b.add("STRIDED_SLICE", [tr, b.const(p + f"b{i}", [i, 0, 0, 0, 0]),
                                             b.const(p + f"e{i}", [i + 1, 0, 0, 0, 0]),
                                             b.const(p + "s", [1, 1, 1, 1, 1])],
                           p + nm, [1, heads, t, hd], begin_mask=30, end_mask=30,
                           shrink_axis_mask=1))
    q, k, v = parts
    s = b.add("BATCH_MATMUL", [q, k], p + "MatMul", [1, heads, t, t], adj_y=True)
    c = np.float32(1.0 / np.sqrt(hd)) if scale is None else scale
    s = b.add("MUL", [s, b.const(p + "c_scale", np.asarray(c, np.float32))], p + "truediv",
              [1, heads, t, t])
    if mask is not None:
        s = b.add("ADD", [s, b.const(p + "c_mask", mask.reshape(1, 1, t, t))], p + "add",
                  [1, heads, t, t])
    w = b.add("SOFTMAX", [s], p + "Softmax", [1, heads, t, t], beta=1.0)
    o = b.add("BATCH_MATMUL", [w, v], p + "MatMul_1", [1, heads, t, hd])
    o = b.add("TRANSPOSE", [o, b.const(p + "c_perm2", [0, 2, 1, 3])], p + "transpose_2",
              [1, t, heads, hd])
    o = b.add("RESHAPE", [o, b.const(p + "c_shape2", [t, d])], p + "Reshape_1", [t, d])
    return dense(b, o, [t], d, d, f"{name}_1/out", int8)


def attention_mha(b: Builder, x: int, t: int, d: int, heads: int, name: str,
                  causal: bool = True, mask_value: float = -1e9, select: bool = True,
                  post: bool = True, int8: bool = False) -> int:
    """Keras MultiHeadAttention's form: query, key and value denses, each RESHAPE [1, T, H, hd]
    and TRANSPOSE [0, 2, 1, 3]; the query scaled before the scores; the mask by SELECT_V2 (or
    an ADD) and the weights multiplied by it after the softmax; the output dense."""
    hd = d // heads
    p = f"m/{name}_1/"
    heads_of = []
    for role in ("query", "key", "value"):
        y = dense(b, x, [1, t], d, d, f"{name}_1/{role}", int8)
        y = b.add("RESHAPE", [y, b.const(p + role + "_shape", [1, t, heads, hd])],
                  p + role + "/Reshape", [1, t, heads, hd])
        y = b.add("TRANSPOSE", [y, b.const(p + role + "_perm", [0, 2, 1, 3])],
                  p + role + "/transpose", [1, heads, t, hd])
        heads_of.append(y)
    q, k, v = heads_of
    q = b.add("MUL", [q, b.const(p + "c_scale", np.float32(1.0 / np.sqrt(hd)))], p + "Mul",
              [1, heads, t, hd])
    s = b.add("BATCH_MATMUL", [q, k], p + "einsum/Einsum", [1, heads, t, t], adj_y=True)
    allowed = np.tril(np.ones((t, t))) > 0 if causal else np.ones((t, t), bool)
    if select:
        s = b.add("SELECT_V2", [b.const(p + "c_mask", allowed.reshape(1, 1, t, t), "bool"), s,
                                b.const(p + "c_neg", np.float32(mask_value))],
                  p + "softmax/SelectV2", [1, heads, t, t])
    else:
        add = np.where(allowed, 0.0, mask_value).astype(np.float32)
        s = b.add("ADD", [s, b.const(p + "c_add", add.reshape(1, 1, t, t))], p + "softmax/add",
                  [1, heads, t, t])
    w = b.add("SOFTMAX", [s], p + "softmax/Softmax", [1, heads, t, t], beta=1.0)
    if post:
        w = b.add("MUL", [w, b.const(p + "c_ones", allowed.astype(np.float32).reshape(
            1, 1, t, t))], p + "softmax/mul", [1, heads, t, t])
    o = b.add("BATCH_MATMUL", [w, v], p + "einsum_1/Einsum", [1, heads, t, hd])
    o = b.add("TRANSPOSE", [o, b.const(p + "c_perm_o", [0, 2, 1, 3])], p + "transpose",
              [1, t, heads, hd])
    o = b.add("RESHAPE", [o, b.const(p + "c_shape_o", [1, t, d])], p + "Reshape", [1, t, d])
    return dense(b, o, [1, t], d, d, f"{name}_1/attention_output", int8)


def decoder(t: int = 16, d: int = 32, heads: int = 4, vocab: int = 40, layers: int = 2,
            style: str = "fused", int8: bool = True, **att: Any) -> bytes:
    """tokens [1, T] -> logits [1, T, vocab]: a pre-norm transformer decoder."""
    b = Builder()
    tokens = b.input("tokens", [1, t], "int32")
    zero = b.const("m/token_embedding_1/zero", np.int32(0))
    less = b.add("LESS", [tokens, zero], "m/token_embedding_1/Less", [1, t], "bool")
    wrap = b.add("ADD", [tokens, b.const("m/token_embedding_1/n", np.int32(vocab))],
                 "m/token_embedding_1/add", [1, t], "int32")
    ids = b.add("SELECT_V2", [less, wrap, tokens], "m/token_embedding_1/SelectV2", [1, t],
                "int32")
    table = b.int8("m/token_embedding_1/GatherV2/ReadVariableOp", rnd(vocab, d), axis=None)
    e = b.add("GATHER", [table, ids], "m/token_embedding_1/GatherV2", [1, t, d], "int8")
    b.tensors[e]["scales"] = b.tensors[table]["scales"]
    e = b.add("DEQUANTIZE", [e], "m/token_embedding_1/deq", [1, t, d])
    x = b.add("ADD", [e, b.const("m/position_embedding_1/strided_slice", rnd(1, t, d))],
              "m/position_embedding_1/add", [1, t, d])
    for i in range(layers):
        a = layer_norm(b, x, [1, t, d], f"ln_{i}_a")
        if style == "fused":
            h = attention_fused(b, a, t, d, heads, f"attention_{i}", int8=int8, **att)
        else:
            h = attention_mha(b, a, t, d, heads, f"attention_{i}", int8=int8, **att)
        x = b.add("ADD", [x, h], f"m/add_{i}", [1, t, d])
        f = layer_norm(b, x, [1, t, d], f"ln_{i}_b")
        f = dense(b, f, [1, t], d, 2 * d, f"ff_{i}_in", int8)
        f = b.add("GELU", [f], f"m/ff_{i}_in_1/Gelu", [1, t, 2 * d], approximate=False)
        f = dense(b, f, [1, t], 2 * d, d, f"ff_{i}_out", int8)
        x = b.add("ADD", [x, f], f"m/add_{i}_b", [1, t, d])
    x = layer_norm(b, x, [1, t, d], "ln_final")
    logits = dense(b, x, [1, t], d, vocab, "logits", int8)
    b.output(logits, "logits")
    return b.build()


def classifier(int8: bool = True) -> bytes:
    """features [1, 12] -> probabilities [1, 5]: two dense layers and a softmax."""
    b = Builder()
    x = b.input("features", [1, 12])
    h = dense(b, x, [1], 12, 16, "hidden", int8, act=1)
    y = dense(b, h, [1], 16, 5, "out", int8)
    p = b.add("SOFTMAX", [y], "m/out_1/Softmax", [1, 5], beta=1.0)
    b.output(p, "probabilities")
    return b.build()


def conv(b: Builder, x: int, t: int, c: int, o: int, size: int, name: str,
         dilation: int = 1) -> int:
    """A Keras Conv1D, same padding, relu: EXPAND_DIMS, CONV_2D, RESHAPE; dilated as
    SPACE_TO_BATCH_ND, CONV_2D (valid), BATCH_TO_SPACE_ND and the bias in an ADD."""
    p = f"m/{name}_1/"
    k = b.int8(p + "convolution", rnd(o, 1, size, c))
    bias = rnd(o, scale=0.1)
    if dilation == 1:
        e = b.add("EXPAND_DIMS", [x, b.const(p + "dim", np.int32(1))], p + "ExpandDims",
                  [1, 1, t, c])
        y = b.add("CONV_2D", [e, k, b.const(p + "Relu;BiasAdd", bias)], p + "Relu;conv",
                  [1, 1, t, o], padding=0, stride_w=1, stride_h=1, dilation_w_factor=1,
                  dilation_h_factor=1, fused_activation_function=1)
        return b.add("RESHAPE", [y, b.const(p + "Squeeze", [1, t, o])], p + "Squeeze",
                     [1, t, o])
    d = dilation
    pad = (size - 1) * d // 2
    total = t + 2 * pad
    extra = -total % d
    s2b = b.add("SPACE_TO_BATCH_ND", [x, b.const(p + "block", [d]),
                                      b.const(p + "paddings", [[pad, pad + extra]])],
                p + "SpaceToBatchND", [d, (total + extra) // d, c])
    e = b.add("EXPAND_DIMS", [s2b, b.const(p + "dim", np.int32(1))], p + "ExpandDims",
              [d, 1, (total + extra) // d, c])
    n = (total + extra) // d - size + 1
    y = b.add("CONV_2D", [e, k, b.const(p + "zeros", np.zeros(o, np.float32))],
              p + "convolution1", [d, 1, n, o], padding=1, stride_w=1, stride_h=1,
              dilation_w_factor=1, dilation_h_factor=1)
    y = b.add("RESHAPE", [y, b.const(p + "sq", [d, n, o])], p + "Squeeze1", [d, n, o])
    y = b.add("BATCH_TO_SPACE_ND", [y, b.const(p + "block2", [d]),
                                    b.const(p + "crops", [[0, extra]])],
              p + "BatchToSpaceND", [1, t, o])
    return b.add("ADD", [y, b.const(p + "Squeeze", bias)], p + "Relu;BiasAdd", [1, t, o],
                 fused_activation_function=1)


def reader(t: int = 24, vocab: int = 50, split: bool = True) -> bytes:
    """question [1, 2T] (ids, then shapes) -> intent [1, 6], slots [1, T, 4]: embeddings,
    convolutions, masked max and mean pooling, as the Ask reader."""
    b = Builder()
    q = b.input("question", [1, 2 * t], "int32")
    tok = b.add("STRIDED_SLICE", [q, b.const("m/b0", [0, 0]), b.const("m/e0", [0, t]),
                                  b.const("m/s", [1, 1])], "m/strided_slice_1", [1, t], "int32",
                begin_mask=1, end_mask=1)
    shp = b.add("STRIDED_SLICE", [q, b.const("m/b1", [0, t]), b.const("m/e1", [0, 2 * t]),
                                  b.const("m/s1", [1, 1])], "m/strided_slice", [1, t], "int32",
                begin_mask=1, end_mask=1)
    if not split:
        shp = tok
    mask = b.add("GREATER", [shp, b.const("m/zero", np.int32(0))], "m/Greater", [1, t], "bool")
    mask = b.add("CAST", [mask], "m/Cast", [1, t])
    mask = b.add("EXPAND_DIMS", [mask, b.const("m/dim", np.int32(-1))], "m/ExpandDims",
                 [1, t, 1])
    te = b.int8("m/token_embedding_1/GatherV2/ReadVariableOp", rnd(vocab, 12), axis=None)
    e1 = b.add("GATHER", [te, tok], "m/token_embedding_1/GatherV2", [1, t, 12], "int8")
    b.tensors[e1]["scales"] = b.tensors[te]["scales"]
    e1 = b.add("DEQUANTIZE", [e1], "m/token_embedding_1/deq", [1, t, 12])
    se = b.const("m/shape_embedding_1/GatherV2/ReadVariableOp", rnd(6, 4))
    e2 = b.add("GATHER", [se, shp], "m/shape_embedding_1/GatherV2", [1, t, 4])
    emb = b.add("CONCATENATION", [e1, e2], "m/concatenate_1/concat", [1, t, 16], axis=-1)
    c1 = conv(b, emb, t, 16, 24, 3, "conv_1")
    c2 = conv(b, c1, t, 24, 24, 3, "conv_2", dilation=2)
    x = b.add("ADD", [c1, c2], "m/add_1/Add", [1, t, 24])
    # masked mean and max, as Keras writes them by hand
    one = b.const("m/one", np.float32(1.0))
    inv = b.add("SUB", [one, mask], "m/Sub", [1, t, 1])
    big = b.add("MUL", [inv, b.const("m/big", np.float32(1e4))], "m/Mul_3", [1, t, 1])
    ax = b.const("m/axis", [1])
    cnt = b.add("SUM", [mask, ax], "m/Sum", [1, 1])
    cnt = b.add("MAXIMUM", [cnt, one], "m/Maximum", [1, 1])
    xm = b.add("MUL", [x, mask], "m/Mul_1", [1, t, 24])
    mean = b.add("DIV", [b.add("SUM", [xm, ax], "m/Sum_2", [1, 24]), cnt], "m/truediv", [1, 24])
    top = b.add("REDUCE_MAX", [b.add("SUB", [xm, big], "m/Sub_1", [1, t, 24]), ax], "m/Max",
                [1, 24])
    pooled = b.add("CONCATENATION", [top, mean], "m/concatenate_2_1/concat", [1, 48], axis=-1)
    h = dense(b, pooled, [1], 48, 16, "intent_hidden", act=1)
    intent = b.add("SOFTMAX", [dense(b, h, [1], 16, 6, "intent")], "m/intent_1/Softmax",
                   [1, 6], beta=1.0)
    xe = b.add("CONCATENATION", [x, emb], "m/concatenate_1_2/concat", [1, t, 40], axis=-1)
    sh = dense(b, xe, [1, t], 40, 16, "slot_hidden", act=1, bias_after=True)
    slots = b.add("SOFTMAX", [dense(b, sh, [1, t], 16, 4, "slots", int8=False)],
                  "m/slots_1/Softmax", [1, t, 4], beta=1.0)
    b.output(intent, "intent")
    b.output(slots, "slots")
    return b.build()
