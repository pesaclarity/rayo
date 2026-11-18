"""ONNX models built here (float32), converted, run on rayo and checked against ONNX's own
reference evaluator."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from helpers import decode  # noqa: E402
from rayo_convert import ConvertError, Options, convert  # noqa: E402
from rayo_convert.runtime import Model, Session  # noqa: E402

RNG = np.random.default_rng(11)


def rnd(*shape, scale=0.5):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


class G:
    """A small ONNX graph builder."""

    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def c(self, value, name=None):
        name = name or f"c{len(self.inits)}"
        v = np.asarray(value)
        if v.dtype == np.float64:
            v = v.astype(np.float32)
        self.inits.append(numpy_helper.from_array(v, name))
        return name

    def op(self, kind, inputs, n_out=1, name=None, **attrs):
        self.n += 1
        outs = [f"{name or kind.lower()}_{self.n}_{i}" for i in range(n_out)]
        self.nodes.append(helper.make_node(kind, inputs, outs, name=f"{kind}_{self.n}", **attrs))
        return outs[0] if n_out == 1 else outs

    def model(self, inputs, outputs):
        g = helper.make_graph(self.nodes, "g", inputs, outputs, self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        return m.SerializeToString()


def run_onnx(raw, **feeds):
    return ReferenceEvaluator(onnx.load_from_string(raw)).run(None, feeds)


def test_mlp(lib):
    g = G()
    h = g.op("Gemm", ["x", g.c(rnd(16, 8)), g.c(rnd(16))], transB=1)
    h = g.op("Relu", [h])
    y = g.op("MatMul", [h, g.c(rnd(16, 5))])
    y = g.op("Add", [y, g.c(rnd(5))])
    p = g.op("Softmax", [y], axis=-1)
    g.nodes[-1].output[0] = "probs"
    raw = g.model([helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
                  [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 5])])
    r = convert(raw, "mlp.onnx")
    assert [n["op"] for n in r.graph["nodes"]] == ["dense", "dense", "softmax"]
    s = Session(Model(lib, r.data))
    for _ in range(3):
        x = rnd(1, 8)
        s.run([x])
        np.testing.assert_allclose(s.output(0), run_onnx(raw, x=x)[0].reshape(-1), atol=1e-6)
    del p


def test_conv1d_channels_first(lib):
    g = G()
    y = g.op("Conv", ["x", g.c(rnd(6, 4, 3)), g.c(rnd(6))], pads=[1, 1])
    y = g.op("Relu", [y])
    y = g.op("Conv", [y, g.c(rnd(5, 6, 3))], pads=[2, 2], dilations=[2])
    g.nodes[-1].output[0] = "y"
    raw = g.model([helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 10])],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5, 10])])
    r = convert(raw, "conv.onnx")
    convs = [n for n in r.graph["nodes"] if n["op"] == "conv1d"]
    assert [(c["size"], c["dilation"]) for c in convs] == [(3, 1), (3, 2)]
    s = Session(Model(lib, r.data))
    x = rnd(1, 4, 10)
    s.run([x])
    np.testing.assert_allclose(s.output(0), run_onnx(raw, x=x)[0].reshape(5, 10), atol=1e-5)


def test_conv_valid_padding_refused():
    g = G()
    g.op("Conv", ["x", g.c(rnd(6, 4, 3))])
    g.nodes[-1].output[0] = "y"
    raw = g.model([helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 10])],
                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 8])])
    with pytest.raises(ConvertError, match=r"node 0 \(Conv\): pads \[0, 0\]"):
        convert(raw, "conv.onnx")


def decoder(t=12, d=16, heads=2, vocab=24, layers=2, where=True, causal=True):
    """A GPT-like decoder as PyTorch exports one: an embedding Gather, positions added,
    LayerNormalization, one qkv MatMul split in three, Reshape and Transpose into heads,
    MatMul with the keys transposed, Div by sqrt(head size), the mask by Where, Softmax,
    MatMul, Transpose and Reshape back, GELU through Erf, and the logits."""
    hd = d // heads
    g = G()
    x = g.op("Gather", [g.c(rnd(vocab, d)), "ids"], axis=0)
    x = g.op("Add", [x, g.c(rnd(1, t, d))])
    allowed = np.tril(np.ones((t, t), bool)) if causal else np.ones((t, t), bool)
    for _ in range(layers):
        a = g.op("LayerNormalization", [x, g.c(1 + rnd(d, scale=0.1)), g.c(rnd(d, scale=0.1))],
                 axis=-1, epsilon=1e-5)
        qkv = g.op("Add", [g.op("MatMul", [a, g.c(rnd(d, 3 * d))]), g.c(rnd(3 * d))])
        q, k, v = g.op("Split", [qkv, g.c(np.asarray([d, d, d], np.int64))], n_out=3, axis=2)
        hs = []
        for z, perm in ((q, [0, 2, 1, 3]), (k, [0, 2, 3, 1]), (v, [0, 2, 1, 3])):
            z = g.op("Reshape", [z, g.c(np.asarray([1, t, heads, hd], np.int64))])
            hs.append(g.op("Transpose", [z], perm=perm))
        s = g.op("MatMul", [hs[0], hs[1]])
        s = g.op("Div", [s, g.c(np.float32(np.sqrt(hd)))])
        if where:
            s = g.op("Where", [g.c(allowed.reshape(1, 1, t, t)), s, g.c(np.float32(-np.inf))])
        else:
            s = g.op("Add", [s, g.c(np.where(allowed, 0, -1e9).astype(np.float32))])
        w = g.op("Softmax", [s], axis=-1)
        o = g.op("MatMul", [w, hs[2]])
        o = g.op("Transpose", [o], perm=[0, 2, 1, 3])
        o = g.op("Reshape", [o, g.c(np.asarray([1, t, d], np.int64))])
        o = g.op("Add", [g.op("MatMul", [o, g.c(rnd(d, d))]), g.c(rnd(d))])
        x = g.op("Add", [x, o])
        f = g.op("LayerNormalization", [x, g.c(1 + rnd(d, scale=0.1)), g.c(rnd(d, scale=0.1))],
                 axis=-1, epsilon=1e-5)
        f = g.op("Add", [g.op("MatMul", [f, g.c(rnd(d, 2 * d))]), g.c(rnd(2 * d))])
        e = g.op("Erf", [g.op("Div", [f, g.c(np.float32(np.sqrt(2.0)))])])
        e = g.op("Add", [e, g.c(np.float32(1.0))])
        f = g.op("Mul", [g.op("Mul", [f, e]), g.c(np.float32(0.5))])
        f = g.op("Add", [g.op("MatMul", [f, g.c(rnd(2 * d, d))]), g.c(rnd(d))])
        x = g.op("Add", [x, f])
    x = g.op("LayerNormalization", [x, g.c(np.ones(d, np.float32)), g.c(np.zeros(d, np.float32))],
             axis=-1)
    g.op("MatMul", [x, g.c(rnd(d, vocab))])
    g.nodes[-1].output[0] = "logits"
    return g.model([helper.make_tensor_value_info("ids", TensorProto.INT64, [1, t])],
                   [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, t, vocab])])


@pytest.mark.parametrize("where", [True, False])
def test_decoder_cache(lib, where):
    t, vocab = 12, 24
    raw = decoder(t, vocab=vocab, where=where)
    r = convert(raw, "gpt.onnx")
    assert r.mode == "decode", r.notes
    ops = [n["op"] for n in r.graph["nodes"]]
    assert ops.count("attention") == 2 and ops.count("cache_write") == 4
    assert "gelu_erf" in [n.get("activation") for n in r.graph["nodes"]]
    assert any("int64 in the source" in n for n in r.notes)
    s = Session(Model(lib, r.data))
    tokens = [int(v) for v in RNG.integers(0, vocab, t)]
    want = run_onnx(raw, ids=np.asarray([tokens], np.int64))[0].reshape(t, vocab)
    for j, logits in enumerate(decode(s, tokens, 4)):
        np.testing.assert_allclose(logits, want[3 + j], atol=1e-4)


def test_decoder_not_causal(lib):
    t, vocab = 8, 16
    raw = decoder(t, vocab=vocab, causal=False, layers=1)
    r = convert(raw, "enc.onnx", Options(decode="on"))
    assert r.mode == "sequence"
    s = Session(Model(lib, r.data))
    tokens = RNG.integers(0, vocab, t)
    s.bind("T", t)
    s.run([tokens.astype(np.int32)])
    want = run_onnx(raw, ids=tokens.reshape(1, t).astype(np.int64))[0].reshape(t, vocab)
    np.testing.assert_allclose(s.output(0), want, atol=1e-4)


def test_unsupported_onnx_op():
    g = G()
    g.op("LSTM", ["x", g.c(rnd(1, 8, 4)), g.c(rnd(1, 8, 2))], hidden_size=2)
    raw = g.model([helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 4])],
                  [helper.make_tensor_value_info("lstm_1_0", TensorProto.FLOAT, None)])
    with pytest.raises(ConvertError, match=r"node 0 \(LSTM.*the ONNX op LSTM is not supported"):
        convert(raw, "lstm.onnx")
