"""int8 from float graphs (``--weights int8``): the converter quantises the weights, rayo's
kernels the activations. Checked three ways:

- against the reference (``rayo_convert.ref``) running the quantised source graph: the same
  int8 tensors; outputs within 1e-5, but where an activation lands on a rounding boundary of
  the on-the-fly quantisation (rayo sums in float32, the reference in float64, so the next
  int8 value can be taken in one of them: one step of that layer's input scale);
- against the TFLite path: the released Ask reader and writer, their int8 weights turned back
  into floats and quantised again here, give the very file the TFLite conversion gives, byte
  for byte (every int8 value, every scale, the graph), so rayo runs them identically;
- against ONNX's own float evaluator: the same labels, probabilities within int8's error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from rayo_convert import ConvertError, Options, convert, convert_file, onnx_frontend  # noqa: E402
from rayo_convert import passes, quantise, ref, tflite_frontend  # noqa: E402
from rayo_convert.cli import main  # noqa: E402
from rayo_convert.runtime import Model, Session  # noqa: E402
from test_onnx import G  # noqa: E402

ASK = Path(__file__).resolve().parent.parent.parent / "tests" / "ask"
RNG = np.random.default_rng(23)
ALL = dict(weights="int8", int8_min_size=0)


def rnd(*shape, scale=0.5):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def reference(raw: bytes, opts: Options, feeds: list[np.ndarray]) -> list[np.ndarray]:
    """The ONNX graph's outputs from the converter's reference, int8 as the converter
    quantises it."""
    g = onnx_frontend.read(raw)
    passes.fold(g)
    if opts.weights == "int8":
        quantise.apply(g, quantise.Int8(tuple(opts.int8_tensors), opts.int8_min_size))
    v = ref.run(g, {i: np.asarray(x).reshape(g.tensors[i].shape)
                    for i, x in zip(g.inputs, feeds, strict=True)})
    return [v[o] for o in g.outputs]


def float_onnx(raw: bytes, **feeds) -> list[np.ndarray]:
    return ReferenceEvaluator(onnx.load_from_string(raw)).run(None, feeds)


# The op families of the message reader and the Ask reader, as PyTorch exports them.


def reader_like(t: int = 16, vocab: int = 300, width: int = 32) -> bytes:
    """Embedding, channels-first Conv1d (one dilated, residual) with ReLU between, layer
    norm, masked mean pooling (PyTorch's clamp as Clip), dense layers, two heads."""
    g = G()
    mask = g.op("Cast", [g.op("Greater", ["tokens", g.c(np.asarray(0, np.int64))])],
                to=TensorProto.FLOAT)
    mask = g.op("Unsqueeze", [mask, g.c(np.asarray([-1], np.int64))])
    x = g.op("Gather", [g.c(rnd(vocab, 24), "token_embedding.weight"), "tokens"])
    g.nodes[-1].name = "/token_embedding/Gather"
    x = g.op("Transpose", [x], perm=[0, 2, 1])
    x = g.op("Conv", [x, g.c(rnd(width, 24, 3), "conv_1.weight"), g.c(rnd(width), "conv_1.bias")],
             pads=[1, 1])
    x = g.op("Relu", [x])
    r = g.op("Conv", [x, g.c(rnd(width, width, 3, scale=0.2), "conv_2.weight"),
                      g.c(rnd(width), "conv_2.bias")], pads=[4, 4], dilations=[4])
    x = g.op("Add", [x, g.op("Relu", [r])])  # a residual, channels first
    x = g.op("Transpose", [x], perm=[0, 2, 1])
    x = g.op("LayerNormalization", [x, g.c(1 + rnd(width, scale=0.1), "ln.weight"),
                                    g.c(rnd(width, scale=0.1), "ln.bias")], axis=-1)
    h = g.op("MatMul", [x, g.c(rnd(width, 48, scale=0.3), "onnx::MatMul_7")])
    g.nodes[-1].name = "/slot_hidden/MatMul"
    h = g.op("Relu", [g.op("Add", [g.c(rnd(48), "slot_hidden.bias"), h])])
    slots = g.op("MatMul", [h, g.c(rnd(48, 7, scale=0.3), "onnx::MatMul_9")])
    g.nodes[-1].name = "/slots/MatMul"
    g.op("Add", [g.c(rnd(7), "slots.bias"), slots])
    g.nodes[-1].output[0] = "slots"
    s = g.op("ReduceSum", [g.op("Mul", [h, mask]), g.c(np.asarray([1], np.int64))], keepdims=0)
    n = g.op("ReduceSum", [mask, g.c(np.asarray([1], np.int64))], keepdims=0)
    n = g.op("Clip", [n, g.c(np.asarray(1, np.float32))])
    pooled = g.op("Div", [s, n])
    y = g.op("Gemm", [pooled, g.c(rnd(5, 48, scale=0.3), "intent.weight"),
                      g.c(rnd(5), "intent.bias")], transB=1)
    g.op("Softmax", [y], axis=-1)
    g.nodes[-1].output[0] = "intent"
    return g.model([vi("tokens", [1, t], TensorProto.INT64)],
                   [vi("slots", [1, t, 7]), vi("intent", [1, 5])])


@pytest.fixture(scope="module")
def reader_raw():
    return reader_like()


def ids(t, n, vocab=300):
    x = np.zeros(t, np.int64)
    x[:n] = RNG.integers(1, vocab, n)
    return x


def test_reader_like_float_and_int8(lib, reader_raw):
    t = 16
    for weights in ("float32", "int8"):
        opts = Options(weights=weights, int8_min_size=0)
        r = convert(reader_raw, "reader.onnx", opts)
        ops = [n["op"] for n in r.graph["nodes"]]
        assert "transpose" not in ops  # channels first undone, ReLUs fused into the convolutions
        assert ops[:5] == ["compare", "reshape", "gather", "conv1d", "conv1d"]
        assert r.graph["nodes"][2]["output"] == "token_embedding.gather"
        convs = [n for n in r.graph["nodes"] if n["op"] == "conv1d"]
        assert [(c["dilation"], c.get("activation")) for c in convs] == [(1, "relu"), (4, "relu")]
        dense = [n for n in r.graph["nodes"] if n["op"] == "dense"]
        if weights == "int8":
            assert all(c["input_quantization"] == "tensor_phases" for c in convs)
            assert all(d["input_quantization"] == "rows" for d in dense)
        names = {w.name: w for w in r.weights}
        assert {"token_embedding.table", "conv_1.kernel", "conv_1.bias", "conv_2.kernel",
                "ln.gamma", "ln.beta", "slot_hidden.kernel", "slot_hidden.bias", "slots.kernel",
                "intent.kernel", "intent.bias"} <= set(names)
        for k in ("token_embedding.table", "conv_1.kernel", "conv_2.kernel",
                  "slot_hidden.kernel", "slots.kernel", "intent.kernel"):
            assert names[k].data.dtype == (np.int8 if weights == "int8" else np.float32), k
        s = Session(Model(lib, r.data))
        diffs, worst_float = [], 0.0
        for n in [t, 9, 1, *RNG.integers(1, t + 1, 17).tolist()]:
            x = ids(t, n)
            s.bind("T", t)
            s.run([x.astype(np.int32)])
            got = [s.output(0), s.output(1)]
            want = reference(reader_raw, opts, [x])
            flt = float_onnx(reader_raw, tokens=x[None])
            diffs += [np.abs(a - b.reshape(a.shape)).ravel() for a, b in zip(got, want,
                                                                              strict=True)]
            worst_float = max(worst_float, float(np.abs(got[1] - flt[1].reshape(-1)).max()))
            assert got[1].argmax() == flt[1].argmax()
        d = np.concatenate(diffs)
        if weights == "float32":
            assert d.max() <= 1e-5
            assert worst_float <= 1e-6
        else:
            # rayo sums in float32, the reference in float64: an activation on a rounding
            # boundary of the on-the-fly quantisation can take the next int8 value in one of
            # them (one step of that layer's input scale). Everything else agrees to 1e-5.
            assert np.mean(d <= 1e-5) >= 0.99, np.mean(d <= 1e-5)
            assert d.max() <= 5e-2
            assert worst_float <= 2e-2


def test_dilated_int8_quantises_each_phase(lib):
    """rayo's dilated conv1d quantises each phase alone; the reference does too, and a model
    whose phases differ in range shows it (one scale for the whole input would differ)."""
    g = G()
    g.op("Conv", ["x", g.c(rnd(8, 4, 3), "c.weight")], pads=[2, 2], dilations=[2])
    g.nodes[-1].output[0] = "y"
    raw = g.model([vi("x", [1, 4, 12])], [vi("y", [1, 8, 12])])
    opts = Options(**ALL)
    r = convert(raw, "conv.onnx", opts)
    s = Session(Model(lib, r.data))
    x = rnd(1, 4, 12)
    x[:, :, 1::2] *= 20  # the odd phase's range is 20 times the even's
    s.run([x])
    want = reference(raw, opts, [x])[0].reshape(8, 12)
    np.testing.assert_allclose(s.output(0), want, atol=1e-5)
    g2 = onnx_frontend.read(raw)
    passes.fold(g2)
    quantise.apply(g2, quantise.Int8(min_size=0))
    for o in g2.ops:
        o.options.pop("input_phases", None)
    v = ref.run(g2, {g2.inputs[0]: x})
    assert np.abs(v[g2.outputs[0]].reshape(8, 12) - want).max() > 1e-3


@pytest.mark.parametrize("name", ["ask-reader", "ask-writer"])
def test_same_file_as_tflite_hybrid(name):
    """A released dynamic-range model with its weights turned back into floats, quantised
    here: the same file as converting the TFLite model as it is, byte for byte."""
    if name == "ask-reader":
        kw = dict(name=name, split_inputs={"question": [("tokens", 96), ("shapes", 96)]},
                  outputs=["slots", "intent"])
    else:
        kw = dict(name=name, rename={"output_0": "logits"})
    raw = (ASK / f"{name}.tflite").read_bytes()
    hybrid = convert_file(ASK / f"{name}.tflite", Options(**kw))
    g = tflite_frontend.read(raw)
    n = 0
    for t in g.tensors:
        if t.const and t.dtype == "int8" and t.quant is not None:
            t.data, t.dtype, t.quant = t.real(), "float32", None
            n += 1
    for o in g.ops:
        if o.kind == "DEQUANTIZE":
            o.kind = "IDENTITY"  # of a float tensor now
    assert n >= 9
    again = convert(raw, f"{name}.tflite", Options(**kw, weights="int8"), graph=g)
    assert sum(w.data.dtype == np.int8 for w in again.weights) >= n  # qkv splits in three
    assert again.data == hybrid.data


def test_symmetric():
    w = np.asarray([[0.5, -1.0, 0.25], [0.0, 0.0, 0.0], [3.0, 1.0, -0.004]], np.float32)
    q, s = quantise.symmetric(w, 0)
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert q[0].tolist() == [64, -127, 32] and s[0] == np.float32(1.0) / np.float32(127.0)
    assert q[1].tolist() == [0, 0, 0] and s[1] == 1.0  # an all-zero output: scale 1
    assert q[2].tolist() == [127, 42, 0]
    qt, st = quantise.symmetric(w, None)
    assert st.shape == (1,) and np.abs(qt).max() == 127 and qt[0, 1] == -42


def test_classes_and_min_size(reader_raw):
    def dtypes(**kw):
        r = convert(reader_raw, "reader.onnx", Options(weights="int8", **kw))
        return {w.name: w.data.dtype.name for w in r.weights if w.data.ndim > 1}, r.notes

    only_dense, _ = dtypes(int8_tensors=("dense",), int8_min_size=0)
    assert only_dense["slot_hidden.kernel"] == "int8"
    assert only_dense["conv_1.kernel"] == only_dense["token_embedding.table"] == "float32"
    default, notes = dtypes()  # TFLite's threshold: fewer than 1024 values stay float32
    assert default["token_embedding.table"] == default["conv_1.kernel"] == "int8"  # 7200, 2304
    assert default["slots.kernel"] == default["intent.kernel"] == "float32"  # 336, 240
    assert any("fewer than 1024 values stay float32" in n and "slots (dense)" in n
               for n in notes)
    assert quantise.parse_classes("conv1d,dense") == ("dense", "conv1d")
    with pytest.raises(ConvertError, match="--int8-tensors: lstm"):
        quantise.parse_classes("dense,lstm")


def test_deterministic(reader_raw):
    a = convert(reader_raw, "reader.onnx", Options(**ALL)).data
    b = convert(reader_raw, "reader.onnx", Options(**ALL)).data
    assert a == b


def test_cli(tmp_path, capsys, reader_raw):
    src = tmp_path / "reader.onnx"
    src.write_bytes(reader_raw)
    out = tmp_path / "reader.weights"
    assert main([str(src), "-o", str(out), "--weights", "int8", "--int8-tensors", "conv1d"]) == 0
    assert "int8 weights: 2 conv1d" in capsys.readouterr().out
    assert out.read_bytes() == convert(reader_raw, "reader.onnx", Options(
        weights="int8", int8_tensors=("conv1d",))).data
    with pytest.raises(SystemExit, match="q4 is for language models"):
        main([str(src), "--weights", "q4"])
    assert main([str(src), "-o", str(out), "-q"]) == 0  # the default stays float32
    assert out.read_bytes() == convert(reader_raw, "reader.onnx").data


def test_shared_weight_stays_float():
    """A table that something other than a lookup also reads stays float32, noted."""
    g = G()
    table = g.c(rnd(40, 30), "emb.weight")
    g.op("Gather", [table, "ids"])
    g.nodes[-1].output[0] = "y"
    g.op("Mul", ["z", table])
    g.nodes[-1].output[0] = "w"
    raw = g.model([vi("ids", [1, 4], TensorProto.INT64), vi("z", [40, 30])],
                  [vi("y", [1, 4, 30]), vi("w", [40, 30])])
    g2 = onnx_frontend.read(raw)
    notes = quantise.apply(g2, quantise.Int8(min_size=0))
    assert g2.tensors[0].dtype == "float32"
    assert notes == ["int8: emb stays float32: it is read by more than an embedding lookup"]
