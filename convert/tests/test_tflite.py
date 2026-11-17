"""TFLite models built here, converted, and run on rayo's C library against the reference."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

import models
from helpers import decode, reference, session
from rayo_convert import ConvertError, Options, convert
from rayo_convert.naming import Namer
from tflite_build import Builder


def header(data: bytes) -> dict:
    n = struct.unpack("<I", data[8:12])[0]
    return json.loads(data[12 : 12 + n])


def ops_of(r) -> list[str]:
    return [n["op"] for n in r.graph["nodes"]]


@pytest.mark.parametrize("int8", [True, False])
def test_classifier(lib, int8):
    raw = models.classifier(int8)
    s, r = session(lib, raw, "classifier.tflite")
    assert r.mode == "fixed"
    assert ops_of(r) == ["dense", "dense", "softmax"]
    assert r.graph["nodes"][0]["activation"] == "relu"
    for _ in range(5):
        x = models.rnd(12)
        s.run([x])
        want = reference(raw, x)[0].reshape(-1)
        np.testing.assert_allclose(s.output(0), want, atol=2e-6)
    h = header(r.data)
    assert h["model"] == "classifier" and h["format_version"] == 1
    assert h["source"]["size_bytes"] == len(raw)


def test_reader_split_input_and_sequence(lib):
    t = 24
    raw = models.reader(t)
    s, r = session(lib, raw, split_inputs={"question": [("tokens", t), ("shapes", t)]},
                   outputs=["slots", "intent"])
    assert r.mode == "sequence"
    assert r.graph["symbols"] == {"T": t}
    assert [i["name"] for i in r.graph["inputs"]] == ["tokens", "shapes"]
    assert [o["name"] for o in r.graph["outputs"]] == ["slots", "intent"]
    convs = [n for n in r.graph["nodes"] if n["op"] == "conv1d"]
    assert [(c["size"], c["dilation"]) for c in convs] == [(3, 1), (3, 2)]
    assert all(c["input_quantization"] == "tensor_phases" for c in convs)
    rng = np.random.default_rng(1)
    for n in (t, 9):  # all positions, then a shorter question
        ids = np.zeros(2 * t, np.int32)
        ids[:n] = rng.integers(1, 50, n)
        ids[t : t + n] = rng.integers(1, 6, n)
        intent, slots = reference(raw, ids)
        s.bind("T", t)
        s.run([ids[:t], ids[t:]])
        np.testing.assert_allclose(s.output(1), intent.reshape(-1), atol=1e-4)
        np.testing.assert_allclose(s.output(0), slots.reshape(t, -1), atol=1e-4)
    # T is a symbol: the first n positions alone give the same as padding them to t,
    # for what reads them all (intent): masked pooling ignores the padding.
    s.bind("T", 9)
    s.run([ids[:9], ids[t : t + 9]])
    assert s.output(0).shape == (9, 4)


def test_split_input_refused_across_parts():
    raw = models.reader(8)
    with pytest.raises(ConvertError, match="--split-input question: the parts add up to 10"):
        convert(raw, "m.tflite", Options(split_inputs={"question": [("a", 5), ("b", 5)]}))
    with pytest.raises(ConvertError, match="reads across parts"):
        convert(raw, "m.tflite", Options(split_inputs={"question": [("a", 4), ("b", 12)]}))


@pytest.mark.parametrize("style", ["fused", "mha"])
def test_decoder_cache(lib, style):
    """Causal self-attention rewritten for a key/value cache: token by token gives the whole
    sequence's logits, position by position."""
    t, vocab = 16, 40
    raw = models.decoder(t, vocab=vocab, style=style, int8=(style == "fused"))
    s, r = session(lib, raw, "decoder.tflite")
    assert r.mode == "decode", r.notes
    assert r.graph["symbols"] == {"T": t, "P": t - 1} and r.graph["plan"] == {"P": 0}
    assert len(r.graph["states"]) == 4
    assert ops_of(r).count("cache_write") == 4 and ops_of(r).count("attention") == 2
    assert ops_of(r)[-3:] == ["last_row", "layer_norm", "dense"]
    assert all("BATCH_MATMUL" not in n for n in r.notes)
    rng = np.random.default_rng(3)
    tol = 2e-3 if style == "fused" else 2e-5
    for _ in range(3):
        n = int(rng.integers(6, t + 1))
        tokens = [int(v) for v in rng.integers(1, vocab, n)]
        padded = np.zeros(t, np.int32)
        padded[:n] = tokens
        want = reference(raw, padded)[0].reshape(t, vocab)
        got = decode(s, tokens, 3)
        for j, logits in enumerate(got):
            np.testing.assert_allclose(logits, want[2 + j], atol=tol, rtol=tol)


def test_decoder_all_positions(lib):
    t, vocab = 12, 30
    raw = models.decoder(t, vocab=vocab, int8=False)
    s, r = session(lib, raw, positions="all")
    assert "last_row" not in ops_of(r)
    tokens = [3, 5, 7, 11, 2]
    padded = np.zeros(t, np.int32)
    padded[:5] = tokens
    want = reference(raw, padded)[0].reshape(t, vocab)
    s.reset()
    s.bind("P", 0)
    s.bind("T", 5)
    s.run([np.asarray(tokens, np.int32)])
    np.testing.assert_allclose(s.output(0), want[:5], atol=2e-5)


def whole(s, tokens):
    s.bind("T", len(tokens))
    s.run([np.asarray(tokens, np.int32)])
    return s.output(0)


def test_decline_not_causal_runs_whole_sequence(lib):
    """A mask that hides nothing: attention, but not causal, so no cache; the whole sequence."""
    t, vocab = 10, 20
    raw = models.decoder(t, vocab=vocab, style="mha", int8=False, causal=False)
    s, r = session(lib, raw)
    assert r.mode == "sequence"
    assert ops_of(r).count("attention") == 2 and "cache_write" not in ops_of(r)
    assert all(n["causal"] == 0 for n in r.graph["nodes"] if n["op"] == "attention")
    tokens = [int(v) for v in np.random.default_rng(5).integers(1, vocab, t)]
    want = reference(raw, np.asarray(tokens, np.int32))[0].reshape(t, vocab)
    np.testing.assert_allclose(whole(s, tokens), want, atol=2e-5)


def test_decode_on_refused_for_not_causal():
    raw = models.decoder(8, vocab=12, style="mha", int8=False, causal=False)
    r = convert(raw, "m.tflite", Options(decode="on"))
    assert r.mode == "sequence"
    assert any("decoding with a key/value cache declined" in n and "not causal" in n
               for n in r.notes)


@pytest.mark.parametrize("why,kwargs,expect", [
    ("weak mask", {"mask_value": -1.0}, "neither 0 nor at most -10000"),
    ("no attention", {}, None),
])
def test_decline_one_to_one(lib, why, kwargs, expect):
    """A block that is not exactly rayo's attention is converted op by op (option b): every
    position recomputed at each step, the same numbers."""
    t, vocab = 8, 16
    raw = models.decoder(t, vocab=vocab, style="mha", int8=False, layers=1, **kwargs)
    s, r = session(lib, raw, attention=(why != "no attention"))
    assert "attention" not in ops_of(r) and "matmul" in ops_of(r)
    assert r.mode == "fixed"
    if expect:
        assert any(expect in n and "converted one by one" in n for n in r.notes), r.notes
    tokens = np.asarray(np.random.default_rng(2).integers(1, vocab, t), np.int32)
    want = reference(raw, tokens)[0].reshape(t, vocab)
    s.run([tokens])
    np.testing.assert_allclose(s.output(0).reshape(t, vocab), want, atol=2e-5)


def test_decline_varying_scale(lib):
    """A scale that is not one number: declined, and the fused qkv layout converted op by op
    (its rank-5 transpose sunk below the slices)."""
    t, vocab = 8, 12
    raw = models.decoder(t, vocab=vocab, style="fused", layers=1, int8=False,
                         mask=models.causal_mask(t), scale=np.arange(1, 9, dtype=np.float32))
    s, r = session(lib, raw)
    assert r.mode == "fixed"
    assert any("a constant that varies" in n for n in r.notes)
    assert "attention" not in ops_of(r) and "transpose" in ops_of(r)
    tokens = np.asarray(np.random.default_rng(4).integers(1, vocab, t), np.int32)
    s.run([tokens])
    want = reference(raw, tokens)[0].reshape(t, vocab)
    np.testing.assert_allclose(s.output(0).reshape(t, vocab), want, atol=2e-5, rtol=1e-5)


def test_fused_with_mask_and_scale_attribute(lib):
    t, vocab = 8, 16
    raw = models.decoder(t, vocab=vocab, style="fused", layers=1, int8=False,
                         mask=models.causal_mask(t, -1e30), scale=0.25)
    s, r = session(lib, raw)
    att = next(n for n in r.graph["nodes"] if n["op"] == "attention")
    assert att["scale"] == 0.25 and att["causal"] == 1
    tokens = [1, 4, 9, 3, 2, 7]
    padded = np.zeros(t, np.int32)
    padded[:6] = tokens
    want = reference(raw, padded)[0].reshape(t, vocab)
    for j, logits in enumerate(decode(s, tokens, 2)):
        np.testing.assert_allclose(logits, want[1 + j], atol=2e-5)


def test_unsupported_ops_named():
    b = Builder()
    x = b.input("image", [1, 4, 4, 3])
    y = b.add("AVERAGE_POOL_2D", [x], "m/pool_1/AvgPool", [1, 2, 2, 3], padding=1, stride_w=2,
              stride_h=2, filter_width=2, filter_height=2)
    z = b.add("TANH", [y], "m/act/Tanh", [1, 2, 2, 3])
    b.output(z, "y")
    with pytest.raises(ConvertError) as e:
        convert(b.build(), "pool.tflite")
    assert "op 0 (AVERAGE_POOL_2D" in str(e.value) and "no 2-D pooling" in str(e.value)


def test_symmetric_hybrid_refused():
    b = Builder()
    x = b.input("x", [1, 4])
    k = b.int8("m/d_1/MatMul", models.rnd(3, 4))
    y = b.add("FULLY_CONNECTED", [x, k, -1], "m/d_1/out", [1, 3],
              asymmetric_quantize_inputs=False)
    b.output(y, "y")
    with pytest.raises(ConvertError, match="symmetrically"):
        convert(b.build(), "m.tflite")


def test_layer_norm_fused(lib):
    b = Builder()
    x = b.input("x", [1, 5, 8])
    y = models.layer_norm(b, x, [1, 5, 8], "ln")
    b.output(y, "y")
    raw = b.build()
    s, r = session(lib, raw, sequence="none")
    assert ops_of(r) == ["layer_norm"]
    v = models.rnd(5, 8, scale=3)
    s.run([v])
    np.testing.assert_allclose(s.output(0), reference(raw, v)[0].reshape(5, 8), atol=1e-5)


def test_elementwise_and_reductions(lib):
    """Ops converted one to one, with broadcasting and constants as weights."""
    b = Builder()
    x = b.input("x", [1, 6, 4])
    c = b.const("m/c", models.rnd(4))
    y = b.add("MUL", [x, c], "m/mul", [1, 6, 4])
    y = b.add("SUB", [y, b.const("m/s", models.rnd(6, 1))], "m/sub", [1, 6, 4],
              fused_activation_function=1)
    m = b.add("MEAN", [y, b.const("m/ax", [1])], "m/mean", [1, 1, 4], keep_dims=True)
    y = b.add("DIV", [y, b.add("ADD", [m, b.const("m/one", np.float32(1))], "m/a", [1, 1, 4])],
              "m/div", [1, 6, 4])
    y = b.add("TRANSPOSE", [y, b.const("m/p", [0, 2, 1])], "m/t", [1, 4, 6])
    y = b.add("REDUCE_MAX", [y, b.const("m/ax2", [2])], "m/max", [1, 4])
    y = b.add("LOGISTIC", [y], "m/sig", [1, 4])
    b.output(y, "y")
    raw = b.build()
    s, r = session(lib, raw)
    assert {"mul", "sub", "relu", "reduce", "div", "transpose", "sigmoid"} <= set(ops_of(r))
    v = models.rnd(6, 4, scale=2)
    s.run([v])
    np.testing.assert_allclose(s.output(0), reference(raw, v)[0].reshape(-1), atol=1e-5)


def test_naming_keras3():
    n = Namer(["model_1/conv_1_1/convolution", "model_1/attention_0_1/qkv_1/Tensordot/MatMul",
               "arith.constant"])
    assert n.layer("model_1/conv_1_1/convolution") == "conv_1"
    assert n.layer("model_1/attention_0_1/qkv_1/Tensordot/MatMul") == "attention_0.qkv"
    assert n.layer("model_1/ln_0_a_1/Reshape;x") == "ln_0_a"
    assert n.layer("arith.constant") is None
    n2 = Namer(["sequential/dense/MatMul", "sequential/dense/BiasAdd"])
    assert n2.layer("sequential/dense/MatMul") == "dense"


def test_cli(tmp_path, capsys):
    from rayo_convert.cli import main

    src = tmp_path / "dec.tflite"
    src.write_bytes(models.decoder(8, vocab=12, layers=1, int8=False))
    out = tmp_path / "dec.weights"
    assert main([str(src), "-o", str(out), "--name", "tiny", "--rename", "logits=scores"]) == 0
    text = capsys.readouterr().out
    assert "decoder: 1 attention layers" in text and "decode" in text
    h = header(out.read_bytes())
    assert h["model"] == "tiny" and h["graph"]["outputs"][0]["name"] == "scores"
    bad = tmp_path / "bad.tflite"
    bad.write_bytes(b"not a model")
    assert main([str(bad)]) == 1
