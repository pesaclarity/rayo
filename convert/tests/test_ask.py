"""The released Ask reader and writer (tests/ask/*.tflite, as the Pesa Clarity app shipped
them) converted here and run on rayo: the reader's every label as TFLite's and probabilities
within 1e-4; the writer decoding with its key/value cache, TFLite's next token at every step,
and its logits bit for bit those of the hand-written graph rayo's C tests run
(tests/ask/ask-writer.weights)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helpers import decode
from rayo_convert import Options, convert_file
from rayo_convert.runtime import Model, Session

ASK = Path(__file__).resolve().parent.parent.parent / "tests" / "ask"

READER = Options(name="ask-reader", split_inputs={"question": [("tokens", 96), ("shapes", 96)]},
                 outputs=["slots", "intent"])
WRITER = Options(name="ask-writer", rename={"output_0": "logits"})


@pytest.fixture(scope="module")
def reader():
    return convert_file(ASK / "ask-reader.tflite", READER)


@pytest.fixture(scope="module")
def writer():
    return convert_file(ASK / "ask-writer.tflite", WRITER)


def test_reader(lib, reader):
    g = reader.graph
    assert reader.mode == "sequence" and g["symbols"] == {"T": 96}
    assert [i["name"] for i in g["inputs"]] == ["tokens", "shapes"]
    assert [o["name"] for o in g["outputs"]] == ["slots", "intent"]
    s = Session(Model(lib, reader.data))
    s.bind("T", 96)
    readings = json.loads((ASK / "ask_reader_decodes.json").read_text())["readings"]
    same, worst = 0, 0.0
    for r in readings:
        x = np.asarray(r["input"], np.int32)
        s.run([x[:96], x[96:]])
        slots, intent = s.output(0), s.output(1)
        wi, ws = np.asarray(r["intent"]), np.asarray(r["slots"])
        worst = max(worst, float(np.abs(intent - wi).max()),
                    float(np.abs(slots[: len(ws)] - ws).max()))
        same += int(intent.argmax() == wi.argmax()
                    and (slots[: len(ws)].argmax(1) == ws.argmax(1)).all())
    assert same == len(readings) == 30
    assert worst <= 1e-4


def released_tensors(name: str) -> tuple[dict, bytes]:
    blob = (ASK / name).read_bytes()
    n = int.from_bytes(blob[8:12], "little")
    return {t["name"]: t for t in json.loads(blob[12 : 12 + n])["tensors"]}, blob[12 + n :]


def same_tensors(weights: list, name: str) -> int:
    """How many tensors of the released file ``name`` ``weights`` holds byte for byte."""
    want, data = released_tensors(name)
    got = {w.name: w for w in weights}
    for t in want.values():
        w = got[t["name"]]
        assert list(w.data.shape) == t["shape"], t["name"]
        assert np.ascontiguousarray(w.data).tobytes() == data[t["offset"] :
                                                              t["offset"] + t["bytes"]]
        q = t.get("quantization")
        if q:
            s = data[q["scales_offset"] : q["scales_offset"] + q["scales_bytes"]]
            assert w.scales.astype("<f4").tobytes() == s and w.axis == q["axis"], t["name"]
    return len(want)


def test_writer_tensors_are_the_released_files(writer):
    assert same_tensors(writer.weights, "ask-writer.weights") == len(writer.weights) == 54


def test_reader_tensors_are_the_released_files(reader):
    """The same tensors, under the same names, as pesaclarity-ml read out of the TFLite file
    (and three scalars of the masked pooling, which the hand-written graph did in its op)."""
    released = (ASK / "ask-reader.weights").read_bytes()
    n = int.from_bytes(released[8:12], "little")
    want = {t["name"]: t for t in json.loads(released[12 : 12 + n])["tensors"]}
    data = released[12 + n :]
    got = {w.name: w for w in reader.weights}
    assert set(want) <= set(got) and len(got) - len(want) == 3
    for name, t in want.items():
        w = got[name]
        raw = data[t["offset"] : t["offset"] + t["bytes"]]
        assert list(w.data.shape) == t["shape"]
        assert np.ascontiguousarray(w.data).tobytes() == raw, name
        q = t.get("quantization")
        if q:
            scales = data[q["scales_offset"] : q["scales_offset"] + q["scales_bytes"]]
            assert w.scales.astype("<f4").tobytes() == scales and w.axis == q["axis"], name


def test_writer_decodes_with_cache(lib, writer):
    g = writer.graph
    assert writer.mode == "decode"
    assert g["symbols"] == {"T": 64, "P": 63} and len(g["states"]) == 6
    ops = [n["op"] for n in g["nodes"]]
    assert ops.count("cache_write") == 6 and ops.count("attention") == 3
    assert ops[-3:] == ["last_row", "layer_norm", "dense"]
    s = Session(Model(lib, writer.data))
    hand = Session(Model(lib, (ASK / "ask-writer.weights").read_bytes()))
    seqs = json.loads((ASK / "ask_writer_tflite.json").read_text())["sequences"]
    steps = same = identical = 0
    worst = 0.0
    for q in seqs:
        toks, n0 = q["tokens"][:64], q["prefix"]
        got = decode(s, toks, n0)
        ref = decode(hand, toks, n0)
        for j, (want, top) in enumerate(zip(q["argmax"], q["top"], strict=False)):
            logits = got[j]
            steps += 1
            same += int(logits.argmax() == want)
            worst = max(worst, abs(float(logits[want]) - top))
            identical += int(np.array_equal(logits.view(np.uint32), ref[j].view(np.uint32)))
    assert steps == 2243 and same == steps
    assert worst < 0.2
    assert identical == steps
