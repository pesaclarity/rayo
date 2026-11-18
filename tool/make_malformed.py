"""Writes tests/malformed: model files, tokeniser tables, tokenizer.json files and texts that
are each wrong in one way an attacker could choose, and tests/malformed/expected.txt, the
status rayo must give each (tests/test_security.c checks them; tool/ci/fuzz.sh replays them
through the fuzz targets). Python only, no NumPy; run from the repository's root.
"""

from __future__ import annotations

import copy
import json
import os
import struct
import sys

sys.path.insert(0, "convert")
from rayo_convert import tokenizer_table  # noqa: E402

OUT = "tests/malformed"
expected: list[tuple[str, str, str]] = []  # (path, stage, status)


def pcwt(header: dict | bytes, data: bytes, header_len: int | None = None,
         version: int = 1) -> bytes:
    h = header if isinstance(header, bytes) else json.dumps(header).encode()
    h += b" " * (-(12 + len(h)) % 4)
    n = len(h) if header_len is None else header_len
    return b"PCWT" + struct.pack("<II", version, n) + h + data


TABLE = struct.pack("<64f", *[i / 64 for i in range(64)])  # float32 [4, 16]


def base() -> dict:
    return {"format_version": 1, "model": "malformed",
            "tensors": [{"name": "table", "dtype": "float32", "shape": [4, 16], "offset": 0,
                         "bytes": 256}],
            "graph": {"version": 1, "symbols": {"T": 8},
                      "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]}],
                      "nodes": [{"op": "gather", "inputs": ["table", "tokens"],
                                 "output": "emb"}],
                      "outputs": [{"name": "emb", "node": "emb"}]}}


def model(name: str, stage: str, status: str, header: dict | bytes, data: bytes = TABLE,
          **kw) -> None:
    path = f"model/{name}.weights"
    os.makedirs(f"{OUT}/model", exist_ok=True)
    with open(f"{OUT}/{path}", "wb") as f:
        f.write(pcwt(header, data, **kw))
    expected.append((path, stage, status))


def edit(fn) -> dict:
    h = base()
    fn(h)
    return h


def main() -> None:
    model("valid", "run", "OK", base())
    model("header-past-end", "load", "FORMAT", base(), header_len=0x7FFFFFFC)
    model("header-unaligned", "load", "FORMAT", base(), header_len=401)
    model("version-2", "load", "UNSUPPORTED", base(), version=2)
    model("tensor-past-end", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(offset=4)))
    model("tensor-offset-overflow", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(offset=2147483644, bytes=8, shape=[2])))
    model("tensor-negative-offset", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(offset=-256)))
    model("tensor-size-mismatch", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(shape=[4, 17])))
    # 65536 x 65536 float32: 2^34 bytes, whose 32-bit product would wrap to 0.
    model("shape-overflow", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(shape=[65536, 65536, 1], bytes=0)), data=b"")
    model("shape-rank-5", "load", "FORMAT",
          edit(lambda h: h["tensors"][0].update(shape=[1, 1, 4, 16, 1])))
    model("tensor-twice", "load", "FORMAT",
          edit(lambda h: h["tensors"].append(dict(h["tensors"][0]))))
    model("int8-scales-outside", "load", "FORMAT", edit(lambda h: h["tensors"][0].update(
        dtype="int8", shape=[4, 64], quantization={"axis": None, "zero_point": 0,
                                                   "scales_offset": 1 << 30,
                                                   "scales_bytes": 4})))
    model("int4-group-16", "load", "UNSUPPORTED", edit(lambda h: h["tensors"][0].update(
        dtype="int4", shape=[4, 128], bytes=256, quantization={
            "group_size": 16, "zero_point": 8, "scales_offset": 0, "scales_bytes": 128})))
    model("node-forward-reference", "load", "FORMAT", edit(lambda h: h["graph"]["nodes"].insert(
        0, {"op": "add", "inputs": ["emb", "emb"], "output": "early"})))
    model("node-reads-itself", "load", "FORMAT", edit(lambda h: h["graph"]["nodes"].append(
        {"op": "add", "inputs": ["loop", "emb"], "output": "loop"})))
    model("output-not-a-node", "load", "FORMAT",
          edit(lambda h: h["graph"]["outputs"].__setitem__(0, {"name": "x", "node": "tokens"})))
    model("unknown-op", "load", "UNSUPPORTED",
          edit(lambda h: h["graph"]["nodes"][0].update(op="system")))
    model("too-few-inputs", "load", "FORMAT",
          edit(lambda h: h["graph"]["nodes"][0].update(inputs=["table"])))
    model("graph-version-9", "load", "UNSUPPORTED",
          edit(lambda h: h["graph"].update(version=9)))
    model("symbol-too-large", "load", "FORMAT",
          edit(lambda h: h["graph"]["symbols"].update(T=1 << 30)))
    model("too-many-symbols", "load", "LIMIT",
          edit(lambda h: h["graph"]["symbols"].update({f"S{i}": 1 for i in range(65)})))
    model("dimension-not-integer", "load", "FORMAT",
          edit(lambda h: h["graph"]["inputs"][0].update(shape=[2.5])))
    model("attribute-out-of-range", "load", "FORMAT", edit(lambda h: h["graph"]["nodes"].append(
        {"op": "attention", "inputs": ["emb", "emb", "emb"], "output": "p", "causal": 2})))
    model("bad-utf8-name", "load", "FORMAT",
          json.dumps(base()).replace('"malformed"', '"mal\\u00e9formed"').encode().replace(
              b"\\u00e9", b"\xc3"))
    model("nested-too-deep", "load", "FORMAT",
          json.dumps(base())[:-1].encode() + b', "deep": ' + b"[" * 40 + b"]" * 40 + b"}")
    model("not-json", "load", "FORMAT", b'{"format_version": 1, "tensors": [}')
    model("nul-in-string", "load", "FORMAT",
          json.dumps(base()).replace("malformed", "mal\\u0000formed").encode())
    model("tokenizer-outside", "load", "FORMAT",
          edit(lambda h: h.update(tokenizer={"offset": 256, "bytes": 4096,
                                             "format": "rayo-tokenizer-table", "version": 1})))
    # Three [2^24, 16] float32 activations alive at once: 3 GiB to plan, over
    # RAYO_MAX_ARENA_BYTES (and past 2^32, where a 32-bit plan's sum would wrap).
    model("arena-too-large", "session", "LIMIT", edit(lambda h: (
        h["graph"]["symbols"].update(T=1 << 24),
        h["graph"]["nodes"].extend([
            {"op": "gather", "inputs": ["table", "tokens"], "output": "b"},
            {"op": "add", "inputs": ["emb", "b"], "output": "c"}]),
        h["graph"]["outputs"].append({"name": "c", "node": "c"}))))

    # Tokeniser tables, from a valid one, each changed in one field.
    doc = json.load(open("tests/fuzz/tokenizers/sentencepiece.json"))
    table = bytearray(tokenizer_table.table(doc))
    os.makedirs(f"{OUT}/tokenizer_table", exist_ok=True)

    def word(t: bytearray, i: int) -> int:
        return struct.unpack_from("<I", t, 4 * i)[0]

    def tab(name: str, status: str, fn) -> None:
        t = bytearray(table)
        fn(t)
        path = f"tokenizer_table/{name}.table"
        open(f"{OUT}/{path}", "wb").write(bytes(t))
        expected.append((path, "load", status))

    def put(t: bytearray, i: int, v: int) -> None:
        struct.pack_into("<I", t, 4 * i, v & 0xFFFFFFFF)

    tab("valid", "OK", lambda t: None)
    tab("bad-magic", "FORMAT", lambda t: t.__setitem__(slice(0, 4), b"XTOK"))
    tab("version-2", "UNSUPPORTED", lambda t: put(t, 1, 2))
    tab("vocab-size-huge", "FORMAT", lambda t: put(t, 4, 0x7FFFFFFF))
    tab("vocab-offset-outside", "FORMAT", lambda t: put(t, 13, len(t)))
    tab("pool-outside", "FORMAT", lambda t: put(t, 12, 0xFFFFFFF0))
    tab("total-mismatch", "FORMAT", lambda t: put(t, 22, len(t) + 4))
    tab("steps-too-many", "FORMAT", lambda t: put(t, 10, 9))

    def full_vocab(t: bytearray) -> None:
        # Every slot used: a lookup of a missing string would probe for ever.
        at, slots = word(t, 13), word(t, 7)
        for i in range(slots):
            if struct.unpack_from("<i", t, at + 12 * i + 8)[0] < 0:
                struct.pack_into("<III", t, at + 12 * i, 0, 0, 0)
    tab("full-vocabulary", "FORMAT", full_vocab)

    def slot_outside(t: bytearray) -> None:
        at, slots = word(t, 13), word(t, 7)
        for i in range(slots):
            if struct.unpack_from("<i", t, at + 12 * i + 8)[0] >= 0:
                struct.pack_into("<I", t, at + 12 * i, 0x7FFFFFF0)
                return
    tab("vocab-string-outside", "FORMAT", slot_outside)

    def merge_id(t: bytearray) -> None:
        at, slots = word(t, 14), word(t, 8)
        for i in range(slots):
            if struct.unpack_from("<i", t, at + 16 * i)[0] >= 0:
                struct.pack_into("<i", t, at + 16 * i + 12, 1 << 20)
                return
    tab("merge-result-outside", "FORMAT", merge_id)

    def piece_outside(t: bytearray) -> None:
        struct.pack_into("<II", t, word(t, 15), 4, 0x7FFFFFFF)
    tab("piece-outside", "FORMAT", piece_outside)

    def byte_id(t: bytearray) -> None:
        struct.pack_into("<i", t, word(t, 20) + 4 * 65, 1 << 20)
    tab("byte-id-outside", "FORMAT", byte_id)

    # tokenizer.json files.
    os.makedirs(f"{OUT}/tokenizer_json", exist_ok=True)

    def tj(name: str, status: str, d: dict | bytes) -> None:
        path = f"tokenizer_json/{name}.json"
        data = d if isinstance(d, bytes) else json.dumps(d).encode()
        open(f"{OUT}/{path}", "wb").write(data)
        expected.append((path, "load", status))

    def tdoc(fn) -> dict:
        d = copy.deepcopy(doc)
        fn(d)
        return d

    split = {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": False}

    def pre(pattern: str):
        return lambda d: d.update(pre_tokenizer={"type": "Sequence", "pretokenizers": [
            dict(split, pattern={"Regex": pattern}),
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}]})

    tj("valid", "OK", doc)
    tj("id-huge", "LIMIT", tdoc(lambda d: d["model"]["vocab"].update({"x": 2147483000})))
    tj("id-negative", "FORMAT", tdoc(lambda d: d["model"]["vocab"].update({"x": -5})))
    tj("added-id-outside", "FORMAT", tdoc(lambda d: d["added_tokens"].append(
        {"id": -1, "content": "<x>"})))
    tj("unigram-score-huge", "FORMAT", {"model": {"type": "Unigram", "unk_id": 0, "vocab": [
        ["<unk>", 0.0], ["a", 1e300]]}})
    tj("groups-too-deep", "UNSUPPORTED", tdoc(pre("(" * 20 + "a" + ")" * 20)))
    tj("pattern-too-long", "LIMIT", tdoc(pre("a" * 5000)))
    tj("pattern-lazy", "UNSUPPORTED", tdoc(pre("a+?")))
    tj("not-utf8", "FORMAT", json.dumps(doc).encode().replace(b'"ab"', b'"\xff\xfe"'))
    tj("vocab-not-object", "FORMAT", tdoc(lambda d: d["model"].update(vocab=[1, 2])))
    tj("unknown-model", "UNSUPPORTED", tdoc(lambda d: d["model"].update(type="WordPiece")))
    # Encoded: patterns and normalisers that would take time or memory without bound.
    tj("redos-pattern", "OK", tdoc(lambda d: (pre("\\s*" * 12 + "x")(d), d.update(normalizer=None))))
    tj("normalizer-blowup", "OK", tdoc(lambda d: d.update(normalizer={
        "type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"String": " "}, "content": " " * 64}] * 4})))

    with open(f"{OUT}/expected.txt", "w") as f:
        f.write("# file stage status: what tests/test_security.c expects (tool/make_malformed.py)\n")
        for path, stage, status in expected:
            f.write(f"{path} {stage} {status}\n")


if __name__ == "__main__":
    main()
