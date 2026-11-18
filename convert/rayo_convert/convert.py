"""The conversion: a TFLite or ONNX model to a rayo model (PCWT version 1 with a graph)."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, attention, passes, quantise
from .ir import ConvertError, Graph
from .lower import Lowered, Lowerer, SequenceError, Weight
from .naming import Namer
from .passes import Notes, rshape
from .pcwt import write


@dataclass
class Options:
    name: str | None = None  # the model's name in the file; the file's stem by default
    split_inputs: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    outputs: list[str] | None = None  # which outputs, in this order (source names)
    rename: dict[str, str] = field(default_factory=dict)  # input and output names
    sequence: str = "auto"  # auto, none, or input names joined by commas
    decode: str = "auto"  # auto, on, off
    positions: str = "auto"  # what a decoder gives: last (the default) or all positions
    attention: bool = True  # recognise attention blocks
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] | None = None
    # float32: the weights as the source holds them (a TFLite dynamic-range model's int8 too);
    # int8: the float32 weights quantised by the converter (quantise.py)
    weights: str = "float32"
    int8_tensors: tuple[str, ...] = quantise.CLASSES  # with int8: which classes
    int8_min_size: int = quantise.MIN_SIZE  # with int8: smaller weights stay float32


@dataclass
class Result:
    data: bytes
    graph: dict[str, Any]
    weights: list[Weight]
    notes: list[str]
    mode: str  # fixed, sequence or decode


def read_model(raw: bytes, path: str) -> Graph:
    if path.lower().endswith(".onnx") or (raw[:1] == b"\x08" and raw[4:8] != b"TFL3"):
        from . import onnx_frontend

        return onnx_frontend.read(raw)
    from . import tflite_frontend

    return tflite_frontend.read(raw)


def convert_file(path: str | Path, options: Options | None = None) -> Result:
    p = Path(path)
    raw = p.read_bytes()
    return convert(raw, p.name, options)


def convert(raw: bytes, file_name: str, options: Options | None = None,
            graph: Graph | None = None) -> Result:
    """``raw`` (the file ``file_name``) converted. ``graph``: the file already read, when the
    caller has changed it (the tests turn a dynamic-range model back into floats)."""
    opts = options or Options()
    if opts.weights not in ("float32", "int8"):
        raise ConvertError(f"--weights {opts.weights}: a TFLite or ONNX model takes float32 "
                           "or int8 (q4 is for language models)")
    g = graph if graph is not None else read_model(raw, file_name)
    notes = Notes()
    names = {i: g.input_names.get(i) or g.tensors[i].name for i in g.inputs}
    out_names = {i: g.output_names.get(i) or g.tensors[i].name for i in g.outputs}
    passes.check_ops(g)
    passes.fold(g)
    if opts.weights == "int8":
        for line in quantise.apply(g, quantise.Int8(tuple(opts.int8_tensors),
                                                    opts.int8_min_size)):
            notes.add(line)
    overrides = {}
    for i in g.inputs:
        if names[i] in opts.input_shapes:
            overrides[i] = tuple(opts.input_shapes[names[i]])
            g.tensors[i].shape = overrides[i]
    sh = passes.shapes(g, overrides)
    if opts.split_inputs:
        passes.split_inputs(g, sh, opts.split_inputs, names)
    passes.simplify(g, sh, notes)
    passes.fuse_conv1d(g, sh, notes)
    passes.fuse_bias_activation(g)
    passes.fuse_layer_norm(g, sh)
    passes.fuse_bias_activation(g)
    passes.simplify(g, sh, notes)
    namer0 = Namer([t.name for t in g.tensors])
    blocks = []
    if opts.attention:
        found, declined = attention.find(g, sh)
        for line in declined:
            notes.add(line + "; its ops are converted one by one")
        for blk in found:
            try:
                attention.apply(g, sh, blk, namer0.layer)
            except attention.Declined as e:
                notes.add(f"{g.describe(blk.softmax)}: attention not rewritten: {e}; its ops are "
                          "converted one by one")
                continue
            blocks.append(blk)
            scale = np.float32(blk.scale)
            default = np.float32(1.0) / np.float32(np.sqrt(np.float32(blk.head_size)))
            notes.add(f"{g.describe(blk.softmax)}: attention, {blk.heads} heads of "
                      f"{blk.head_size}, {'causal' if blk.causal else 'not causal'}, scale "
                      f"{float(scale):g}{' (1/sqrt(head size))' if scale == default else ''}")
        passes._dead(g)
    passes.sink_slices(g, sh)
    passes.simplify(g, sh, notes)
    # Outputs: which, in what order, named how.
    if opts.outputs:
        by_name = {v: k for k, v in out_names.items()}
        missing = [n for n in opts.outputs if n not in by_name]
        if missing:
            raise ConvertError(f"--outputs: no output {', '.join(missing)} (outputs: "
                               f"{', '.join(out_names.values())})")
        g.outputs = [by_name[n] for n in opts.outputs]
        passes._dead(g)
    for d in (names, out_names):
        for k, v in list(d.items()):
            d[k] = opts.rename.get(v, v)
    # The sequence inputs.
    seq: list[int] = []
    length = 0
    if opts.sequence == "auto":
        cand = [i for i in g.inputs if g.tensors[i].dtype in ("int32", "int64")
                and len(rshape(sh[i])) == 1 and rshape(sh[i])[0] > 1]
        lens = {rshape(sh[i])[0] for i in cand}
        if cand and len(lens) == 1:
            seq, length = cand, lens.pop()
    elif opts.sequence != "none":
        wanted = [s.strip() for s in opts.sequence.split(",") if s.strip()]
        by = {names[i]: i for i in g.inputs}
        for w in wanted:
            if w not in by:
                raise ConvertError(f"--sequence: no input {w}")
            if len(rshape(sh[by[w]])) != 1:
                raise ConvertError(f"--sequence {w}: rayo's sequence inputs are [T]; this one "
                                   f"is {rshape(sh[by[w]])}")
        seq = [by[w] for w in wanted]
        lens = {rshape(sh[i])[0] for i in seq}
        if len(lens) != 1:
            raise ConvertError("--sequence: the inputs differ in length")
        length = lens.pop()
    causal = any(b.causal for b in blocks)
    modes = ["fixed"]
    if seq:
        modes = ["sequence", "fixed"]
        if opts.decode == "on" or (opts.decode == "auto" and causal):
            modes = ["decode", "sequence", "fixed"]
    elif opts.decode == "on":
        raise ConvertError("--decode on: no sequence input (int32 ids [1, T]); name it with "
                           "--sequence")
    positions = opts.positions if opts.positions != "auto" else "last"
    lowered: Lowered | None = None
    mode = modes[0]
    for k, mode in enumerate(modes):
        try:
            lowered = Lowerer(g, sh, mode, seq, length, names, out_names, positions,
                              Namer([t.name for t in g.tensors]), notes).run()
            break
        except SequenceError as e:
            nxt = modes[k + 1] if k + 1 < len(modes) else None
            if mode == "decode":
                notes.add(f"decoding with a key/value cache declined: {e}. The graph runs the "
                          "whole sequence instead: bind T to the tokens so far and run them all "
                          "at every step")
            elif mode == "sequence":
                notes.add(f"sizes fixed at the source's: {e}")
            if nxt is None:
                raise ConvertError("\n".join([str(e), *("note: " + n for n in notes.lines)])
                                   ) from e
        except ConvertError as e:
            raise ConvertError("\n".join([str(e), *("note: " + n for n in notes.lines)])) from e
    assert lowered is not None
    if mode == "decode":
        n = len(lowered.graph.get("states", [])) // 2
        notes.add(f"decoder: {n} attention layers keep their keys and values in states; run the "
                  f"prompt with P = 0 and T = its length n, then each new token with T = 1 at "
                  f"P = n, n + 1, ... (at most {length} positions in all)")
    elif mode == "sequence":
        notes.add(f"sequence: T, at most {length}, the length of "
                  f"{', '.join(names[i] for i in seq)}")
    header: dict[str, Any] = {
        "format_version": 1,
        "model": opts.name or Path(file_name).stem,
        "source": {"file": Path(file_name).name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "size_bytes": len(raw)},
        "layers": [],
        "graph": lowered.graph,
        "converter": {"name": "rayo-convert", "version": __version__},
    }
    if opts.metadata is not None:
        header["metadata"] = opts.metadata
    data = write(header, lowered.weights)
    return Result(data, lowered.graph, lowered.weights, notes.lines, mode)


def copy_graph(g: Graph) -> Graph:
    return copy.deepcopy(g)
