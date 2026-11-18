"""rayo-convert: a TFLite (or ONNX) model to a rayo model.

    rayo-convert model.tflite -o model.weights
    rayo-convert reader.tflite -o reader.weights --name ask-reader \\
        --split-input question=tokens:96,shapes:96 --outputs slots,intent
    rayo-convert writer.tflite -o writer.weights --rename output_0=logits

It prints what it recognised and decided (attention rewritten for a key/value cache, sizes
made symbols, ops it could not convert) and exits 1, naming every op it cannot convert, when
the model does not fit rayo. docs/CONVERTING.md describes each step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .convert import Options, convert_file
from .ir import ConvertError


def _pairs(text: str, what: str) -> dict[str, str]:
    out = {}
    for item in text.split(","):
        if "=" not in item:
            raise SystemExit(f"rayo-convert: {what}: {item!r} is not NAME=VALUE")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _split(values: list[str]) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"rayo-convert: --split-input {v!r}: NAME=PART:SIZE,PART:SIZE")
        name, parts = v.split("=", 1)
        out[name] = []
        for p in parts.split(","):
            pname, _, size = p.partition(":")
            if not size.isdigit():
                raise SystemExit(f"rayo-convert: --split-input {v!r}: PART:SIZE")
            out[name].append((pname, int(size)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rayo-convert", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .tflite (or .onnx) file")
    ap.add_argument("-o", "--output", help="the rayo model to write (default: MODEL.weights)")
    ap.add_argument("--name", help="the model's name in the file (default: the file's stem)")
    ap.add_argument("--split-input", action="append", default=[], metavar="NAME=PART:SIZE,...",
                    help="present input NAME as several, cut along its last axis")
    ap.add_argument("--outputs", help="the outputs to keep, in this order (source names)")
    ap.add_argument("--rename", help="OLD=NEW,...: rename inputs or outputs")
    ap.add_argument("--sequence", default="auto",
                    help="inputs whose length becomes the size symbol T: auto (int32 ids "
                         "[1, T]), none, or names")
    ap.add_argument("--decode", choices=["auto", "on", "off"], default="auto",
                    help="rewrite causal self-attention for token-by-token decoding with a "
                         "key/value cache (auto: when the model has it)")
    ap.add_argument("--positions", choices=["auto", "last", "all"], default="auto",
                    help="a decoder's outputs: the last new position (default) or every one")
    ap.add_argument("--no-attention", action="store_true",
                    help="do not recognise attention; convert its ops one by one")
    ap.add_argument("--input-shape", action="append", default=[], metavar="NAME=D,D,...",
                    help="the shape of an input whose sizes the file leaves open")
    ap.add_argument("--metadata", help="a JSON object for the file's metadata")
    ap.add_argument("--graph", action="store_true", help="print the graph written")
    ap.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")
    ap.add_argument("--version", action="version", version=f"rayo-convert {__version__}")
    a = ap.parse_args(argv)
    shapes = {}
    for s in a.input_shape:
        name, _, dims = s.partition("=")
        shapes[name] = tuple(int(d) for d in dims.split(","))
    opts = Options(
        name=a.name,
        split_inputs=_split(a.split_input),
        outputs=[s.strip() for s in a.outputs.split(",")] if a.outputs else None,
        rename=_pairs(a.rename, "--rename") if a.rename else {},
        sequence=a.sequence,
        decode=a.decode,
        positions=a.positions,
        attention=not a.no_attention,
        input_shapes=shapes,
        metadata=json.loads(a.metadata) if a.metadata else None,
    )
    try:
        result = convert_file(a.model, opts)
    except ConvertError as e:
        print(f"rayo-convert: {a.model}: cannot convert:", file=sys.stderr)
        for line in str(e).splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"rayo-convert: {e}", file=sys.stderr)
        return 2
    out = Path(a.output) if a.output else Path(a.model).with_suffix(".weights")
    out.write_bytes(result.data)
    if not a.quiet:
        for line in result.notes:
            print(f"note: {line}")
        g = result.graph
        print(f"wrote {out} ({len(result.data)} bytes): {len(g['nodes'])} nodes, "
              f"{len(result.weights)} tensors, {result.mode}"
              + (f", symbols {g['symbols']}" if "symbols" in g else ""))
        if a.graph:
            print(json.dumps(g, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
