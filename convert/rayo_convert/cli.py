"""rayo-convert: a TFLite (or ONNX) model, or a Hugging Face decoder-only language model, to a
rayo model.

    rayo-convert model.tflite -o model.weights
    rayo-convert SmolLM2-135M-Instruct/ -o smollm2.weights --weights q4 --context 2048
    rayo-convert reader.tflite -o reader.weights --name ask-reader \\
        --split-input question=tokens:96,shapes:96 --outputs slots,intent
    rayo-convert writer.tflite -o writer.weights --rename output_0=logits
    rayo-convert reader.onnx -o reader.weights --weights int8

It prints what it recognised and decided (attention rewritten for a key/value cache, sizes
made symbols, ops it could not convert) and exits 1, naming every op it cannot convert, when
the model does not fit rayo. docs/CONVERTING.md describes each step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, quantise
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
    ap.add_argument("model", help="a .tflite (or .onnx) file, or a Hugging Face model's "
                    "directory (config.json, *.safetensors, tokenizer.json)")
    ap.add_argument("--weights", choices=["q4", "int8", "float32"], default="q4",
                    help="a language model's weights: 4-bit groups (default), int8 or float32")
    ap.add_argument("--scales", choices=["float16", "float32"], default="float16",
                    help="q4 (and q3) group scales: float16 (default, 4.5 bits a weight) or "
                         "float32 (5 bits, as files before)")
    ap.add_argument("--q3", metavar="ROLES",
                    help="with --weights q4, 3-bit groups for these matrices, by their role in "
                         "the graph: attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_up, mlp_down, "
                         "embed, or the groups attn, mlp, mlp_in (gate and up), comma-separated")
    ap.add_argument("--no-tokenizer-table", action="store_true",
                    help="do not write the tokeniser's table into the model file (it is read "
                         "in place from the mapped file, with no tokenizer.json at run time)")
    ap.add_argument("--context", type=int,
                    help="a language model's longest context, in tokens (default 2048, at most "
                         "its own); a session plans its key/value cache for its own context "
                         "(the chat program's default: 1024)")
    ap.add_argument("--chunk", type=int, default=64,
                    help="a language model's prompt tokens per run (default 64): more is a "
                         "little faster on long prompts and plans more memory")
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
    ap.add_argument("--int8-tensors", default="all", metavar="CLASSES",
                    help="with --weights int8 for a TFLite or ONNX model: the weights quantised, "
                         "by class: dense, conv1d, embedding, comma-separated (default all)")
    ap.add_argument("--int8-min-size", type=int, default=1024, metavar="N",
                    help="with --weights int8: weights with fewer than N values stay float32 "
                         "(default 1024, as TFLite's converter)")
    ap.add_argument("--graph", action="store_true", help="print the graph written")
    ap.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")
    ap.add_argument("--version", action="version", version=f"rayo-convert {__version__}")
    a = ap.parse_args(argv)
    if Path(a.model).is_dir():
        return _language_model(a)
    # --weights for a TFLite or ONNX model: float32 (as the source holds them, the default) or
    # int8 (quantised here); q4 is a language model's default only.
    given = any(x == "--weights" or x.startswith("--weights=")
                for x in (sys.argv[1:] if argv is None else argv))
    if given and a.weights == "q4":
        raise SystemExit("rayo-convert: --weights q4 is for language models; a TFLite or ONNX "
                         "model takes float32 or int8")
    try:
        classes = quantise.parse_classes(a.int8_tensors)
    except ConvertError as e:
        raise SystemExit(f"rayo-convert: {e}") from e
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
        weights="int8" if a.weights == "int8" else "float32",
        int8_tensors=classes,
        int8_min_size=a.int8_min_size,
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


def _language_model(a: argparse.Namespace) -> int:
    from . import hf

    out = Path(a.output) if a.output else Path(a.model.rstrip("/") + ".weights")
    try:
        opts = hf.HfOptions(name=a.name, weights=a.weights, context=a.context,
                            positions="all" if a.positions == "all" else "last", chunk=a.chunk,
                            metadata=json.loads(a.metadata) if a.metadata else None,
                            scales=a.scales, q3=hf.q3_roles(a.q3),
                            tokenizer_table=not a.no_tokenizer_table)
        size, notes = hf.convert_dir(a.model, out, opts)
    except ConvertError as e:
        print(f"rayo-convert: {a.model}: cannot convert: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"rayo-convert: {e}", file=sys.stderr)
        return 2
    if not a.quiet:
        for line in notes:
            print(f"note: {line}")
        print(f"wrote {out} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
