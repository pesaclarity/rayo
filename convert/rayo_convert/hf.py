"""Hugging Face decoder-only language models (a directory with ``config.json`` and
``*.safetensors``) to rayo models.

Everything the graph needs comes from the model's own ``config.json``: the number of layers,
heads and key/value heads, the head size, the rotary embedding's base, scaling and rotated
dimensions, the norm's epsilon, the activation, whether the embedding is tied to the output,
and the vocabulary. The tensors are found by the names the Llama family of models uses
(``model.layers.{i}.self_attn.q_proj.weight`` and so on), which Llama, Mistral, Qwen2, SmolLM
and others share; optional biases (Qwen2's on q, k and v) are used when they are there, and
fused projections (``qkv_proj``, ``gate_up_proj``) are split. No model has code of its own here:
what does not fit is refused, naming what.

The safetensors files are read with NumPy alone (no ``safetensors`` or ``torch`` package), and
the rayo file is written tensor by tensor, so converting a model takes about one tensor's
memory beyond the files.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .ir import ConvertError

GROUP = 32  # int4 group size: rayo's RAYO_PCWT_INT4_GROUP
ALIGN = 64  # tensors start at multiples of this in the file, for mapped reads

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64}


class SafeTensors:
    """The tensors of one or more ``.safetensors`` files, mapped, read on demand as float32."""

    def __init__(self, files: list[Path]) -> None:
        self.where: dict[str, tuple[np.memmap, dict[str, Any]]] = {}
        for f in files:
            with f.open("rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            mm = np.memmap(f, dtype=np.uint8, mode="r")
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if info["dtype"] not in _ST_DTYPES:
                    raise ConvertError(f"tensor {name}: dtype {info['dtype']} is not read")
                info = dict(info, base=8 + n)
                self.where[name] = (mm, info)

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def names(self) -> list[str]:
        return list(self.where)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.where[name][1]["shape"])

    def get(self, name: str) -> np.ndarray:
        mm, info = self.where[name]
        a, b = info["data_offsets"]
        raw = mm[info["base"] + a: info["base"] + b]
        dt = _ST_DTYPES[info["dtype"]]
        arr = np.frombuffer(raw, dtype=np.dtype(dt).newbyteorder("<"))
        if info["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32).reshape(info["shape"])


def open_safetensors(d: Path) -> SafeTensors:
    index = d / "model.safetensors.index.json"
    if index.exists():
        files = sorted({d / f for f in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(d.glob("*.safetensors"))
    if not files:
        raise ConvertError(f"{d}: no .safetensors files")
    return SafeTensors(files)


@dataclass
class Decoder:
    """What the graph needs, read from config.json."""

    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    eps: float
    activation: str  # silu, gelu_erf or gelu_tanh: the gate's
    rotary_dims: int
    inv_freq: np.ndarray  # [rotary_dims / 2], float32
    attention_scaling: float
    tied: bool
    context: int  # the most positions a session plans for
    notes: list[str] = field(default_factory=list)


_ACTIVATIONS = {"silu": "silu", "swish": "silu", "gelu": "gelu_erf",
                "gelu_pytorch_tanh": "gelu_tanh", "gelu_new": "gelu_tanh",
                "gelu_fast": "gelu_tanh"}


def rope_frequencies(cfg: dict[str, Any], rotary_dims: int) -> tuple[np.ndarray, float, str]:
    """The rotary embedding's inverse frequencies, its cos/sin scaling and a note, as the
    model's configuration describes them (the rope_scaling types of the transformers
    library that do not depend on the sequence's length)."""
    theta = float(cfg.get("rope_theta", 10000.0))
    base = 1.0 / theta ** (np.arange(0, rotary_dims, 2, dtype=np.int64).astype(np.float32)
                           / rotary_dims)
    base = base.astype(np.float32)
    sc = cfg.get("rope_scaling") or {}
    kind = sc.get("rope_type", sc.get("type", "default")) if sc else "default"
    if kind == "default":
        return base, 1.0, f"rope theta {theta:g}"
    if kind == "linear":
        f = float(sc["factor"])
        return (base / f).astype(np.float32), 1.0, f"rope theta {theta:g}, linear scaling {f:g}"
    if kind == "llama3":
        f = float(sc["factor"])
        lo, hi = float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
        old = float(sc["original_max_position_embeddings"])
        lo_wl, hi_wl = old / lo, old / hi
        wl = 2 * math.pi / base
        out = np.where(wl > lo_wl, base / f, base)
        smooth = (old / wl - lo) / (hi - lo)
        mid = (1 - smooth) * out / f + smooth * out
        medium = (wl >= hi_wl) & (wl <= lo_wl)
        out = np.where(medium, mid, out).astype(np.float32)
        return out, 1.0, f"rope theta {theta:g}, llama3 scaling {f:g}"
    if kind == "dynamic":
        # Dynamic NTK changes nothing up to the original length, which a session plans for.
        return base, 1.0, f"rope theta {theta:g} (dynamic scaling: none within the context)"
    raise ConvertError(f"rope_scaling {kind!r} is not converted (default, linear, llama3, "
                       "dynamic are)")


def read_config(d: Path, context: int | None) -> Decoder:
    cfg = json.loads((d / "config.json").read_text())
    if "text_config" in cfg:
        raise ConvertError("a multimodal model: convert its text model alone")
    missing = [k for k in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                           "intermediate_size", "vocab_size") if k not in cfg]
    if missing:
        raise ConvertError(f"config.json lacks {', '.join(missing)}: not a decoder-only model")
    if cfg.get("is_encoder_decoder"):
        raise ConvertError("an encoder-decoder model: only decoder-only models are converted")
    heads = int(cfg["num_attention_heads"])
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    act = cfg.get("hidden_act", cfg.get("hidden_activation", "silu"))
    if act not in _ACTIVATIONS:
        raise ConvertError(f"hidden_act {act!r} is not converted ({', '.join(_ACTIVATIONS)} are)")
    rotary = int(head_dim * float(cfg.get("partial_rotary_factor", 1.0)))
    rotary -= rotary % 2
    inv, scaling, note = rope_frequencies(cfg, rotary)
    longest = int(cfg.get("max_position_embeddings", 2048))
    ctx = min(context or 2048, longest)
    window = cfg.get("sliding_window")
    if window and cfg.get("use_sliding_window", True) and int(window) < ctx:
        raise ConvertError(f"sliding-window attention ({window}) shorter than the context {ctx}: "
                           f"convert with --context {window} or less")
    for k in ("attn_logit_softcapping", "final_logit_softcapping"):
        if cfg.get(k):
            raise ConvertError(f"{k} is not converted")
    return Decoder(
        layers=int(cfg["num_hidden_layers"]), hidden=hidden, heads=heads,
        kv_heads=int(cfg.get("num_key_value_heads") or heads), head_dim=head_dim,
        intermediate=int(cfg["intermediate_size"]), vocab=int(cfg["vocab_size"]),
        eps=float(cfg.get("rms_norm_eps", 1e-6)), activation=_ACTIVATIONS[act],
        rotary_dims=rotary, inv_freq=inv, attention_scaling=scaling,
        tied=bool(cfg.get("tie_word_embeddings", False)), context=ctx,
        notes=[note, f"context {ctx} of {longest}"])


# ---- quantisation -----------------------------------------------------------------------------

def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 with a scale per row (rayo's int8 kernels' layout)."""
    top = np.abs(w).max(axis=1)
    scale = np.where(top > 0, top / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def quantize_int4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-bit, a float32 scale per group of 32 along the rows (docs/FORMAT.md, ``int4``): the
    value of largest magnitude maps to -8, the scale is it / -8, values round to nearest in
    [-8, 7] and are stored plus 8, byte j of a group holding value j (low four bits) and
    value j + 16 (high four)."""
    rows, k = w.shape
    if k % GROUP:
        raise ConvertError(f"int4 needs rows of a multiple of {GROUP}, not {k}")
    g = w.reshape(rows, k // GROUP, GROUP)
    idx = np.abs(g).argmax(axis=2)
    top = np.take_along_axis(g, idx[..., None], axis=2)[..., 0]
    scale = (top / -8.0).astype(np.float32)
    inv = np.where(scale != 0, 1.0 / np.where(scale != 0, scale, 1), 0.0).astype(np.float32)
    q = np.clip(np.round(g * inv[..., None]), -8, 7).astype(np.int16) + 8
    q = q.astype(np.uint8)
    packed = q[..., :16] | (q[..., 16:] << 4)
    return packed.reshape(rows, k // 2), scale.reshape(rows, k // GROUP)


def dequantize_int4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows = packed.shape[0]
    p = packed.reshape(rows, -1, 16)
    q = np.concatenate([p & 15, p >> 4], axis=2).astype(np.float32) - 8
    return (q * scale[..., None]).reshape(rows, -1).astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale[:, None]).astype(np.float32)


# ---- the file --------------------------------------------------------------------------------

@dataclass
class Planned:
    """A tensor of the output file: its entry and how to make its bytes."""

    name: str
    dtype: str  # float32, int8 or int4
    shape: list[int]
    make: Any  # () -> (data bytes, scales bytes or None)
    nbytes: int
    scale_bytes: int = 0
    entry: dict[str, Any] = field(default_factory=dict)


def _layout(tensors: list[Planned]) -> None:
    at = 0
    for t in tensors:
        t.entry = {"name": t.name, "dtype": t.dtype, "shape": t.shape, "offset": at,
                   "bytes": t.nbytes}
        at += t.nbytes + (-t.nbytes % ALIGN)
        if t.dtype == "int8":
            t.entry["quantization"] = {"axis": 0, "zero_point": 0, "scales_offset": at,
                                       "scales_bytes": t.scale_bytes}
            at += t.scale_bytes + (-t.scale_bytes % ALIGN)
        elif t.dtype == "int4":
            t.entry["quantization"] = {"group_size": GROUP, "zero_point": 8,
                                       "scales_offset": at, "scales_bytes": t.scale_bytes}
            at += t.scale_bytes + (-t.scale_bytes % ALIGN)


def write_file(path: Path, header: dict[str, Any], tensors: list[Planned]) -> int:
    """Writes PCWT version 1, its data starting (and each tensor starting) at a multiple of
    ALIGN bytes from the start of the file, so a mapped file's weights are aligned. The data
    ends on the last tensor's padded end, as rayo checks."""
    _layout(tensors)
    # The last tensor's padding is to 4, not ALIGN (rayo checks the data's end exactly).
    last = tensors[-1]
    last_end = (last.entry.get("quantization", {}).get("scales_offset", last.entry["offset"])
                + (last.scale_bytes if last.dtype != "float32" else last.nbytes))
    last_end += -last_end % 4
    full = {**header, "tensors": [t.entry for t in tensors]}
    h = json.dumps(full, separators=(",", ":")).encode()
    h += b" " * (-(12 + len(h)) % ALIGN)
    with path.open("wb") as f:
        f.write(b"PCWT" + struct.pack("<II", 1, len(h)) + h)
        written = 0
        for t in tensors:
            data, scales = t.make()
            _put(f, t.entry["offset"] - written)
            f.write(data)
            written = t.entry["offset"] + len(data)
            if scales is not None:
                q = t.entry["quantization"]
                _put(f, q["scales_offset"] - written)
                f.write(scales)
                written = q["scales_offset"] + len(scales)
        _put(f, last_end - written)
    return 12 + len(h) + last_end


def _put(f: BinaryIO, n: int) -> None:
    if n < 0:
        raise AssertionError("tensors out of order")
    f.write(b"\0" * n)


# ---- the graph -------------------------------------------------------------------------------

@dataclass
class HfOptions:
    name: str | None = None
    weights: str = "q4"  # float32, int8 or q4
    context: int | None = None
    positions: str = "last"  # last or all
    metadata: dict[str, Any] | None = None


def _proj(st: SafeTensors, prefix: str, parts: list[tuple[str, int]],
          fused: str | None) -> dict[str, tuple[str, slice | None]]:
    """Where each projection's weight is: its own tensor, or rows of a fused one."""
    out: dict[str, tuple[str, slice | None]] = {}
    if all(f"{prefix}.{p}.weight" in st for p, _ in parts):
        for p, _ in parts:
            out[p] = (f"{prefix}.{p}.weight", None)
        return out
    if fused and f"{prefix}.{fused}.weight" in st:
        at = 0
        for p, n in parts:
            out[p] = (f"{prefix}.{fused}.weight", slice(at, at + n))
            at += n
        return out
    raise ConvertError(f"{prefix}: no {', '.join(p for p, _ in parts)} weights"
                       + (f" (nor {fused})" if fused else ""))


def build(src: Path, opts: HfOptions) -> tuple[dict[str, Any], list[Planned], list[str]]:
    """The header (graph and metadata) and the tensors to write."""
    dec = read_config(src, opts.context)
    st = open_safetensors(src)
    notes = list(dec.notes)
    if opts.weights not in ("float32", "int8", "q4"):
        raise ConvertError(f"weights {opts.weights!r}: float32, int8 or q4")
    tensors: list[Planned] = []
    have: dict[str, str] = {}

    def plan_float(name: str, src_name: str, rows: slice | None = None,
                   shape: list[int] | None = None) -> str:
        def make() -> tuple[bytes, None]:
            a = st.get(src_name)
            if rows is not None:
                a = a[rows]
            return np.ascontiguousarray(a, "<f4").tobytes(), None
        s = list(shape or st.shape(src_name))
        if rows is not None:
            s[0] = rows.stop - rows.start
        tensors.append(Planned(name, "float32", s, make, int(np.prod(s)) * 4))
        return name

    def plan_matrix(name: str, src_name: str, rows: slice | None = None) -> str:
        key = f"{src_name}|{rows}"
        if key in have:
            return have[key]
        shape = list(st.shape(src_name))
        if rows is not None:
            shape[0] = rows.stop - rows.start
        if len(shape) != 2:
            raise ConvertError(f"{src_name}: a matrix expected, shape {shape}")
        mode = opts.weights
        if mode == "q4" and shape[1] % GROUP:
            notes.append(f"{name}: {shape[1]} columns are not a multiple of {GROUP}: int8")
            mode = "int8"

        def load() -> np.ndarray:
            a = st.get(src_name)
            return a[rows] if rows is not None else a

        if mode == "float32":
            plan_float(name, src_name, rows)
        elif mode == "int8":
            def make8() -> tuple[bytes, bytes]:
                q, s = quantize_int8(load())
                return q.tobytes(), s.astype("<f4").tobytes()
            tensors.append(Planned(name, "int8", shape, make8, shape[0] * shape[1],
                                   shape[0] * 4))
        else:
            def make4() -> tuple[bytes, bytes]:
                q, s = quantize_int4(load())
                return q.tobytes(), s.astype("<f4").tobytes()
            tensors.append(Planned(name, "int4", shape, make4, shape[0] * shape[1] // 2,
                                   shape[0] * shape[1] // GROUP * 4))
        have[key] = name
        return name

    hd, H, KV = dec.head_dim, dec.heads, dec.kv_heads
    embed_name = "model.embed_tokens.weight"
    if embed_name not in st:
        raise ConvertError("no model.embed_tokens.weight: not a Llama-style decoder")
    if list(st.shape(embed_name)) != [dec.vocab, dec.hidden]:
        notes.append(f"embedding {st.shape(embed_name)}, config vocab {dec.vocab}")
    vocab = st.shape(embed_name)[0]
    emb = plan_matrix("embed", embed_name)
    nodes: list[dict[str, Any]] = [
        {"op": "range", "inputs": [], "output": "pos", "start": "P", "size": "T"},
        {"op": "gather", "inputs": [emb, "tokens"], "output": "x0"},
    ]
    freq = "rope.inv_freq"
    tensors.append(Planned(freq, "float32", [len(dec.inv_freq)],
                           lambda: (dec.inv_freq.astype("<f4").tobytes(), None),
                           len(dec.inv_freq) * 4))
    states = []
    x = "x0"
    rope: dict[str, Any] = {"rotary_dims": dec.rotary_dims, "style": "half"}
    if dec.attention_scaling != 1.0:
        rope["attention_scaling"] = dec.attention_scaling
    for i in range(dec.layers):
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        qkv = _proj(st, a, [("q_proj", H * hd), ("k_proj", KV * hd), ("v_proj", KV * hd)],
                    "qkv_proj")
        mlp = _proj(st, f"{p}.mlp", [("gate_proj", dec.intermediate),
                                     ("up_proj", dec.intermediate)], "gate_up_proj")
        for norm in ("input_layernorm", "post_attention_layernorm"):
            if f"{p}.{norm}.weight" not in st:
                raise ConvertError(f"{p}: no {norm}.weight (a Llama-style pre-norm block)")
        extra = [n for n in (f"{p}.input_layernorm.bias", f"{a}.o_proj.bias",
                             f"{p}.mlp.down_proj.bias", f"{a}.q_norm.weight") if n in st]
        if extra:
            raise ConvertError(f"{p}: {', '.join(extra)} not converted (no op takes them yet)")
        ln1 = plan_float(f"l{i}.ln1", f"{p}.input_layernorm.weight")
        ln2 = plan_float(f"l{i}.ln2", f"{p}.post_attention_layernorm.weight")
        dense = {}
        for part, (sname, rows) in qkv.items():
            bias = []
            bname = f"{a}.{part}.bias"
            if bname in st:
                bias = [plan_float(f"l{i}.{part[0]}.bias", bname)]
            elif rows is not None and f"{a}.qkv_proj.bias" in st:
                bias = [plan_float(f"l{i}.{part[0]}.bias", f"{a}.qkv_proj.bias", rows)]
            dense[part] = [plan_matrix(f"l{i}.{part[0]}", sname, rows), *bias]
        o = plan_matrix(f"l{i}.o", f"{a}.o_proj.weight")
        gate = plan_matrix(f"l{i}.gate", *mlp["gate_proj"])
        up = plan_matrix(f"l{i}.up", *mlp["up_proj"])
        down = plan_matrix(f"l{i}.down", f"{p}.mlp.down_proj.weight")
        states += [{"name": f"keys_{i}", "dtype": "float32", "shape": [dec.context, KV * hd]},
                   {"name": f"values_{i}", "dtype": "float32", "shape": [dec.context, KV * hd]}]
        n = f"l{i}"
        nodes += [
            {"op": "rms_norm", "inputs": [x, ln1], "output": f"{n}.h", "epsilon": dec.eps},
            {"op": "dense", "inputs": [f"{n}.h", *dense["q_proj"]], "output": f"{n}.q"},
            {"op": "dense", "inputs": [f"{n}.h", *dense["k_proj"]], "output": f"{n}.k"},
            {"op": "dense", "inputs": [f"{n}.h", *dense["v_proj"]], "output": f"{n}.v"},
            {"op": "rope", "inputs": [f"{n}.q", "pos", freq], "output": f"{n}.qr", "heads": H,
             **rope},
            {"op": "rope", "inputs": [f"{n}.k", "pos", freq], "output": f"{n}.kr", "heads": KV,
             **rope},
            {"op": "cache_write", "inputs": [f"keys_{i}", f"{n}.kr"], "output": f"{n}.ks",
             "at": "P"},
            {"op": "cache_write", "inputs": [f"values_{i}", f"{n}.v"], "output": f"{n}.vs",
             "at": "P"},
            {"op": "attention", "inputs": [f"{n}.qr", f"{n}.ks", f"{n}.vs"], "output": f"{n}.a",
             "heads": H, "kv_heads": KV, "causal": 1},
            {"op": "dense", "inputs": [f"{n}.a", o], "output": f"{n}.o"},
            {"op": "add", "inputs": [x, f"{n}.o"], "output": f"{n}.x1"},
            {"op": "rms_norm", "inputs": [f"{n}.x1", ln2], "output": f"{n}.h2",
             "epsilon": dec.eps},
            {"op": "dense", "inputs": [f"{n}.h2", gate], "output": f"{n}.g"},
            {"op": "dense", "inputs": [f"{n}.h2", up], "output": f"{n}.u"},
        ]
        if dec.activation == "silu":
            nodes.append({"op": "swiglu", "inputs": [f"{n}.g", f"{n}.u"], "output": f"{n}.m"})
        else:
            nodes += [{"op": dec.activation, "inputs": [f"{n}.g"], "output": f"{n}.ga"},
                      {"op": "mul", "inputs": [f"{n}.ga", f"{n}.u"], "output": f"{n}.m"}]
        nodes += [
            {"op": "dense", "inputs": [f"{n}.m", down], "output": f"{n}.d"},
            {"op": "add", "inputs": [f"{n}.x1", f"{n}.d"], "output": f"{n}.x"},
        ]
        x = f"{n}.x"
    fin = plan_float("norm", "model.norm.weight")
    if opts.positions == "last":
        nodes.append({"op": "last_row", "inputs": [x], "output": "xl"})
        x = "xl"
    if "lm_head.weight" in st and not dec.tied:
        head = plan_matrix("lm_head", "lm_head.weight")
    else:
        head = emb
        notes.append("output tied to the embedding")
    nodes += [
        {"op": "rms_norm", "inputs": [x, fin], "output": "xn", "epsilon": dec.eps},
        {"op": "dense", "inputs": ["xn", head], "output": "logits"},
    ]
    graph = {
        "version": 1,
        "symbols": {"T": dec.context, "P": dec.context - 1},
        "plan": {"P": 0},
        "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]}],
        "states": states,
        "nodes": nodes,
        "outputs": [{"name": "logits", "node": "logits"}],
    }
    meta = {
        "architecture": "decoder",
        "layers": dec.layers, "hidden": dec.hidden, "heads": H, "kv_heads": KV,
        "head_dim": hd, "vocab": vocab, "context": dec.context, "weights": opts.weights,
    }
    meta.update(generation_metadata(src))
    if opts.metadata:
        meta.update(opts.metadata)
    header = {"format_version": 1, "model": opts.name or src.name, "source": "huggingface",
              "graph": graph, "metadata": meta}
    notes.append(f"{dec.layers} layers, {H} heads ({KV} key/value), head {hd}, "
                 f"{dec.activation}, weights {opts.weights}")
    return header, tensors, notes


# ---- the tokenizer and the chat template -----------------------------------------------------

def _ids(v: Any) -> list[int]:
    if v is None:
        return []
    return [int(x) for x in (v if isinstance(v, list) else [v])]


def generation_metadata(src: Path) -> dict[str, Any]:
    """The tokens that end a reply and whether a prompt starts with BOS, from the model's
    generation_config.json, config.json and tokenizer_config.json; and its chat template,
    rendered into the pieces around each turn (``chat_turns``)."""
    meta: dict[str, Any] = {}
    cfg = json.loads((src / "config.json").read_text())
    gen_p = src / "generation_config.json"
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}
    tc_p = src / "tokenizer_config.json"
    tc = json.loads(tc_p.read_text()) if tc_p.exists() else {}
    eos = _ids(gen.get("eos_token_id")) or _ids(cfg.get("eos_token_id"))
    tok_p = src / "tokenizer.json"
    if tok_p.exists():
        tok = json.loads(tok_p.read_text())
        added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
        name = tc.get("eos_token")
        if isinstance(name, dict):
            name = name.get("content")
        if name in added and added[name] not in eos:
            eos.append(added[name])
    meta["eos_token_ids"] = eos
    bos = gen.get("bos_token_id", cfg.get("bos_token_id"))
    if bos is not None:
        meta["bos_token_id"] = int(bos)
    meta["add_bos_token"] = bool(tc.get("add_bos_token", False))
    template = tc.get("chat_template")
    if isinstance(template, list):  # named templates: the default one
        template = next((t["template"] for t in template if t.get("name") == "default"), None)
    if template:
        try:
            meta["chat_turns"] = chat_turns(template, tc)
        except ImportError:
            meta["chat_note"] = "jinja2 was not installed: no chat template"
        except Exception as e:  # a template this rendering does not fit
            meta["chat_note"] = f"chat template not rendered: {e}"
    return meta


def chat_turns(template: str, tc: dict[str, Any]) -> dict[str, str]:
    """The model's own chat template (Jinja), rendered on conversations whose texts are marker
    characters, cut at the markers into the pieces a program puts around each turn's text:

    - ``first``: before the first user text (the template's default system prompt, if any);
    - ``system`` and ``after_system``: before and after a system text, when there is one;
    - ``user_end``: after a user text, up to where the reply starts;
    - ``next``: after a reply, up to the next user text.

    A conversation is then first + user + user_end, reply, next + user + user_end, reply, ...
    (or system + text + after_system + user + user_end, ...)."""
    import jinja2  # optional: rayo-convert[chat]
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(m: str) -> None:
        raise jinja2.TemplateError(m)

    env.globals["raise_exception"] = raise_exception
    t = env.from_string(template)

    def special(k: str) -> str:
        v = tc.get(k) or ""
        return v.get("content", "") if isinstance(v, dict) else v

    def render(msgs: list[dict[str, str]]) -> str:
        return t.render(messages=msgs, add_generation_prompt=True, bos_token=special("bos_token"),
                        eos_token=special("eos_token"), tools=None)

    U, A, U2, S = "\x11", "\x12", "\x13", "\x14"
    one = render([{"role": "user", "content": U}])
    first, user_end = one.split(U)
    two = render([{"role": "user", "content": U}, {"role": "assistant", "content": A},
                  {"role": "user", "content": U2}])
    if not two.startswith(first) or two.count(A) != 1:
        raise ValueError("turns do not repeat")
    nxt = two.split(A)[1].split(U2)[0]
    if two.split(U2)[1] != user_end:
        raise ValueError("turns end differently")
    out = {"first": first, "user_end": user_end, "next": nxt}
    try:
        sys = render([{"role": "system", "content": S}, {"role": "user", "content": U}])
        before, rest = sys.split(S)
        out["system"], out["after_system"] = before, rest.split(U)[0]
    except Exception:
        pass
    return out


def convert_dir(src: str | Path, out: str | Path, opts: HfOptions | None = None) -> tuple[int, list[str]]:
    """Converts the model in ``src`` to the rayo file ``out``, and copies its tokenizer.json
    next to it (``out`` with ``.tokenizer.json``). The file's size and the notes."""
    opts = opts or HfOptions()
    src, out = Path(src), Path(out)
    header, tensors, notes = build(src, opts)
    size = write_file(out, header, tensors)
    tok = src / "tokenizer.json"
    if tok.exists():
        dest = tokenizer_path(out)
        dest.write_bytes(tok.read_bytes())
        notes.append(f"tokenizer: {dest}")
    return size, notes


def tokenizer_path(model: Path) -> Path:
    return model.with_name(model.stem + ".tokenizer.json")
