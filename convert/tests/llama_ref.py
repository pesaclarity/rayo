"""A NumPy reference for Llama-style decoders, written from the models' definition and not
from rayo's graph, and tiny random models of that kind written as Hugging Face directories
(config.json and model.safetensors), for rayo-convert's tests. No downloads."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], bf16: bool = False) -> None:
    header: dict[str, Any] = {}
    blobs = []
    at = 0
    for name, a in tensors.items():
        if bf16:
            raw = (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype("<u2")
            dtype = "BF16"
        else:
            raw = np.ascontiguousarray(a, "<f4")
            dtype = "F32"
        b = raw.tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def bf16_round(a: np.ndarray) -> np.ndarray:
    return ((np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16) << 16).view(np.float32)


def tiny_config(**over: Any) -> dict[str, Any]:
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
           "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
           "num_key_value_heads": 2, "intermediate_size": 96, "vocab_size": 300,
           "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu",
           "max_position_embeddings": 64, "tie_word_embeddings": True,
           "bos_token_id": 1, "eos_token_id": 2}
    cfg.update(over)
    return cfg


def tiny_weights(cfg: dict[str, Any], seed: int = 0, qkv_bias: bool = False,
                 fused: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    hd = cfg.get("head_dim") or D // H
    inter, V = cfg["intermediate_size"], cfg["vocab_size"]

    def m(*shape: int, s: float = 0.08) -> np.ndarray:
        return (rng.standard_normal(shape) * s).astype(np.float32)

    w = {"model.embed_tokens.weight": m(V, D, s=0.5), "model.norm.weight": 1 + m(D, s=0.1)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        q, k, v = m(H * hd, D), m(KV * hd, D), m(KV * hd, D)
        if fused:
            w[f"{p}.self_attn.qkv_proj.weight"] = np.concatenate([q, k, v])
        else:
            w[f"{p}.self_attn.q_proj.weight"] = q
            w[f"{p}.self_attn.k_proj.weight"] = k
            w[f"{p}.self_attn.v_proj.weight"] = v
        if qkv_bias:
            w[f"{p}.self_attn.q_proj.bias"] = m(H * hd, s=0.2)
            w[f"{p}.self_attn.k_proj.bias"] = m(KV * hd, s=0.2)
            w[f"{p}.self_attn.v_proj.bias"] = m(KV * hd, s=0.2)
        w[f"{p}.self_attn.o_proj.weight"] = m(D, H * hd)
        g, u = m(inter, D), m(inter, D)
        if fused:
            w[f"{p}.mlp.gate_up_proj.weight"] = np.concatenate([g, u])
        else:
            w[f"{p}.mlp.gate_proj.weight"] = g
            w[f"{p}.mlp.up_proj.weight"] = u
        w[f"{p}.mlp.down_proj.weight"] = m(D, inter)
        w[f"{p}.input_layernorm.weight"] = 1 + m(D, s=0.1)
        w[f"{p}.post_attention_layernorm.weight"] = 1 + m(D, s=0.1)
    if not cfg.get("tie_word_embeddings", False):
        w["lm_head.weight"] = m(V, D, s=0.3)
    return w


def write_model(d: Path, cfg: dict[str, Any], w: dict[str, np.ndarray],
                extra: dict[str, Any] | None = None, bf16: bool = False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    write_safetensors(d / "model.safetensors", w, bf16=bf16)
    for name, obj in (extra or {}).items():
        (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return d


def _split(w: dict[str, np.ndarray], p: str, a: str, parts: list[tuple[str, int]],
           fused: str) -> list[np.ndarray]:
    if f"{p}.{parts[0][0]}.weight" in w:
        return [w[f"{p}.{n}.weight"] for n, _ in parts]
    f, out, at = w[f"{p}.{fused}.weight"], [], 0
    for _, n in parts:
        out.append(f[at:at + n])
        at += n
    del a
    return out


def forward(cfg: dict[str, Any], w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """Logits [len(ids), vocab] in float64, every position, as the transformers library's
    Llama computes them (rotate_half rotary embeddings, repeated key/value heads)."""
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    hd = cfg.get("head_dim") or D // H
    inter, eps = cfg["intermediate_size"], cfg["rms_norm_eps"]
    W = {k: v.astype(np.float64) for k, v in w.items()}
    T = len(ids)
    rot = int(hd * cfg.get("partial_rotary_factor", 1.0))
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    def rms(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return g * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))

    def rope(x: np.ndarray, heads: int) -> np.ndarray:
        x = x.reshape(T, heads, hd).copy()
        a, b = x[..., : rot // 2].copy(), x[..., rot // 2: rot].copy()
        c, s = cos[:, None, :], sin[:, None, :]
        x[..., : rot // 2] = a * c - b * s
        x[..., rot // 2: rot] = b * c + a * s
        return x

    act = {"silu": lambda v: v / (1 + np.exp(-v)),
           "gelu_pytorch_tanh": lambda v: 0.5 * v * (1 + np.tanh(0.7978845608028654
                                                                 * (v + 0.044715 * v ** 3)))}
    f = act[cfg.get("hidden_act", "silu")]
    x = W["model.embed_tokens.weight"][ids]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        h = rms(x, W[f"{p}.input_layernorm.weight"])
        wq, wk, wv = _split(W, a, a, [("q_proj", H * hd), ("k_proj", KV * hd),
                                      ("v_proj", KV * hd)], "qkv_proj")
        q, k, v = h @ wq.T, h @ wk.T, h @ wv.T
        if f"{a}.q_proj.bias" in W:
            q, k, v = q + W[f"{a}.q_proj.bias"], k + W[f"{a}.k_proj.bias"], v + W[f"{a}.v_proj.bias"]
        q, k = rope(q, H), rope(k, KV)
        v = v.reshape(T, KV, hd)
        rep = H // KV
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, v).reshape(T, H * hd)
        x = x + o @ W[f"{a}.o_proj.weight"].T
        h = rms(x, W[f"{p}.post_attention_layernorm.weight"])
        g, u = _split(W, f"{p}.mlp", "", [("gate_proj", inter), ("up_proj", inter)],
                      "gate_up_proj")
        x = x + (f(h @ g.T) * (h @ u.T)) @ W[f"{p}.mlp.down_proj.weight"].T
    head = W.get("lm_head.weight") if not cfg.get("tie_word_embeddings") else None
    if head is None:
        head = W["model.embed_tokens.weight"]
    return rms(x, W["model.norm.weight"]) @ head.T
