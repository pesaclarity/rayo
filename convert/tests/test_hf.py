"""Hugging Face decoders (tiny random Llama-style models, made here) converted and run by rayo,
against the NumPy reference in llama_ref.py."""

from __future__ import annotations

import json

import numpy as np
import pytest
from llama_ref import bf16_round, forward, tiny_config, tiny_weights, write_model

from rayo_convert import hf
from rayo_convert.ir import ConvertError
from rayo_convert.runtime import Model, Session

PROMPTS = [[1, 5, 9, 200, 17, 3], [1, 250, 251, 252], [1, 42], [1, 7, 7, 7, 7, 7, 7, 7, 8, 9]]


def convert(tmp_path, cfg, w, weights="float32", positions="last", name="m", **kw):
    d = write_model(tmp_path / f"{name}-src", cfg, w, **kw)
    out = tmp_path / f"{name}.weights"
    hf.convert_dir(d, out, hf.HfOptions(weights=weights, positions=positions))
    return out


def prompt_logits(s: Session, ids: list[int]) -> np.ndarray:
    s.reset()
    s.bind("P", 0)
    s.bind("T", len(ids))
    s.run([np.asarray(ids, np.int32)])
    return s.output(0)


def decode_logits(s: Session, ids: list[int], prompt: int) -> list[np.ndarray]:
    """The last position's logits after the prompt, then after each further token."""
    out = [prompt_logits(s, ids[:prompt])]
    for p in range(prompt, len(ids)):
        s.bind("P", p)
        s.bind("T", 1)
        s.run([np.asarray(ids[p: p + 1], np.int32)])
        out.append(s.output(0))
    return out


def close(got, want, tol):
    err = float(np.abs(got - want).max() / max(1.0, np.abs(want).max()))
    assert err < tol, err
    return err


@pytest.mark.parametrize("variant", ["llama", "qwen2_bias", "fused_untied_gelu", "bf16"])
def test_float_matches_reference_every_position(lib, tmp_path, variant):
    cfg = tiny_config()
    kw = {}
    if variant == "qwen2_bias":
        cfg = tiny_config(model_type="qwen2", rope_theta=1000000.0)
        w = tiny_weights(cfg, qkv_bias=True)
    elif variant == "fused_untied_gelu":
        cfg = tiny_config(tie_word_embeddings=False, hidden_act="gelu_pytorch_tanh",
                          num_key_value_heads=4)
        w = tiny_weights(cfg, fused=True)
    else:
        w = tiny_weights(cfg)
    if variant == "bf16":
        w = {k: bf16_round(v) for k, v in w.items()}
        kw["bf16"] = True
    path = convert(tmp_path, cfg, w, positions="all", **kw)
    s = Session(Model(lib, path=path))
    for ids in PROMPTS:
        close(prompt_logits(s, ids), forward(cfg, w, ids), 2e-5)


def test_decoding_with_the_cache_matches_the_whole_sequence(lib, tmp_path):
    cfg = tiny_config()
    w = tiny_weights(cfg, seed=3)
    s = Session(Model(lib, path=convert(tmp_path, cfg, w)))
    ids = PROMPTS[3] + [11, 12, 13, 14, 15]
    want = forward(cfg, w, ids)
    for step, got in enumerate(decode_logits(s, ids, 4)):
        close(got, want[3 + step], 2e-5)


def test_threads_change_nothing(lib, tmp_path):
    cfg = tiny_config(intermediate_size=256)
    w = tiny_weights(cfg, seed=4)
    for weights in ("float32", "q4", "int8"):
        path = convert(tmp_path, cfg, w, weights=weights, name=weights)
        one = Session(Model(lib, path=path))
        four = Session(Model(lib, path=path), threads=4)
        assert four.threads() == 4
        for ids in PROMPTS:
            a, b = decode_logits(one, ids + [3, 4], len(ids)), decode_logits(four, ids + [3, 4],
                                                                            len(ids))
            for x, y in zip(a, b, strict=True):
                assert np.array_equal(x, y)


@pytest.mark.parametrize("weights,tol", [("int8", 0.05), ("q4", 0.25)])
def test_quantised_same_top_token(lib, tmp_path, weights, tol):
    """int8 and q4 against the float reference: the same next token at every position of the
    fixed prompts, logits within `tol` of the largest (int8: 5%, q4: 25%, on these random
    weights, which are harder to quantise than trained ones)."""
    cfg = tiny_config(hidden_size=128, intermediate_size=256, num_attention_heads=4)
    w = tiny_weights(cfg, seed=5)
    path = convert(tmp_path, cfg, w, weights=weights, positions="all")
    s = Session(Model(lib, path=path))
    agree = total = 0
    worst = 0.0
    for ids in PROMPTS:
        got, want = prompt_logits(s, ids), forward(cfg, w, ids)
        worst = max(worst, close(got, want, tol))
        agree += int((got.argmax(-1) == want.argmax(-1)).sum())
        total += len(ids)
    print(f"{weights}: top token {agree}/{total}, worst {worst:.3f}")
    assert agree >= total - 1


def test_quantised_against_its_own_dequantised_weights(lib, tmp_path):
    """q4: rayo's logits against the reference run on the dequantised int4 weights, which
    leaves only the activations' int8 rounding."""
    cfg = tiny_config(hidden_size=128, intermediate_size=256)
    w = tiny_weights(cfg, seed=6)
    deq = {k: (hf.dequantize_int4(*hf.quantize_int4(v)) if v.ndim == 2 else v)
           for k, v in w.items()}
    path = convert(tmp_path, cfg, w, weights="q4", positions="all")
    s = Session(Model(lib, path=path))
    for ids in PROMPTS:
        got, want = prompt_logits(s, ids), forward(cfg, deq, ids)
        close(got, want, 0.03)
        assert (got.argmax(-1) == want.argmax(-1)).all()


def test_int4_layout():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((3, 64)).astype(np.float32)
    q, s = hf.quantize_int4(w)
    assert q.shape == (3, 32) and s.shape == (3, 2)
    d = hf.dequantize_int4(q, s)
    # Each value within half a step; the largest of each group exact (it is -8 steps).
    assert np.all(np.abs(d - w) <= np.abs(s).repeat(32, axis=1) * 0.5 + 1e-6)
    # Byte j of a group: value j low, value j + 16 high.
    u = np.clip(np.round(w[0, :32] / s[0, 0]), -8, 7).astype(int) + 8
    assert q[0, 3] == u[3] | (u[19] << 4)


def test_mapped_and_read_load_the_same(lib, tmp_path):
    cfg = tiny_config()
    w = tiny_weights(cfg, seed=7)
    path = convert(tmp_path, cfg, w, weights="q4")
    a = prompt_logits(Session(Model(lib, path=path)), PROMPTS[0])
    b = prompt_logits(Session(Model(lib, path=path, read_file=True)), PROMPTS[0])
    c = prompt_logits(Session(Model(lib, data=path.read_bytes())), PROMPTS[0])
    assert np.array_equal(a, b) and np.array_equal(a, c)
    # Tensors are 64-byte aligned in the file, for mapped reads.
    raw = path.read_bytes()
    hlen = int.from_bytes(raw[8:12], "little")
    assert (12 + hlen) % 64 == 0
    header = json.loads(raw[12: 12 + hlen])
    assert all(t["offset"] % 64 == 0 for t in header["tensors"])


def test_refuses_what_it_cannot_convert(tmp_path):
    cfg = tiny_config(rope_scaling={"rope_type": "longrope", "factor": 2.0})
    d = write_model(tmp_path / "x", cfg, tiny_weights(tiny_config()))
    with pytest.raises(ConvertError, match="longrope"):
        hf.build(d, hf.HfOptions())
    cfg = tiny_config(hidden_act="relu2")
    d = write_model(tmp_path / "y", cfg, tiny_weights(tiny_config()))
    with pytest.raises(ConvertError, match="relu2"):
        hf.build(d, hf.HfOptions())


def test_rope_scaling_llama3():
    inv, scale, _ = hf.rope_frequencies(
        {"rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 8.0,
                                                   "low_freq_factor": 1.0,
                                                   "high_freq_factor": 4.0,
                                                   "original_max_position_embeddings": 8192}},
        64)
    base = 1.0 / 500000.0 ** (np.arange(0, 64, 2) / 64)
    assert scale == 1.0 and inv.shape == (32,)
    assert np.isclose(inv[0], base[0]) and np.isclose(inv[-1], base[-1] / 8)


CHATML = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")

HEADERS = (
    "{{ bos_token }}{% for m in messages %}{{ '<|start|>' + m['role'] + '<|sep|>\\n\\n' + "
    "m['content'] | trim + '<|end|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start|>assistant<|sep|>\\n\\n' }}{% endif %}")


def test_chat_turns_from_the_models_template():
    pytest.importorskip("jinja2")
    t = hf.chat_turns(CHATML, {})
    assert t["first"] == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
                         "<|im_start|>user\n"
    assert t["user_end"] == "<|im_end|>\n<|im_start|>assistant\n"
    assert t["next"] == "<|im_end|>\n<|im_start|>user\n"
    assert t["system"] == "<|im_start|>system\n" and t["after_system"] == \
        "<|im_end|>\n<|im_start|>user\n"
    t = hf.chat_turns(HEADERS, {"bos_token": "<s>"})
    assert t["first"] == "<s><|start|>user<|sep|>\n\n"
    assert t["next"] == "<|end|><|start|>user<|sep|>\n\n"


def test_metadata_has_stop_tokens_and_turns(tmp_path):
    pytest.importorskip("jinja2")
    cfg = tiny_config()
    tok = {"added_tokens": [{"id": 2, "content": "<|im_end|>", "special": True}],
           "model": {"type": "BPE", "vocab": {}, "merges": []}}
    d = write_model(tmp_path / "c", cfg, tiny_weights(cfg), extra={
        "tokenizer.json": tok,
        "tokenizer_config.json": {"chat_template": CHATML, "eos_token": "<|im_end|>"},
        "generation_config.json": {"eos_token_id": [2], "bos_token_id": 1}})
    header, _, _ = hf.build(d, hf.HfOptions())
    m = header["metadata"]
    assert m["eos_token_ids"] == [2] and m["chat_turns"]["user_end"].endswith("assistant\n")
