"""rayo's C tokeniser (include/rayo/text.h) on tokenizer.json files made here, against the
tokenizers library that writes them, when it is installed, and against ids worked out by
hand when it is not. Synthetic text only."""

from __future__ import annotations

import json

import pytest

from rayo_convert.runtime import Tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's 2026; we're at 3.14159!",
    "Hello world\n\nHello   world\tagain   \n  indented line",
    "naïve café résumé — “quotes” and ‘apostrophes’, emoji 🙂🚀 and 中文字符 mixed",
    "numbers 1234567 and 42, code: def f(x): return x**2 # comment",
    "Don't you'll they've I'm we'd SHE'LL It'S",
] * 20

TEXTS = [
    "The lazy dog's café, 2026!",
    "  leading spaces and trailing   ",
    "line one\nline two\r\n\r\nend",
    "emoji 🙂 mixed 中文 and naïve",
    "x**2 + y**2 = 1234567890",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "unseenwordsthatmergeoddly ZZZQQQ",
    "",
]

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def _bpe(pre: str):
    tk = pytest.importorskip("tokenizers")
    from tokenizers import Regex, decoders, models, pre_tokenizers, trainers

    t = tk.Tokenizer(models.BPE())
    if pre == "gpt2":
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    elif pre == "digits":  # SmolLM2's: digits apart, then GPT-2's
        t.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False)])
    else:  # Qwen2's split, then byte-level without its own pattern
        pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
                   r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        t.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator(CORPUS, trainer)
    return t


def _sentencepiece_bpe():
    tk = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, normalizers, trainers

    t = tk.Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>"))
    t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"),
                                         normalizers.Replace(" ", "▁")])
    t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(),
                                   decoders.Fuse()])
    specials = ["<unk>", *SPECIALS] + [f"<0x{b:02X}>" for b in range(256)]
    t.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=700, special_tokens=specials))
    return t


def _unigram():
    tk = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, trainers

    t = tk.Tokenizer(models.Unigram())
    t.pre_tokenizer = pre_tokenizers.Metaspace()
    t.train_from_iterator(CORPUS, trainers.UnigramTrainer(vocab_size=300, special_tokens=SPECIALS,
                                                          unk_token="<|endoftext|>"))
    return t


@pytest.mark.parametrize("kind", ["gpt2", "digits", "qwen2", "sentencepiece", "unigram"])
def test_same_ids_as_the_tokenizers_library(lib, kind):
    t = {"sentencepiece": _sentencepiece_bpe, "unigram": _unigram}.get(kind,
                                                                      lambda: _bpe(kind))()
    ours = Tokenizer(lib, t.to_str().encode())
    for text in TEXTS + CORPUS[:5]:
        want = t.encode(text, add_special_tokens=False).ids
        got = ours.encode(text)
        assert got == want, (kind, text, got, want)
        if kind in ("gpt2", "digits", "qwen2"):
            assert ours.decode(got) == text
    # Added tokens are found in the text as themselves.
    end = t.token_to_id("<|im_end|>")
    assert ours.encode("a<|im_end|>b").count(end) == 1


def test_by_hand(lib):
    """A tiny byte-level BPE written out: merges apply lowest rank first."""
    vocab = {chr(c): i for i, c in enumerate(range(ord("a"), ord("z") + 1))}
    vocab |= {"Ġ": 26, "ab": 27, "abc": 28, "Ġab": 29, "<|end|>": 30}
    doc = {"version": "1.0", "added_tokens": [{"id": 30, "content": "<|end|>", "special": True}],
           "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                                                 "trim_offsets": True, "use_regex": True},
           "decoder": {"type": "ByteLevel"},
           "model": {"type": "BPE", "vocab": vocab, "merges": ["a b", "ab c", "Ġ ab"]}}
    t = Tokenizer(lib, json.dumps(doc).encode())
    assert t.encode("abc ab<|end|>") == [28, 29, 30]
    assert t.decode([28, 29, 30]) == "abc ab<|end|>"
    with pytest.raises(Exception, match="WordPiece"):
        Tokenizer(lib, json.dumps(doc | {"model": {"type": "WordPiece", "vocab": vocab}}).encode())
