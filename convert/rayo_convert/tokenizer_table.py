"""The tokeniser's table (docs/FORMAT.md, "The tokeniser's table"): a tokenizer.json read the
way rayo's C reader reads it (src/text/tokenizer_json.c) and laid out as the same bytes, which
rayo reads in place from the mapped model file, with no JSON at run time. The C tests and
tests/test_tokenizer.py check that both give the same bytes."""

from __future__ import annotations

import json
import struct
from typing import Any

from .ir import ConvertError

VERSION = 1
HEADER_WORDS = 32
MAX_STEPS = 8
NO_PIECE = 0xFFFFFFFF

UNIGRAM, BYTE_LEVEL, BYTE_FALLBACK, IGNORE_MERGES, METASPACE_DECODE = 1, 2, 4, 8, 16
SPLIT, BYTELEVEL, DIGITS, METASPACE = 0, 1, 2, 3
USE_REGEX, PREFIX, REMOVED, INDIVIDUAL, SPLIT_WORDS, FIRST_ONLY = 1, 2, 4, 8, 16, 32
REPLACE, PREPEND = 0, 1
META = "▁"


def hash_bytes(s: bytes) -> int:
    h = 2166136261
    for c in s:
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h


def hash_pair(a: int, b: int) -> int:
    h = (((a & 0xFFFFFFFF) * 2654435761) & 0xFFFFFFFF) ^ ((b + 0x9E3779B9) & 0xFFFFFFFF)
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def slots(n: int) -> int:
    size = 16
    while size < 2 * n + 1:
        size *= 2
    return size


def _byte_chars() -> dict[str, int]:
    """GPT-2's byte-to-character mapping, reversed."""
    def prints(b: int) -> bool:
        return 33 <= b <= 126 or 161 <= b <= 172 or b >= 174
    out, n = {}, 0
    for b in range(256):
        if prints(b):
            out[chr(b)] = b
        else:
            out[chr(256 + n)] = b
            n += 1
    return out


_UNBYTE = _byte_chars()


def _unmap(s: str) -> bytes:
    return b"".join(bytes([_UNBYTE[c]]) if c in _UNBYTE else c.encode() for c in s)


def _bool(obj: Any, key: str, fallback: bool) -> bool:
    v = obj.get(key) if isinstance(obj, dict) else None
    return v if isinstance(v, bool) else fallback


def _str(obj: Any, key: str) -> str | None:
    v = obj.get(key) if isinstance(obj, dict) else None
    return v if isinstance(v, str) else None


def _int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and (v != int(v)):
        return None
    v = int(v)
    return v if -2**31 <= v < 2**31 else None


def _has_meta(v: Any) -> bool:
    """Whether a JSON value holds SentencePiece's ▁ anywhere (keys or strings)."""
    return META in json.dumps(v, ensure_ascii=False)


class _Builder:
    def __init__(self) -> None:
        self.flags = 0
        self.unk = -1
        self.pre: list[tuple[int, int, bytes]] = []
        self.norm: list[tuple[int, bytes, bytes]] = []
        self.byte_ids = [-1] * 256
        self.max_len = 0
        self.min_score = 0.0
        self.scores: list[float] | None = None
        self.added: list[tuple[bytes, int]] = []

    def normalizer(self, n: Any) -> None:
        if n is None:
            return
        t = _str(n, "type")
        if t is None:
            raise ConvertError("tokenizer: a normalizer without a type")
        if t == "Sequence":
            for x in n.get("normalizers") or []:
                self.normalizer(x)
            return
        if t == "NFC":
            return
        if len(self.norm) == MAX_STEPS:
            raise ConvertError("tokenizer: too many normalizers")
        if t == "Replace":
            f, to = _str(n.get("pattern"), "String"), _str(n, "content")
            if not f or to is None:
                raise ConvertError("tokenizer: Replace of a string only")
            self.norm.append((REPLACE, f.encode(), to.encode()))
            return
        if t == "Prepend":
            to = _str(n, "prepend")
            if to is None:
                raise ConvertError("tokenizer: Prepend without prepend")
            self.norm.append((PREPEND, b"", to.encode()))
            return
        raise ConvertError(f"tokenizer: normalizer {t}")

    def pre_tokenizer(self, n: Any) -> None:
        if n is None:
            return
        t = _str(n, "type")
        if t is None:
            raise ConvertError("tokenizer: a pre_tokenizer without a type")
        if t == "Sequence":
            for x in n.get("pretokenizers") or []:
                self.pre_tokenizer(x)
            return
        if len(self.pre) == MAX_STEPS:
            raise ConvertError("tokenizer: too many pre_tokenizers")
        if t == "Split":
            pattern = _str(n.get("pattern"), "Regex")
            literal = _str(n.get("pattern"), "String")
            behavior = _str(n, "behavior")
            if _bool(n, "invert", False) or behavior not in ("Isolated", "Removed"):
                raise ConvertError(f"tokenizer: Split with behavior {behavior}")
            if pattern is None and literal is not None:
                pattern = "".join("\\" + c if c in "\\.^$|?*+()[]{}" else c for c in literal)
            if pattern is None:
                raise ConvertError("tokenizer: Split without a pattern")
            self.pre.append((SPLIT, REMOVED if behavior == "Removed" else 0, pattern.encode()))
        elif t == "ByteLevel":
            f = (PREFIX if _bool(n, "add_prefix_space", False) else 0) | \
                (USE_REGEX if _bool(n, "use_regex", True) else 0)
            self.pre.append((BYTELEVEL, f, b""))
            self.flags |= BYTE_LEVEL
        elif t == "Digits":
            self.pre.append((DIGITS, INDIVIDUAL if _bool(n, "individual_digits", False) else 0,
                             b""))
        elif t == "Metaspace":
            rep = _str(n, "replacement")
            scheme = _str(n, "prepend_scheme")
            f = 0
            if (scheme != "never") if scheme is not None else _bool(n, "add_prefix_space", True):
                f |= PREFIX
            if scheme == "first":
                f |= FIRST_ONLY
            if _bool(n, "split", True):
                f |= SPLIT_WORDS
            self.pre.append((METASPACE, f, (rep if rep is not None else META).encode()))
            self.flags |= METASPACE_DECODE
        else:
            raise ConvertError(f"tokenizer: pre_tokenizer {t}")

    def set_piece(self, id_: int, s: str, special: bool) -> None:
        if self.piece[id_] is not None:
            return
        raw = s.encode()
        if (not self.flags & BYTE_LEVEL and len(raw) == 6 and raw[:3] == b"<0x"
                and raw[5:] == b">" and all(chr(c) in "0123456789abcdefABCDEF" for c in raw[3:5])):
            v = int(raw[3:5], 16)
            out = bytes([v])
            if self.byte_ids[v] < 0:
                self.byte_ids[v] = id_
        elif special:
            out = raw
        elif self.flags & BYTE_LEVEL:
            out = _unmap(s)
        elif self.flags & METASPACE_DECODE:
            out = raw.replace(META.encode(), b" ")
        else:
            out = raw
        self.piece[id_] = out

    def add_token(self, s: str, id_: int) -> None:
        form = _unmap(s) if self.flags & BYTE_LEVEL else s.encode()
        self.vocab_put(form, id_)
        self.max_len = max(self.max_len, len(form))
        self.set_piece(id_, s, False)

    def vocab_put(self, form: bytes, id_: int) -> None:
        mask = len(self.vocab_slots) - 1
        i = hash_bytes(form) & mask
        while self.vocab_slots[i] is not None:
            if self.vocab_slots[i][0] == form:
                return
            i = (i + 1) & mask
        self.vocab_slots[i] = (form, id_)

    def vocab_get(self, form: bytes) -> int:
        mask = len(self.vocab_slots) - 1
        i = hash_bytes(form) & mask
        while self.vocab_slots[i] is not None:
            if self.vocab_slots[i][0] == form:
                return self.vocab_slots[i][1]
            i = (i + 1) & mask
        return -1

    def merges_from(self, merges: Any) -> None:
        if not isinstance(merges, list):
            return
        self.merge_slots = [None] * slots(len(merges))
        mask = len(self.merge_slots) - 1
        for rank, m in enumerate(merges):
            if isinstance(m, str):
                if " " not in m:
                    continue
                a, b = m.split(" ", 1)
            elif isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) for x in m):
                a, b = m
            else:
                raise ConvertError(f"tokenizer: merge {rank}")
            if self.flags & BYTE_LEVEL:
                fa, fb = _unmap(a), _unmap(b)
            else:
                fa, fb = a.encode(), b.encode()
            ia, ib, im = self.vocab_get(fa), self.vocab_get(fb), self.vocab_get(fa + fb)
            if ia < 0 or ib < 0 or im < 0:
                continue
            i = hash_pair(ia, ib) & mask
            while self.merge_slots[i] is not None:
                if self.merge_slots[i][:2] == (ia, ib):
                    break
                i = (i + 1) & mask
            else:
                self.merge_slots[i] = (ia, ib, rank, im)

    def model(self, model: Any) -> None:
        t, vocab = _str(model, "type"), model.get("vocab") if isinstance(model, dict) else None
        if t is None or vocab is None:
            raise ConvertError("tokenizer: no model")
        if _bool(model, "byte_fallback", False):
            self.flags |= BYTE_FALLBACK
        if _bool(model, "ignore_merges", False):
            self.flags |= IGNORE_MERGES
        if t == "BPE":
            if not isinstance(vocab, dict):
                raise ConvertError("tokenizer: a BPE vocab is an object")
            for s, v in vocab.items():
                id_ = _int(v)
                if id_ is None or not 0 <= id_ < self.vocab_size:
                    raise ConvertError(f"tokenizer: id of {s}")
                self.add_token(s, id_)
            unk = _str(model, "unk_token")
            if unk is not None:
                self.unk = self.vocab_get(unk.encode())
            self.merges_from(model.get("merges"))
            return
        if t == "Unigram":
            if not isinstance(vocab, list):
                raise ConvertError("tokenizer: a Unigram vocab is a list")
            self.flags |= UNIGRAM
            self.scores = [0.0] * self.vocab_size
            self.min_score = 0.0
            for i, e in enumerate(vocab):
                if (not isinstance(e, list) or len(e) != 2 or not isinstance(e[0], str)
                        or isinstance(e[1], bool) or not isinstance(e[1], (int, float))):
                    raise ConvertError(f"tokenizer: Unigram entry {i}")
                self.add_token(e[0], i)
                sc = struct.unpack("<f", struct.pack("<f", float(e[1])))[0]
                self.scores[i] = sc
                self.min_score = min(self.min_score, sc)
            u = _int(model.get("unk_id"))
            if u is not None:
                self.unk = u
            return
        raise ConvertError(f"tokenizer: model {t} (BPE and Unigram are read)")

    def added_tokens(self, added: Any) -> None:
        if not isinstance(added, list):
            return
        for i, a in enumerate(added):
            c = a.get("content") if isinstance(a, dict) else None
            id_ = _int(a.get("id")) if isinstance(a, dict) else None
            if not isinstance(c, str) or not c or id_ is None or not 0 <= id_ < self.vocab_size:
                raise ConvertError(f"tokenizer: added token {i}")
            self.added.append((c.encode(), id_))
            self.set_piece(id_, c, True)
        self.added.sort(key=lambda x: -len(x[0]))  # stable: the file's order among equals

    def build(self, doc: dict[str, Any]) -> bytes:
        model, added = doc.get("model"), doc.get("added_tokens")
        top = -1
        vocab = model.get("vocab") if isinstance(model, dict) else None
        if isinstance(vocab, dict):
            top = max([_int(v) for v in vocab.values() if _int(v) is not None] + [-1])
        elif isinstance(vocab, list):
            top = len(vocab) - 1
        for a in added if isinstance(added, list) else []:
            v = _int(a.get("id")) if isinstance(a, dict) else None
            if v is not None:
                top = max(top, v)
        if top < 0:
            raise ConvertError("tokenizer: no vocabulary")
        self.vocab_size = top + 1
        self.piece: list[bytes | None] = [None] * self.vocab_size
        self.vocab_slots: list[tuple[bytes, int] | None] = [None] * slots(self.vocab_size)
        self.merge_slots: list[tuple[int, int, int, int] | None] = []
        decoder = doc.get("decoder")
        if _str(decoder, "type") == "ByteLevel":
            self.flags |= BYTE_LEVEL
        if decoder is not None and _has_meta(decoder):
            self.flags |= METASPACE_DECODE
        self.normalizer(doc.get("normalizer"))
        self.pre_tokenizer(doc.get("pre_tokenizer"))
        self.added_tokens(added)
        self.model(model)
        return self.pack()

    def pack(self) -> bytes:
        V = self.vocab_size
        unigram = bool(self.flags & UNIGRAM)
        off, at = [], HEADER_WORDS * 4
        sizes = [len(self.vocab_slots) * 12, len(self.merge_slots) * 16, V * 8,
                 V * 4 if unigram else 0, len(self.added) * 12, len(self.pre) * 16,
                 len(self.norm) * 20, 256 * 4]
        for i, n in enumerate(sizes):
            off.append(at if (i != 3 or unigram) else 0)
            at += n
        off.append(at)  # the pool
        pool = bytearray()

        def add(b: bytes) -> int:
            o = len(pool)
            pool.extend(b)
            return o

        piece_at = [NO_PIECE] * V
        pieces = bytearray()
        for i, p in enumerate(self.piece):
            if p is not None:
                piece_at[i] = add(p)
            pieces += struct.pack("<II", piece_at[i], len(p) if p is not None else 0)

        def text_at(b: bytes, id_: int) -> int:
            p = self.piece[id_] if 0 <= id_ < V else None
            return piece_at[id_] if p is not None and p == b else add(b)

        vocab = bytearray()
        for e in self.vocab_slots:
            if e is None:
                vocab += struct.pack("<III", 0, 0, 0xFFFFFFFF)
            else:
                vocab += struct.pack("<IIi", text_at(e[0], e[1]), len(e[0]), e[1])
        added = bytearray()
        for b, id_ in self.added:
            added += struct.pack("<IIi", text_at(b, id_), len(b), id_)
        pre = bytearray()
        for kind, flags, s in self.pre:
            pre += struct.pack("<IIII", kind, flags, add(s), len(s))
        norm = bytearray()
        for kind, f, to in self.norm:
            fo = add(f)
            norm += struct.pack("<IIIII", kind, fo, len(f), add(to), len(to))
        merges = bytearray()
        for e in self.merge_slots:
            merges += struct.pack("<iiii", *(e if e is not None else (-1, 0, 0, 0)))
        scores = struct.pack(f"<{V}f", *self.scores) if unigram and self.scores else b""
        total = at + (len(pool) + 3) // 4 * 4
        head = [0] * HEADER_WORDS
        head[1:13] = [VERSION, self.flags, self.unk & 0xFFFFFFFF, V, self.max_len,
                      struct.unpack("<I", struct.pack("<f", self.min_score))[0],
                      len(self.vocab_slots), len(self.merge_slots), len(self.added),
                      len(self.pre), len(self.norm), len(pool)]
        head[13:22] = off
        head[22] = total
        out = bytearray(b"RTOK" + struct.pack(f"<{HEADER_WORDS - 1}I", *head[1:]))
        out += vocab + merges + pieces + scores + added + pre + norm
        out += struct.pack("<256i", *self.byte_ids)
        out += pool
        out += b"\0" * (total - len(out))
        assert len(out) == total
        return bytes(out)


def table(tokenizer_json: bytes | str | dict[str, Any]) -> bytes:
    """The tokeniser table of a tokenizer.json (its bytes, text or parsed object)."""
    doc = tokenizer_json if isinstance(tokenizer_json, dict) else json.loads(tokenizer_json)
    if not isinstance(doc, dict):
        raise ConvertError("tokenizer: not an object")
    return _Builder().build(doc)
