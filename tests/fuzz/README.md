# Fuzzing

libFuzzer targets for what rayo reads from untrusted sources (docs/SECURITY.md), each a
`LLVMFuzzerTestOneInput` in C89:

| Target | Input | What it exercises |
| --- | --- | --- |
| `model` | a model file | the PCWT header, the JSON parser, the tensors, the graph and the ops' attributes, a session planned, bound and run with small sizes (float32 and int8 caches), the tokeniser's table in the file |
| `graph` | a model's JSON header | the same, with the data made as long as the header's tensors need (filled with 0.25f), so mutations reach the graph instead of the offsets |
| `tokenizer_table` | a tokeniser table (docs/FORMAT.md) | its checks, then texts encoded, every piece, lookups |
| `tokenizer_json` | a tokenizer.json | the parser, the model, merges, added tokens, normalisers and patterns, the table built, texts encoded |
| `tokenize` | text (the first byte picks one of `tokenizers/`) | byte-level BPE, SentencePiece-style BPE with byte fallback, Unigram with Metaspace, Split and Digits, Llama 3's pattern |

`corpus/<target>` holds the seeds (small: the test models, their headers, the tokenisers);
`*.dict` are dictionaries of the formats' keywords. `tests/malformed` (tool/make_malformed.py)
holds one file for each way a file can be wrong, with the status rayo must give.

```sh
tool/ci/fuzz.sh run 600                # each target 10 minutes (clang, ASan, UBSan)
tool/ci/fuzz.sh run 60 tokenize        # one target
tool/ci/fuzz.sh replay clang -fsanitize=address,undefined   # every seed and malformed file
FUZZ_OUT=out tool/ci/fuzz.sh run 600   # keep the builds and the new inputs in out/
tool/ci/fuzz.sh afl out                # the targets for AFL++ (afl-clang-fast), on stdin:
afl-fuzz -i tests/fuzz/corpus/model -o findings -- out/fuzz_model
```

A failing input is written to the current directory (`crash-*`, `timeout-*`, `oom-*`); run it
again with `out/fuzz_<target> crash-...`. Add it to `tool/make_malformed.py` (or to the
target's corpus) with the fix, so it stays tested. `fuzz_tokenize` reads `tokenizers/` from
`RAYO_FUZZ_DIR` (default `tests/fuzz`).
