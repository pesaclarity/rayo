/* Arbitrary bytes as text (invalid UTF-8, NULs, long words, added tokens'
 * fragments) through the tokenisers of tests/fuzz/tokenizers: byte-level
 * BPE, SentencePiece-style BPE with byte fallback, Unigram with Metaspace,
 * Split patterns and digits. The first byte picks the tokeniser. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuzz.h"
#include "rayo/text.h"

#define MAX_TOKENIZERS 8

static rayo_tokenizer *tokenizers[MAX_TOKENIZERS];
static int tokenizer_count;

static void load(const char *dir, const char *name) {
  char path[1024];
  unsigned char *json;
  size_t size;
  rayo_message msg;
  if (tokenizer_count == MAX_TOKENIZERS || strlen(dir) + strlen(name) + 2 > sizeof path) return;
  sprintf(path, "%s/%s", dir, name);
  json = rayo_fuzz_read(path, &size);
  if (!json) {
    fprintf(stderr, "fuzz_tokenize: cannot read %s (set RAYO_FUZZ_DIR)\n", path);
    abort();
  }
  if (rayo_tokenizer_create((const char *)json, size, &tokenizers[tokenizer_count], &msg) !=
      RAYO_OK) {
    fprintf(stderr, "fuzz_tokenize: %s: %s\n", path, msg.text);
    abort();
  }
  tokenizer_count++;
  free(json);
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  const char *dir = getenv("RAYO_FUZZ_DIR");
  (void)argc;
  (void)argv;
  if (!dir) dir = "tests/fuzz";
  {
    char sub[900];
    if (strlen(dir) + 12 > sizeof sub) abort();
    sprintf(sub, "%s/tokenizers", dir);
    load(sub, "bytelevel.json");
    load(sub, "sentencepiece.json");
    load(sub, "unigram.json");
    load(sub, "split.json");
    load(sub, "llama3.json");
  }
  return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
  if (tokenizer_count == 0) LLVMFuzzerInitialize(NULL, NULL);
  if (size == 0) return 0;
  rayo_fuzz_encode(tokenizers[data[0] % tokenizer_count], (const char *)data + 1, size - 1);
  return 0;
}
