/* A tokeniser table (docs/FORMAT.md), as rayo_tokenizer_create_table and
 * rayo_tokenizer_from_model read one from a model file, then texts encoded
 * with it and its pieces and lookups used. */
#include <stdlib.h>
#include <string.h>

#include "fuzz.h"
#include "rayo/text.h"

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
  static const char *const texts[] = {"Hello there, friends! How are you today?",
                                      "<|im_start|>user\nab ab<|im_end|>",
                                      "numbers 1234567 and 42, code: def f(x): return x**2",
                                      "caf\xc3\xa9 \xe2\x96\x81 \xf0\x9f\x98\x80",
                                      "\xff\xc0\x80\xed\xa0\x80 \t\r\n  ",
                                      ""};
  rayo_tokenizer *t = NULL;
  unsigned char *copy;
  size_t i;
  /* The table is read where it is, 4-byte aligned: malloc's memory is. */
  copy = (unsigned char *)malloc(size ? size : 1);
  if (!copy) return 0;
  if (size) memcpy(copy, data, size);
  if (rayo_tokenizer_create_table(copy, size, &t, NULL) == RAYO_OK) {
    rayo_i32 id, n = rayo_tokenizer_vocab_size(t);
    for (i = 0; i < sizeof texts / sizeof texts[0]; i++) {
      rayo_fuzz_encode(t, texts[i], strlen(texts[i]));
    }
    for (id = -1; id <= n && id < 4096; id++) {
      size_t len;
      rayo_tokenizer_piece(t, id, &len);
    }
    rayo_tokenizer_find(t, "<|im_end|>");
    rayo_tokenizer_find(t, "a");
    rayo_tokenizer_destroy(t);
  }
  free(copy);
  return 0;
}
