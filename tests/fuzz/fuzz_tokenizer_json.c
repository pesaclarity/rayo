/* A tokenizer.json, as rayo_tokenizer_create reads one: the JSON parser,
 * the model, merges, added tokens, normalisers and pre-tokenisers (their
 * patterns compiled), the table built and checked, then texts encoded. */
#include <string.h>

#include "fuzz.h"
#include "rayo/text.h"

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
  static const char *const texts[] = {"Hello there, friends! <|im_end|> 1234 ab ab AB a.b 12",
                                      "caf\xc3\xa9 \xe2\x96\x81\xe2\x96\x81 \xff\xfe \n\n  x"};
  rayo_tokenizer *t = NULL;
  size_t i;
  if (rayo_tokenizer_create((const char *)data, size, &t, NULL) != RAYO_OK) return 0;
  for (i = 0; i < sizeof texts / sizeof texts[0]; i++) {
    rayo_fuzz_encode(t, texts[i], strlen(texts[i]));
  }
  rayo_tokenizer_destroy(t);
  return 0;
}
