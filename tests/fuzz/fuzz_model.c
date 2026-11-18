/* A model file, as rayo_model_create reads one from untrusted bytes: the
 * PCWT header, the JSON parser, the tensors, the graph and every op's
 * attributes, then a session planned, bound and run with small sizes, and
 * the tokeniser's table if the file has one. Seeds: tests/models. */
#include "fuzz.h"

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
  rayo_fuzz_model(data, size);
  return 0;
}
