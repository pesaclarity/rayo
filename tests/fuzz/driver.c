/* Runs a fuzz target without libFuzzer: on each file named (a corpus
 * replayed as a test, with any C89 compiler and sanitizer), or on standard
 * input when none is (AFL: afl-clang-fast ... driver.c, afl-fuzz -- ./target). */
#include <stdio.h>
#include <stdlib.h>

#include "fuzz.h"

int main(int argc, char **argv) {
  int i;
  size_t size;
  unsigned char *data;
  LLVMFuzzerInitialize(&argc, &argv);
  if (argc < 2) {
    data = rayo_fuzz_read("/dev/stdin", &size);
    if (!data) return 1;
    LLVMFuzzerTestOneInput(data, size);
    free(data);
    return 0;
  }
  for (i = 1; i < argc; i++) {
    data = rayo_fuzz_read(argv[i], &size);
    if (!data) {
      fprintf(stderr, "cannot read %s\n", argv[i]);
      return 1;
    }
    LLVMFuzzerTestOneInput(data, size);
    free(data);
  }
  printf("%d inputs\n", argc - 1);
  return 0;
}
