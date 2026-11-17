#include <stdlib.h>
#include <string.h>

#include "test.h"

int rayo_test_failures;

void test_arena(void);
void test_json(void);
void test_sha256(void);
void test_quantize(void);
void test_kernels(void);
void test_pcwt(void);
void test_plan(void);
void test_golden(void);
void test_memory(void);

unsigned char *test_read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  rewind(f);
  unsigned char *p = n >= 0 ? malloc((size_t)n + 1) : NULL;
  if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
    free(p);
    p = NULL;
  }
  fclose(f);
  if (p) {
    p[n] = 0;
    *size = (size_t)n;
  }
  return p;
}

const char *test_models_dir(void) {
  const char *d = getenv("RAYO_TEST_MODELS");
  return d ? d : "tests/models";
}

static const struct {
  const char *name;
  void (*run)(void);
} tests[] = {
    {"arena", test_arena},       {"json", test_json},       {"sha256", test_sha256},
    {"quantize", test_quantize}, {"kernels", test_kernels}, {"pcwt", test_pcwt},
    {"plan", test_plan},         {"golden", test_golden},   {"memory", test_memory},
};

int main(int argc, char **argv) {
  int ran = 0;
  for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
    if (argc > 1 && strcmp(argv[1], tests[i].name) != 0) continue;
    int before = rayo_test_failures;
    tests[i].run();
    printf("%-10s %s\n", tests[i].name, rayo_test_failures == before ? "ok" : "FAILED");
    ran++;
  }
  printf("%d tests, %d failures\n", ran, rayo_test_failures);
  return rayo_test_failures ? 1 : 0;
}
