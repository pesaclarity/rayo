#include <stdlib.h>
#include <string.h>

#include "session.h"
#include "test.h"

int rayo_test_failures;

void test_arena(void);
void test_json(void);
void test_format(void);
void test_sha256(void);
void test_erf(void);
void test_quantize(void);
void test_kernels(void);
void test_pcwt(void);
void test_plan(void);
void test_golden(void);
void test_memory(void);
void test_decoder(void);
void test_ask(void);

unsigned char *test_read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  long n;
  unsigned char *p;
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  n = ftell(f);
  rewind(f);
  p = n >= 0 ? (unsigned char *)malloc((size_t)n + 1) : NULL;
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

unsigned char *test_model_file(const char *name, size_t *size) {
  const char *dir = getenv("RAYO_TEST_MODELS");
  char path[512];
  unsigned char *p;
  if (!dir) dir = "tests/models";
  if (strlen(dir) + strlen(name) + 2 > sizeof path) return NULL;
  strcpy(path, dir);
  strcat(path, "/");
  strcat(path, name);
  p = test_read_file(path, size);
  if (!p) fprintf(stderr, "cannot read %s\n", path);
  return p;
}

#if !defined(RAYO_TEST_COUNT_ALLOCS)
long test_allocations(void) {
  return -1;
}
#endif

void test_choose_kernels(rayo_session *session) {
  const char *k = getenv("RAYO_TEST_KERNELS");
  if (k && strcmp(k, "generic") == 0) rayo_kernels_generic(&session->kernels);
  if (k && strcmp(k, "simd") == 0) rayo_kernels_simd(&session->kernels);
}

typedef struct test_case {
  const char *name;
  void (*run)(void);
} test_case;

static const test_case tests[] = {
    {"arena", test_arena},     {"json", test_json},     {"format", test_format},
    {"sha256", test_sha256},   {"erf", test_erf},       {"quantize", test_quantize},
    {"kernels", test_kernels}, {"pcwt", test_pcwt},     {"plan", test_plan},
    {"golden", test_golden},   {"memory", test_memory}, {"decoder", test_decoder},
    {"ask", test_ask}};

int main(int argc, char **argv) {
  int ran = 0;
  size_t i;
  for (i = 0; i < sizeof tests / sizeof tests[0]; i++) {
    int before = rayo_test_failures;
    if (argc > 1 && strcmp(argv[1], tests[i].name) != 0) continue;
    tests[i].run();
    printf("%-10s %s\n", tests[i].name, rayo_test_failures == before ? "ok" : "FAILED");
    ran++;
  }
  printf("%d tests, %d failures\n", ran, rayo_test_failures);
  return rayo_test_failures ? 1 : 0;
}
