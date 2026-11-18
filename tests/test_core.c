/* core: arena, JSON, formatting, maths; and SHA-256. */
#include <math.h>
#include <stdarg.h>
#include <string.h>

#include "core/arena.h"
#include "core/format.h"
#include "core/json.h"
#include "core/mathx.h"
#include "test.h"

void test_arena(void) {
  rayo_arena a;
  char *s;
  unsigned char *big;
  a.head = NULL;
  a.bytes = 0;
  s = rayo_arena_strndup(&a, "hello", 5);
  EXPECT(s && strcmp(s, "hello") == 0);
  big = (unsigned char *)rayo_arena_alloc(&a, 100000);
  EXPECT(big && big[99999] == 0);
  EXPECT((size_t)rayo_arena_alloc(&a, 3) % 16 == 0);
  EXPECT((size_t)rayo_arena_alloc_aligned(&a, 10) % 64 == 0);
  EXPECT(rayo_arena_array(&a, ((size_t)-1) / 2, 4) == NULL);
  rayo_arena_free(&a);
  rayo_arena_free(&a); /* twice is fine */
  EXPECT(a.head == NULL && a.bytes == 0);
}

static rayo_status parse(rayo_arena *a, const char *text, rayo_json **out) {
  return rayo_json_parse(text, strlen(text), (size_t)-1, a, out, NULL);
}

void test_json(void) {
  static const char *const bad[] = {"",    "{",        "[1,]",    "{\"a\" 1}",   "01",
                                    "1.",  "\"\x01\"", "\"\\x\"", "\"\\ud800\"", "[1] x",
                                    "nul", "\"abc",    "-",       "1e",          "1e999"};
  rayo_arena a;
  rayo_json *v;
  const rayo_json *arr;
  rayo_i32 i;
  size_t k;
  char deep[200];
  a.head = NULL;
  a.bytes = 0;
  EXPECT(parse(&a, " {\"a\": [1, -2.5e-1, true, null], \"b\": \"x\\u00e9\\n\\ud83d\\ude00\"} ",
               &v) == RAYO_OK);
  arr = rayo_json_get(v, "a");
  EXPECT(arr && arr->kind == RAYO_JSON_ARRAY && arr->length == 4);
  EXPECT(rayo_json_as_int(arr->items[0], &i) && i == 1);
  EXPECT(!rayo_json_as_int(arr->items[1], &i) && arr->items[1]->number == -0.25);
  EXPECT(arr->items[2]->kind == RAYO_JSON_BOOL && arr->items[2]->boolean);
  EXPECT(strcmp(rayo_json_as_string(rayo_json_get(v, "b")), "x\xc3\xa9\n\xf0\x9f\x98\x80") == 0);
  EXPECT(rayo_json_get(v, "missing") == NULL);
  EXPECT(rayo_json_get(v, "a")->raw_length == strlen("[1, -2.5e-1, true, null]"));
  EXPECT(parse(&a, "2147483648", &v) == RAYO_OK && !rayo_json_as_int(v, &i));
  for (k = 0; k < sizeof bad / sizeof bad[0]; k++) EXPECT(parse(&a, bad[k], &v) == RAYO_ERR_FORMAT);
  memset(deep, '[', 100);
  memset(deep + 100, ']', 100);
  EXPECT(rayo_json_parse(deep, 200, (size_t)-1, &a, &v, NULL) == RAYO_ERR_FORMAT);
  rayo_arena_free(&a);
}

static void format(char *out, size_t size, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  rayo_vformat(out, size, fmt, ap);
  va_end(ap);
}

void test_format(void) {
  char buf[64];
  format(buf, sizeof buf, "%s=%d, %ld, %u, %lu, %c%%", "x", -7, -2147483647l, 42u, 4294967295ul,
         'z');
  EXPECT(strcmp(buf, "x=-7, -2147483647, 42, 4294967295, z%") == 0);
  format(buf, 6, "%s", "truncated");
  EXPECT(strcmp(buf, "trunc") == 0);
}

/* erf at points away from where the polynomials meet, against erf to 17
 * digits (mpmath); and exp and sqrt against the C library. */
void test_erf(void) {
  static const double x[] = {0.0, 1e-8, 0.1, 0.5, 0.99, 1.0, 1.3, 1.75, 2.2, 2.9, 3.5, 3.99, 4.5};
  /* erf of each x rounded to float. */
  static const double want[] = {0.0,
                                1.1283791602378209e-08,
                                0.11246291768297051,
                                0.5204998778130465,
                                0.838508073593712,
                                0.8427007929497149,
                                0.9340079350125213,
                                0.9866716712191824,
                                0.9981371541274596,
                                0.9999589021458579,
                                0.9999992569016276,
                                0.99999998326079,
                                0.9999999998033839};
  size_t i;
  for (i = 0; i < sizeof x / sizeof x[0]; i++) {
    const float xf = (float)x[i];
    const double w = want[i];
    EXPECT(fabs((double)rayo_erff(xf) - w) <= 1e-7 * (w > 1e-30 ? w : 1.0) + 1e-37);
    EXPECT(rayo_erff(-xf) == -rayo_erff(xf));
  }
  EXPECT(rayo_expf(0.0f) == 1.0f);
  EXPECT(fabs((double)rayo_expf(-1.5f) - exp(-1.5)) < 1e-7);
  EXPECT(rayo_sqrtf(2.25f) == 1.5f);
}

static void hex(const rayo_u8 d[32], char out[65]) {
  static const char digits[] = "0123456789abcdef";
  int i;
  for (i = 0; i < 32; i++) {
    out[2 * i] = digits[d[i] >> 4];
    out[2 * i + 1] = digits[d[i] & 15];
  }
  out[64] = '\0';
}

void test_sha256(void) {
  rayo_u8 d[32];
  char h[65];
  const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; /* 56 bytes */
  rayo_sha256("", 0, d);
  hex(d, h);
  EXPECT(strcmp(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0);
  rayo_sha256("abc", 3, d);
  hex(d, h);
  EXPECT(strcmp(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);
  rayo_sha256(m, strlen(m), d);
  hex(d, h);
  EXPECT(strcmp(h, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0);
}
