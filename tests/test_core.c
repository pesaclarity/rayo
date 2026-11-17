/* core/arena, core/json and sha256. */
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "rayo/sha256.h"
#include "test.h"

void test_arena(void) {
  rayo_arena a = {0};
  char *s = rayo_arena_strndup(&a, "hello", 5);
  EXPECT(s && strcmp(s, "hello") == 0);
  unsigned char *big = rayo_arena_alloc(&a, 100000);
  EXPECT(big && big[99999] == 0);
  EXPECT(((uintptr_t)rayo_arena_alloc(&a, 3) & 15) == 0);
  EXPECT(((uintptr_t)rayo_arena_alloc_aligned(&a, 10) & 63) == 0);
  EXPECT(rayo_arena_array(&a, SIZE_MAX / 2, 4) == NULL);
  rayo_arena_free(&a);
  rayo_arena_free(&a); /* twice is fine */
  EXPECT(a.head == NULL && a.bytes == 0);
}

static rayo_status parse(rayo_arena *a, const char *text, rayo_json **out) {
  return rayo_json_parse(text, strlen(text), a, out, NULL);
}

void test_json(void) {
  rayo_arena a = {0};
  rayo_json *v;
  EXPECT(parse(&a, " {\"a\": [1, -2.5e-1, true, null], \"b\": \"x\\u00e9\\n\\ud83d\\ude00\"} ",
               &v) == RAYO_OK);
  const rayo_json *arr = rayo_json_get(v, "a");
  int64_t i;
  EXPECT(arr && arr->kind == RAYO_JSON_ARRAY && arr->length == 4);
  EXPECT(rayo_json_as_int(arr->items[0], &i) && i == 1);
  EXPECT(!rayo_json_as_int(arr->items[1], &i) && arr->items[1]->number == -0.25);
  EXPECT(arr->items[2]->kind == RAYO_JSON_BOOL && arr->items[2]->boolean);
  EXPECT(strcmp(rayo_json_as_string(rayo_json_get(v, "b")), "x\xc3\xa9\n\xf0\x9f\x98\x80") == 0);
  EXPECT(rayo_json_get(v, "missing") == NULL);
  EXPECT(rayo_json_get(v, "a")->raw_length == strlen("[1, -2.5e-1, true, null]"));
  const char *bad[] = {"",        "{",           "[1,]",  "{\"a\" 1}", "01",    "1.", "\"\x01\"",
                       "\"\\x\"", "\"\\ud800\"", "[1] x", "nul",       "\"abc", "-",  "1e"};
  for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++)
    EXPECT(parse(&a, bad[k], &v) == RAYO_ERR_FORMAT);
  char deep[200];
  memset(deep, '[', 100);
  memset(deep + 100, ']', 100);
  EXPECT(rayo_json_parse(deep, 200, &a, &v, NULL) == RAYO_ERR_FORMAT);
  rayo_arena_free(&a);
}

static void hex(const uint8_t d[32], char out[65]) {
  for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

void test_sha256(void) {
  uint8_t d[32];
  char h[65];
  rayo_sha256("", 0, d);
  hex(d, h);
  EXPECT(strcmp(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0);
  rayo_sha256("abc", 3, d);
  hex(d, h);
  EXPECT(strcmp(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);
  const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; /* 56 bytes */
  rayo_sha256(m, strlen(m), d);
  hex(d, h);
  EXPECT(strcmp(h, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0);
}
