/* Untrusted input (docs/SECURITY.md): the malformed corpus of
 * tests/malformed (tool/make_malformed.py) gives the statuses it lists,
 * the limits of rayo/config.h hold, the tokeniser takes any bytes and long
 * words, and models are verified by SHA-256 and Ed25519 (RFC 8032). */
#include <stdlib.h>
#include <string.h>

#include "core/arena.h"
#include "core/json.h"
#include "crypto/ed25519.h"
#include "crypto/sha512.h"
#include "rayo/verify.h"
#include "session.h"
#include "test.h"

static rayo_status status_of(const char *name) {
  static const char *const names[] = {"OK",          "INVALID_ARGUMENT", "FORMAT",
                                      "UNSUPPORTED", "OUT_OF_MEMORY",    "SHAPE",
                                      "LIMIT",       "INTEGRITY"};
  int i;
  for (i = 0; i < 8; i++) {
    if (strcmp(names[i], name) == 0) return (rayo_status)i;
  }
  return (rayo_status)-1;
}

static unsigned char *malformed_file(const char *name, size_t *size) {
  char path[512];
  const char *dir = getenv("RAYO_TEST_MALFORMED");
  if (!dir) dir = "tests/malformed";
  if (strlen(dir) + strlen(name) + 2 > sizeof path) return NULL;
  strcpy(path, dir);
  strcat(path, "/");
  strcat(path, name);
  return test_read_file(path, size);
}

/* What one file of the corpus gives at its stage. */
static rayo_status try_model(const unsigned char *data, size_t size, const char *stage) {
  rayo_model *m = NULL;
  rayo_session *s = NULL;
  rayo_status st = rayo_model_create(data, size, &m, NULL);
  if (st != RAYO_OK || strcmp(stage, "load") == 0) {
    rayo_model_destroy(m);
    return st;
  }
  st = rayo_session_create(m, &s, NULL);
  if (st == RAYO_OK && strcmp(stage, "run") == 0) {
    rayo_tensor_view in;
    st = rayo_session_bind(s, "T", 2, NULL);
    if (st == RAYO_OK) st = rayo_session_input(s, 0, &in);
    if (st == RAYO_OK) {
      ((rayo_i32 *)in.data)[0] = 3;
      ((rayo_i32 *)in.data)[1] = 0;
      st = rayo_session_run(s, NULL);
    }
  }
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  return st;
}

static rayo_status try_table(const unsigned char *data, size_t size) {
  rayo_tokenizer *t = NULL;
  unsigned char *copy = malloc(size ? size : 1); /* malloc's memory is 4-byte aligned */
  rayo_status st;
  if (!copy) return RAYO_ERR_OUT_OF_MEMORY;
  memcpy(copy, data, size);
  st = rayo_tokenizer_create_table(copy, size, &t, NULL);
  rayo_tokenizer_destroy(t);
  free(copy);
  return st;
}

static void malformed_corpus(void) {
  size_t size, at = 0, n = 0;
  char *list = (char *)malformed_file("expected.txt", &size);
  EXPECT(list != NULL);
  if (!list) return;
  while (at < size) {
    char line[256], path[200], stage[16], want[24];
    size_t len = 0;
    while (at < size && list[at] != '\n' && len + 1 < sizeof line) line[len++] = list[at++];
    line[len] = '\0';
    at++;
    if (line[0] == '#' || len == 0) continue;
    if (sscanf(line, "%199s %15s %23s", path, stage, want) == 3) {
      size_t fsize;
      unsigned char *data = malformed_file(path, &fsize);
      rayo_status got = (rayo_status)-1;
      EXPECT(data != NULL);
      if (!data) continue;
      if (strncmp(path, "model/", 6) == 0) {
        got = try_model(data, fsize, stage);
      } else if (strncmp(path, "tokenizer_table/", 16) == 0) {
        got = try_table(data, fsize);
      } else if (strncmp(path, "tokenizer_json/", 15) == 0) {
        rayo_tokenizer *t = NULL;
        got = rayo_tokenizer_create((const char *)data, fsize, &t, NULL);
        rayo_tokenizer_destroy(t);
      }
      if (got != status_of(want)) {
        fprintf(stderr, "  %s: %s, expected %s\n", path, rayo_status_str(got), want);
      }
      EXPECT(got == status_of(want));
      free(data);
      n++;
    }
  }
  EXPECT(n >= 50);
  free(list);
}

static rayo_tokenizer *malformed_tokenizer(const char *name) {
  size_t size;
  unsigned char *json = malformed_file(name, &size);
  rayo_tokenizer *t = NULL;
  if (json) rayo_tokenizer_create((const char *)json, size, &t, NULL);
  free(json);
  EXPECT(t != NULL);
  return t;
}

/* Patterns that backtrack polynomially, and normalisers that multiply the
 * text, end in RAYO_ERR_LIMIT, not in time or memory without bound. */
static void tokenizer_limits(void) {
  rayo_tokenizer *redos = malformed_tokenizer("tokenizer_json/redos-pattern.json");
  rayo_tokenizer *blowup = malformed_tokenizer("tokenizer_json/normalizer-blowup.json");
  char *text = malloc(4096);
  rayo_i32 ids[16];
  size_t n;
  if (text && redos && blowup) {
    memset(text, ' ', 4096);
    EXPECT(rayo_tokenizer_encode(redos, text, 4096, ids, 16, &n, NULL) == RAYO_ERR_LIMIT);
    EXPECT(rayo_tokenizer_encode(redos, "a x", 3, ids, 16, &n, NULL) == RAYO_OK);
    /* 64^4 times 300 bytes: over RAYO_MAX_TEXT_BYTES. */
    EXPECT(rayo_tokenizer_encode(blowup, text, 300, ids, 16, &n, NULL) == RAYO_ERR_LIMIT);
    EXPECT(rayo_tokenizer_encode(blowup, "ab", 2, ids, 16, &n, NULL) == RAYO_OK);
  }
  rayo_tokenizer_destroy(redos);
  rayo_tokenizer_destroy(blowup);
  free(text);
}

/* Text longer than the first working buffer (a regression: the byte-level
 * pre-tokeniser copied bytes of a buffer into itself while growing it, and
 * read freed memory), invalid UTF-8, and a long word, which the BPE model
 * merges in O(n log n). The pieces join back to the text. */
static void tokenizer_text(void) {
  char path[512];
  const char *dir = getenv("RAYO_TEST_MODELS");
  rayo_tokenizer *t = NULL;
  const size_t big = 300000;
  char *text = malloc(big);
  rayo_i32 *ids = malloc(big * sizeof *ids);
  size_t n = 0, i, len = 0, k;
  rayo_bool same = RAYO_TRUE;
  if (!dir) dir = "tests/models";
  sprintf(path, "%.400s/llama.tokenizer.json", dir);
  EXPECT(rayo_tokenizer_load(path, &t, NULL) == RAYO_OK);
  if (!t || !text || !ids) {
    rayo_tokenizer_destroy(t);
    free(text);
    free(ids);
    return;
  }
  for (k = 0; k < 2; k++) {
    const size_t size = k == 0 ? 5000 : big;
    for (i = 0; i < size; i++) {
      /* k = 0: words, spaces and stray bytes; k = 1: one long word. */
      text[i] = k == 1 ? 'a' : (char)(i % 11 == 0 ? ' ' : i % 97 == 5 ? 0xff : 'a' + (int)(i % 7));
    }
    EXPECT(rayo_tokenizer_encode(t, text, size, ids, big, &n, NULL) == RAYO_OK);
    len = 0;
    for (i = 0; i < n && n <= big; i++) {
      size_t piece_len;
      const char *p = rayo_tokenizer_piece(t, ids[i], &piece_len);
      if (!p || len + piece_len > size || memcmp(text + len, p, piece_len) != 0) {
        same = RAYO_FALSE;
        break;
      }
      len += piece_len;
    }
    EXPECT(same && len == size);
  }
  /* RAYO_MAX_TEXT_BYTES is checked before anything is read. */
  EXPECT(rayo_tokenizer_encode(t, text, (size_t)RAYO_MAX_TEXT_BYTES + 1, ids, 0, &n, NULL) ==
         RAYO_ERR_LIMIT);
  rayo_tokenizer_destroy(t);
  free(text);
  free(ids);
}

/* The JSON parser: strings are UTF-8, and values are counted. */
static void json_limits(void) {
  rayo_arena a;
  rayo_json *v;
  memset(&a, 0, sizeof a);
  EXPECT(rayo_json_parse("[1,2,3]", 7, 4, &a, &v, NULL) == RAYO_OK);
  EXPECT(rayo_json_parse("[1,2,3]", 7, 3, &a, &v, NULL) == RAYO_ERR_LIMIT);
  EXPECT(rayo_json_parse("\"caf\xc3\xa9\"", 7, 8, &a, &v, NULL) == RAYO_OK);
  EXPECT(rayo_json_parse("\"caf\xc3\"", 6, 8, &a, &v, NULL) == RAYO_ERR_FORMAT);
  EXPECT(rayo_json_parse("\"\xed\xa0\x80\"", 5, 8, &a, &v, NULL) == RAYO_ERR_FORMAT);
  EXPECT(rayo_json_parse("\"\xc0\xaf\"", 4, 8, &a, &v, NULL) == RAYO_ERR_FORMAT);
  EXPECT(rayo_json_parse("\"\xf4\x90\x80\x80\"", 6, 8, &a, &v, NULL) == RAYO_ERR_FORMAT);
  rayo_arena_free(&a);
  EXPECT(strcmp(rayo_status_str(RAYO_ERR_LIMIT), "over a limit") == 0);
  EXPECT(strcmp(rayo_status_str(RAYO_ERR_INTEGRITY), "integrity check failed") == 0);
}

/* A mapped model must be a regular file. */
static void platform_files(void) {
  rayo_model *m = NULL;
  EXPECT(rayo_model_load("tests", NULL, &m, NULL) != RAYO_OK && m == NULL);
#if defined(__unix__) || defined(__APPLE__)
  EXPECT(rayo_model_load("/dev/null", NULL, &m, NULL) == RAYO_ERR_FORMAT && m == NULL);
#endif
}

static int hex_value(char c) {
  return c <= '9' ? c - '0' : c - 'a' + 10;
}

static void unhex(const char *s, rayo_u8 *out) {
  size_t i;
  for (i = 0; s[2 * i]; i++) out[i] = (rayo_u8)(hex_value(s[2 * i]) * 16 + hex_value(s[2 * i + 1]));
}

static rayo_bool hex_is(const rayo_u8 *d, size_t n, const char *want) {
  rayo_u8 w[64];
  unhex(want, w);
  return strlen(want) == 2 * n && memcmp(d, w, n) == 0;
}

/* FIPS 180-4's examples, and the lengths around a block. */
static void sha512_vectors(void) {
  static const char abc448[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  rayo_sha512 s;
  rayo_u8 d[64], buf[1000];
  int i;
  rayo_sha512_init(&s);
  rayo_sha512_final(&s, d);
  EXPECT(hex_is(d, 64,
                "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0"
                "ff8318d2877eec2f63b931bd47417a81a538327af927da3e"));
  rayo_sha512_init(&s);
  rayo_sha512_update(&s, "abc", 3);
  rayo_sha512_final(&s, d);
  EXPECT(hex_is(d, 64,
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a8"
                "36ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"));
  rayo_sha512_init(&s);
  rayo_sha512_update(&s, abc448, sizeof abc448 - 1);
  rayo_sha512_final(&s, d);
  EXPECT(hex_is(d, 64,
                "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9"
                "aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445"));
  /* A million a's, a thousand at a time. */
  memset(buf, 'a', sizeof buf);
  rayo_sha512_init(&s);
  for (i = 0; i < 1000; i++) rayo_sha512_update(&s, buf, sizeof buf);
  rayo_sha512_final(&s, d);
  EXPECT(hex_is(d, 64,
                "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a"
                "4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"));
}

/* RFC 8032's tests 1 to 3, a message of 1023 bytes signed with
 * Python's cryptography, and what must be refused. */
static void ed25519_vectors(void) {
  static const char *const cases[][3] = {
      {"d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
       "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e3970"
       "1cf9b46bd25bf5f0595bbe24655141438e7a100b"},
      {"3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
       "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613"
       "d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"},
      {"fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
       "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760"
       "984dc6594a7c15e9716ed28dc027beceea1ec40a"}};
  rayo_u8 pk[32], sig[64], msg[2];
  size_t i;
  for (i = 0; i < 3; i++) {
    const size_t n = strlen(cases[i][1]) / 2;
    unhex(cases[i][0], pk);
    unhex(cases[i][1], msg);
    unhex(cases[i][2], sig);
    EXPECT(rayo_ed25519_verify(sig, pk, msg, n));
    /* A changed signature, message or key. */
    sig[5] ^= 1;
    EXPECT(!rayo_ed25519_verify(sig, pk, msg, n));
    sig[5] ^= 1;
    sig[40] ^= 0x10;
    EXPECT(!rayo_ed25519_verify(sig, pk, msg, n));
    sig[40] ^= 0x10;
    if (n) {
      msg[0] ^= 0x80;
      EXPECT(!rayo_ed25519_verify(sig, pk, msg, n));
      msg[0] ^= 0x80;
    }
    pk[0] ^= 2;
    EXPECT(!rayo_ed25519_verify(sig, pk, msg, n));
    pk[0] ^= 2;
  }
  /* S + L, the same signature in another form: refused (S < L). Test 1's
   * S + L is below 2^256. */
  unhex(cases[0][0], pk);
  unhex(cases[0][2], sig);
  {
    static const rayo_u8 order[32] = {0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
                                      0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
                                      0,    0,    0,    0,    0,    0,    0,    0,
                                      0,    0,    0,    0,    0,    0,    0,    0x10};
    unsigned c = 0;
    int k;
    for (k = 0; k < 32; k++) {
      c += (unsigned)sig[32 + k] + order[k];
      sig[32 + k] = (rayo_u8)(c & 255u);
      c >>= 8;
    }
    EXPECT(c == 0 && !rayo_ed25519_verify(sig, pk, "", 0));
  }
  /* A key that is not a point's canonical encoding: y = p. */
  memset(pk, 0xff, 32);
  pk[0] = 0xed;
  pk[31] = 0x7f;
  unhex(cases[0][2], sig);
  EXPECT(!rayo_ed25519_verify(sig, pk, "", 0));
}

/* rayo_verify and the model options' integrity, on a real model. */
static void model_integrity(void) {
  /* reader.weights, its SHA-256, and its Ed25519 signature by RFC 8032
   * test 1's key (written with tool/sign_model.py). */
  size_t size;
  unsigned char *file = test_model_file("reader.weights", &size);
  rayo_u8 digest[32], pk[32], sig[64];
  rayo_integrity want;
  rayo_model_options o;
  rayo_model *m = NULL;
  rayo_message msg;
  EXPECT(file != NULL);
  if (!file) return;
  rayo_sha256(file, size, digest);
  unhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", pk);
  memset(&want, 0, sizeof want);
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_ERR_INVALID_ARGUMENT);
  want.sha256 = digest;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_OK);
  memset(&o, 0, sizeof o);
  o.integrity = &want;
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_OK && m != NULL);
  rayo_model_destroy(m);
  file[size - 1] ^= 1;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_ERR_INTEGRITY);
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_ERR_INTEGRITY && m == NULL);
  file[size - 1] ^= 1;
  /* A key without a signature; the file's signature, with its SHA-256 and
   * without; a changed byte; the wrong signature. */
  want.public_key = pk;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_ERR_INVALID_ARGUMENT);
  unhex(
      "932b2f6915239118d8edb3e10047d8f4e6eb84fa233a7fdb68842d246cb2079bf9385856642a59144177adbe3a0"
      "b59a8060b41881718331a01c8e86427123201",
      sig);
  want.signature = sig;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_OK);
  want.sha256 = NULL;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_OK);
  EXPECT(rayo_model_create_with(file, size, &o, &m, &msg) == RAYO_OK && m != NULL);
  rayo_model_destroy(m);
  file[100] ^= 4;
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_ERR_INTEGRITY);
  file[100] ^= 4;
  want.sha256 = digest;
  memset(sig, 0, sizeof sig);
  EXPECT(rayo_verify(file, size, &want, &msg) == RAYO_ERR_INTEGRITY);
  {
    char path[512];
    const char *dir = getenv("RAYO_TEST_MODELS");
    if (!dir) dir = "tests/models";
    sprintf(path, "%.400s/reader.weights", dir);
    EXPECT(rayo_verify_file(path, &want, &msg) == RAYO_ERR_INTEGRITY);
    EXPECT(rayo_model_load(path, &o, &m, &msg) == RAYO_ERR_INTEGRITY && m == NULL);
    want.public_key = want.signature = NULL;
    EXPECT(rayo_verify_file(path, &want, &msg) == RAYO_OK);
    EXPECT(rayo_model_load(path, &o, &m, &msg) == RAYO_OK && m != NULL);
    rayo_model_destroy(m);
  }
  free(file);
}

/* A graph's names: a node's output shadows an input of the same name for
 * the nodes after it, as before names were hashed. */
static void graph_names(void) {
  static const char header[] =
      "{\"format_version\":1,\"model\":\"names\",\"tensors\":[{\"name\":\"x\",\"dtype\":"
      "\"float32\",\"shape\":[2],\"offset\":0,\"bytes\":8}],\"graph\":{\"version\":1,"
      "\"inputs\":[{\"name\":\"x\",\"dtype\":\"float32\",\"shape\":[2]}],\"nodes\":["
      "{\"op\":\"add\",\"inputs\":[\"x\",\"x\"],\"output\":\"x\"},"
      "{\"op\":\"add\",\"inputs\":[\"x\",\"x\"],\"output\":\"y\"}],"
      "\"outputs\":[{\"name\":\"y\",\"node\":\"y\"}]}}";
  const size_t h = (sizeof header - 1 + 3) / 4 * 4;
  unsigned char *file = calloc(1, 12 + h + 8);
  rayo_model *m = NULL;
  rayo_session *s = NULL;
  rayo_tensor_view in, out;
  rayo_message msg;
  if (!file) return;
  memcpy(file, "PCWT\1\0\0\0", 8);
  file[8] = (unsigned char)(h & 255u);
  file[9] = (unsigned char)(h >> 8);
  memset(file + 12, ' ', h);
  memcpy(file + 12, header, sizeof header - 1);
  EXPECT(rayo_model_create(file, 12 + h + 8, &m, &msg) == RAYO_OK);
  if (m && rayo_session_create(m, &s, &msg) == RAYO_OK) {
    rayo_session_input(s, 0, &in);
    ((float *)in.data)[0] = 1.0f;
    ((float *)in.data)[1] = 2.0f;
    EXPECT(rayo_session_run(s, &msg) == RAYO_OK);
    rayo_session_output(s, 0, &out);
    /* y = (x + x) + (x + x): the second node reads the first's x. */
    EXPECT(((const float *)out.data)[0] == 4.0f && ((const float *)out.data)[1] == 8.0f);
  }
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  free(file);
}

/* A bind that fails leaves every tensor's memory where it was: a node
 * whose infer failed used to lose its data pointer, and the next run wrote
 * through NULL (found by the model fuzz target). Here a reshape to [2]
 * fails for T = 3. */
static void failed_bind(void) {
  static const char header[] =
      "{\"format_version\":1,\"model\":\"bind\",\"tensors\":[],\"graph\":{\"version\":1,"
      "\"symbols\":{\"T\":4},\"plan\":{\"T\":2},\"inputs\":[{\"name\":\"x\",\"dtype\":"
      "\"float32\",\"shape\":[\"T\"]}],\"nodes\":[{\"op\":\"add\",\"inputs\":[\"x\",\"x\"],"
      "\"output\":\"a\"},{\"op\":\"reshape\",\"inputs\":[\"a\"],\"output\":\"b\",\"shape\":[2]},"
      "{\"op\":\"add\",\"inputs\":[\"b\",\"b\"],\"output\":\"c\"}],"
      "\"outputs\":[{\"name\":\"c\",\"node\":\"c\"}]}}";
  const size_t h = (sizeof header - 1 + 3) / 4 * 4;
  unsigned char *file = calloc(1, 12 + h);
  rayo_model *m = NULL;
  rayo_session *s = NULL;
  rayo_tensor_view before, after, in;
  rayo_message msg;
  if (!file) return;
  memcpy(file, "PCWT\1\0\0\0", 8);
  file[8] = (unsigned char)(h & 255u);
  file[9] = (unsigned char)(h >> 8);
  memset(file + 12, ' ', h);
  memcpy(file + 12, header, sizeof header - 1);
  EXPECT(rayo_model_create(file, 12 + h, &m, &msg) == RAYO_OK);
  EXPECT(m && rayo_session_create(m, &s, &msg) == RAYO_OK);
  if (s) {
    rayo_session_output(s, 0, &before);
    EXPECT(rayo_session_bind(s, "T", 3, NULL) == RAYO_ERR_SHAPE);
    rayo_session_input(s, 0, &in);
    ((float *)in.data)[0] = 1.0f;
    ((float *)in.data)[1] = 2.0f;
    EXPECT(rayo_session_run(s, NULL) == RAYO_OK);
    rayo_session_output(s, 0, &after);
    EXPECT(after.data == before.data && after.data != NULL);
    EXPECT(after.data && ((const float *)after.data)[1] == 8.0f);
  }
  rayo_session_destroy(s);
  rayo_model_destroy(m);
  free(file);
}

void test_security(void) {
  failed_bind();
  malformed_corpus();
  tokenizer_limits();
  tokenizer_text();
  json_limits();
  platform_files();
  sha512_vectors();
  ed25519_vectors();
  model_integrity();
  graph_names();
}
