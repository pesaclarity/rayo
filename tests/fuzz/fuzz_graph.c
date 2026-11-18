/* A model's JSON header alone: the target wraps it in a PCWT file whose data
 * is as long as the tensors it names need (filled with 0.25f), so the
 * fuzzer spends its time on the graph, the ops' attributes and shapes, and
 * the plan, not on lining offsets up. Seeds: the headers of tests/models. */
#include <stdlib.h>
#include <string.h>

#include "core/json.h"
#include "fuzz.h"

#define MAX_DATA (4L * 1024 * 1024)

static size_t end_of(const rayo_json *obj, const char *off_key, const char *len_key, size_t end) {
  rayo_i32 off, len;
  if (rayo_json_as_int(rayo_json_get(obj, off_key), &off) &&
      rayo_json_as_int(rayo_json_get(obj, len_key), &len) && off >= 0 && len >= 0 &&
      (size_t)off + (size_t)len + 3 <= (size_t)MAX_DATA) {
    const size_t e = ((size_t)off + (size_t)len + 3) / 4 * 4;
    if (e > end) end = e;
  }
  return end;
}

/* The data's length: the end of the last tensor, scale table or table. */
static size_t data_size(const char *header, size_t n) {
  rayo_arena arena;
  rayo_json *root;
  size_t end = 0, i;
  memset(&arena, 0, sizeof arena);
  if (rayo_json_parse(header, n, (size_t)-1, &arena, &root, NULL) == RAYO_OK) {
    const rayo_json *tensors = rayo_json_get(root, "tensors");
    for (i = 0; tensors && tensors->kind == RAYO_JSON_ARRAY && i < tensors->length; i++) {
      end = end_of(tensors->items[i], "offset", "bytes", end);
      end = end_of(rayo_json_get(tensors->items[i], "quantization"), "scales_offset",
                   "scales_bytes", end);
    }
    end = end_of(rayo_json_get(root, "tokenizer"), "offset", "bytes", end);
  }
  rayo_arena_free(&arena);
  return end;
}

static void put_u32(unsigned char *p, size_t v) {
  p[0] = (unsigned char)(v & 255u);
  p[1] = (unsigned char)((v >> 8) & 255u);
  p[2] = (unsigned char)((v >> 16) & 255u);
  p[3] = (unsigned char)((v >> 24) & 255u);
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
  size_t header, body, total, i;
  unsigned char *file;
  if (size > (size_t)MAX_DATA) return 0;
  header = (size + 3) / 4 * 4;
  body = data_size((const char *)data, size);
  total = 12 + header + body;
  file = (unsigned char *)malloc(total);
  if (!file) return 0;
  memcpy(file, "PCWT", 4);
  put_u32(file + 4, 1);
  put_u32(file + 8, header);
  memcpy(file + 12, data, size);
  memset(file + 12 + size, ' ', header - size);
  for (i = 0; i + 4 <= body; i += 4) {
    /* 0.25f, little-endian: finite scales and small weights. */
    file[12 + header + i] = 0x00;
    file[12 + header + i + 1] = 0x00;
    file[12 + header + i + 2] = 0x80;
    file[12 + header + i + 3] = 0x3E;
  }
  rayo_fuzz_model(file, total);
  free(file);
  return 0;
}
