/* Prints every output float of a model, as its bits, for fixed inputs, with
 * each kernel set:
 *
 *   outputs MODEL.weights...
 *
 * tool/ci/builds.sh builds this with two compilers and two sets of flags
 * and checks that both print the same: rayo gives the same floats whatever
 * compiler or optimisation built it. Inputs are at their largest shape:
 * floats from a fixed sequence in [-1, 1), integers from 0 to 3 (token ids,
 * shape classes and positions every test model reads). A model with a state
 * (a decoder's cache) runs three times, each run's outputs printed. */
#include <stdio.h>
#include <string.h>

#include "bench.h"

static rayo_u32 next(rayo_u32 *s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

static void fill(rayo_session *se, int n) {
  rayo_u32 seed = 12345u;
  int i;
  for (i = 0; i < n; i++) {
    rayo_tensor_view v;
    size_t j;
    if (rayo_session_input(se, i, &v) != RAYO_OK) continue;
    if (v.dtype == RAYO_DTYPE_F32) {
      float *f = v.data;
      for (j = 0; j < v.bytes / 4; j++) f[j] = (float)(next(&seed) >> 8) / 8388608.0f - 1.0f;
    } else if (v.dtype == RAYO_DTYPE_I32) {
      rayo_i32 *p = v.data;
      for (j = 0; j < v.bytes / 4; j++) p[j] = (rayo_i32)(next(&seed) % 4u);
    } else {
      memset(v.data, 0, v.bytes);
    }
  }
}

static int print_outputs(const rayo_session *se, int n) {
  int i;
  for (i = 0; i < n; i++) {
    rayo_tensor_view v;
    size_t j;
    if (rayo_session_output(se, i, &v) != RAYO_OK) return 1;
    printf("  %s:", v.name);
    if (v.dtype == RAYO_DTYPE_F32) {
      for (j = 0; j < v.bytes / 4; j++) {
        rayo_u32 b;
        memcpy(&b, (const float *)v.data + j, 4);
        printf("%s%08lx", j % 8 ? " " : "\n   ", (unsigned long)b);
      }
    } else {
      for (j = 0; j < v.bytes; j++) printf("%s%02x", j % 32 ? "" : "\n   ", ((rayo_u8 *)v.data)[j]);
    }
    printf("\n");
  }
  return 0;
}

static int run(const char *path) {
  static const char *const sets[3] = {"generic", "simd", "best"};
  rayo_message msg;
  rayo_model *m = NULL;
  int k, r, inputs, outputs;
  if (rayo_model_load(path, NULL, &m, &msg) != RAYO_OK) {
    fprintf(stderr, "%s: %s\n", path, msg.text);
    return 1;
  }
  inputs = rayo_model_input_count(m);
  outputs = rayo_model_output_count(m);
  for (k = 0; k < 3; k++) {
    rayo_session *se = NULL;
    if (rayo_session_create(m, &se, &msg) != RAYO_OK) {
      fprintf(stderr, "%s: %s\n", path, msg.text);
      rayo_model_destroy(m);
      return 1;
    }
    bench_choose_kernels(se, sets[k]);
    for (r = 0; r < 3; r++) {
      fill(se, inputs);
      if (rayo_session_run(se, &msg) != RAYO_OK) {
        fprintf(stderr, "%s: %s\n", path, msg.text);
        rayo_session_destroy(se);
        rayo_model_destroy(m);
        return 1;
      }
      printf("%s, %s kernels, run %d\n", rayo_model_name(m), sets[k], r + 1);
      if (print_outputs(se, outputs)) {
        rayo_session_destroy(se);
        rayo_model_destroy(m);
        return 1;
      }
    }
    rayo_session_destroy(se);
  }
  rayo_model_destroy(m);
  return 0;
}

int main(int argc, char **argv) {
  int i, failed = 0;
  if (argc < 2) {
    fprintf(stderr, "usage: %s MODEL.weights...\n", argv[0]);
    return 2;
  }
  for (i = 1; i < argc; i++) failed |= run(argv[i]);
  return failed;
}
