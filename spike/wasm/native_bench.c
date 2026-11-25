#define _POSIX_C_SOURCE 199309L
/* Native comparison: native_bench model.weights inputs.txt split T runs
 * inputs.txt: whitespace-separated int32s. Prints ms per run and output 0..3 of the last output. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rayo/rayo.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
  FILE *f; long size; void *bytes; rayo_model *m; rayo_session *s; rayo_tensor_view v;
  int ids[4096], n = 0, split = atoi(argv[3]), T = atoi(argv[4]), runs = atoi(argv[5]), i, k; double t0;
  f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); size = ftell(f); rewind(f);
  bytes = malloc(size); fread(bytes, 1, size, f); fclose(f);
  f = fopen(argv[2], "r"); while (fscanf(f, "%d", &ids[n]) == 1) n++; fclose(f);
  if (rayo_model_create(bytes, size, &m, NULL) || rayo_session_create(m, &s, NULL)) return 1;
  if (T) rayo_session_bind(s, "T", T, NULL);
  t0 = 0;
  for (i = 0; i < runs + 10; i++) {
    if (i == 10) t0 = now();
    rayo_session_input(s, 0, &v); memcpy(v.data, ids, split ? split * 4 : n * 4);
    if (split) { rayo_session_input(s, 1, &v); memcpy(v.data, ids + split, (n - split) * 4); }
    if (rayo_session_run(s, NULL)) return 2;
  }
  k = rayo_model_output_count(m) - 1; rayo_session_output(s, k, &v);
  printf("%.4f ms/run kernels=%s out[%d][0]=%.9g\n", (now() - t0) / runs, rayo_session_kernels(s), k, ((float *)v.data)[0]);
  return 0;
}
