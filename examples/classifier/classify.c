/* Runs examples/classifier/classifier.weights on token ids given as
 * arguments and prints each label's probability:
 *
 *   classify classifier.weights 4 17 9
 *
 * The whole of rayo's C API a program needs: load a model, make a session,
 * bind the sequence length, write the input in place, run, read the
 * output in place. */
#include <stdio.h>
#include <stdlib.h>

#include "rayo/rayo.h"

static unsigned char *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  unsigned char *p = NULL;
  long n;
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0) {
    p = malloc((size_t)n);
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
      free(p);
      p = NULL;
    }
    *size = (size_t)n;
  }
  fclose(f);
  return p;
}

int main(int argc, char **argv) {
  rayo_model *model = NULL;
  rayo_session *session = NULL;
  rayo_tensor_view in, out;
  rayo_message msg;
  unsigned char *file;
  size_t size = 0, i;
  int n = argc - 2, status = 1;
  if (argc < 3) {
    fprintf(stderr, "usage: classify MODEL.weights ID...\n");
    return 2;
  }
  file = read_file(argv[1], &size);
  if (!file) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  if (rayo_model_create(file, size, &model, &msg) != RAYO_OK ||
      rayo_session_create(model, &session, &msg) != RAYO_OK ||
      rayo_session_bind(session, "T", n, &msg) != RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    goto done;
  }
  rayo_session_input(session, 0, &in);
  for (i = 0; i < (size_t)n; i++) ((rayo_i32 *)in.data)[i] = (rayo_i32)atol(argv[i + 2]);
  if (rayo_session_run(session, &msg) != RAYO_OK) {
    fprintf(stderr, "%s\n", msg.text);
    goto done;
  }
  rayo_session_output(session, 0, &out);
  for (i = 0; i < out.bytes / sizeof(float); i++) printf("%.7f\n", ((float *)out.data)[i]);
  printf("kernels: %s\n", rayo_session_kernels(session));
  status = 0;
done:
  rayo_session_destroy(session);
  rayo_model_destroy(model);
  free(file);
  return status;
}
