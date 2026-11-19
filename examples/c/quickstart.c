/* quickstart: load a model, make a session, set the input, run, read the
 * output. Runs examples/classifier/classifier.weights on three token ids:
 *
 *   make quickstart
 *   ./examples/c/quickstart examples/classifier/classifier.weights
 *
 * It prints the three labels' probabilities: 0.275287, 0.227712, 0.497001. */
#include <stdio.h>

#include "rayo/rayo.h"

int main(int argc, char **argv) {
  static const rayo_i32 ids[3] = {4, 17, 9};
  rayo_model *model = NULL;
  rayo_session *session = NULL;
  rayo_tensor_view in, out;
  rayo_message msg;
  size_t i;
  int status = 1;
  if (argc != 2) return 2;
  if (rayo_model_load(argv[1], NULL, &model, &msg) != RAYO_OK ||
      rayo_session_create(model, &session, &msg) != RAYO_OK ||
      rayo_session_bind(session, "T", 3, &msg) != RAYO_OK) {
    fprintf(stderr, "rayo: %s\n", msg.text);
    goto done;
  }
  rayo_session_input(session, 0, &in); /* int32 [T]: written in place */
  for (i = 0; i < 3; i++) ((rayo_i32 *)in.data)[i] = ids[i];
  if (rayo_session_run(session, &msg) != RAYO_OK) {
    fprintf(stderr, "rayo: %s\n", msg.text);
    goto done;
  }
  rayo_session_output(session, 0, &out); /* float32 [3]: read in place */
  for (i = 0; i < out.bytes / sizeof(rayo_f32); i++) printf("%.6f\n", ((rayo_f32 *)out.data)[i]);
  status = 0;
done:
  rayo_session_destroy(session); /* before the model; NULL is fine */
  rayo_model_destroy(model);
  return status;
}
