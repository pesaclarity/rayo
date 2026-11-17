#include "model.h"

#include <stdlib.h>
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"

static rayo_status read_header(rayo_model *m, const rayo_model_options *options,
                               rayo_message *msg) {
  const rayo_json *meta;
  rayo_status s = rayo_pcwt_read(m->file, m->file_size, &m->arena, &m->pcwt, msg);
  if (s != RAYO_OK) return s;
  s = rayo_graph_read(rayo_json_get(m->pcwt.header, "graph"), &m->pcwt, options, &m->arena,
                      &m->graph, msg);
  if (s != RAYO_OK) return s;
  meta = rayo_json_get(m->pcwt.header, "metadata");
  if (meta && meta->kind != RAYO_JSON_OBJECT) {
    return rayo_fail(msg, RAYO_ERR_FORMAT, "the header's metadata is not an object");
  }
  m->metadata = meta ? rayo_arena_strndup(&m->arena, meta->raw, meta->raw_length)
                     : rayo_arena_strndup(&m->arena, "{}", 2);
  if (!m->metadata) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  return RAYO_OK;
}

rayo_status rayo_model_create(const void *data, size_t size, rayo_model **out, rayo_message *msg) {
  return rayo_model_create_with(data, size, NULL, out, msg);
}

rayo_status rayo_model_create_with(const void *data, size_t size, const rayo_model_options *options,
                                   rayo_model **out, rayo_message *msg) {
  rayo_model *m;
  rayo_status s;
  if (!out || (!data && size) || (options && options->op_count > 0 && !options->ops)) {
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
  }
  *out = NULL;
  if (size == 0) return rayo_fail(msg, RAYO_ERR_FORMAT, "an empty file");
  m = calloc(1, sizeof *m);
  if (!m) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  m->file = rayo_aligned_alloc(size);
  if (!m->file) {
    free(m);
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory for %lu bytes",
                     (unsigned long)size);
  }
  memcpy(m->file, data, size);
  m->file_size = size;
  s = read_header(m, options, msg);
  if (s != RAYO_OK) {
    rayo_model_destroy(m);
    return s;
  }
  *out = m;
  return RAYO_OK;
}

void rayo_model_destroy(rayo_model *model) {
  if (!model) return;
  rayo_arena_free(&model->arena);
  rayo_aligned_free(model->file);
  free(model);
}

const char *rayo_model_name(const rayo_model *model) {
  return model && model->pcwt.model ? model->pcwt.model : "";
}

const char *rayo_model_metadata_json(const rayo_model *model) {
  return model ? model->metadata : "{}";
}

int rayo_model_input_count(const rayo_model *model) {
  return model ? model->graph.input_count : 0;
}

int rayo_model_output_count(const rayo_model *model) {
  return model ? model->graph.output_count : 0;
}

rayo_status rayo_model_input_info(const rayo_model *model, int index, rayo_tensor_view *out) {
  const rayo_graph_input *in;
  rayo_i32 d;
  if (!model || !out || index < 0 || index >= model->graph.input_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  in = &model->graph.inputs[index];
  memset(out, 0, sizeof *out);
  out->name = in->name;
  out->dtype = in->dtype;
  out->shape.rank = in->rank;
  for (d = 0; d < in->rank; d++) {
    const rayo_dim dim = in->dims[d];
    out->shape.dims[d] = dim.symbol >= 0 ? model->graph.symbol_max[dim.symbol] : dim.value;
  }
  return RAYO_OK;
}

rayo_status rayo_model_output_info(const rayo_model *model, int index, rayo_tensor_view *out) {
  if (!model || !out || index < 0 || index >= model->graph.output_count) {
    return RAYO_ERR_INVALID_ARGUMENT;
  }
  memset(out, 0, sizeof *out);
  out->name = model->graph.outputs[index].name;
  out->dtype = RAYO_DTYPE_F32;
  return RAYO_OK;
}

void rayo_model_memory(const rayo_model *model, rayo_memory_info *out) {
  if (!out) return;
  memset(out, 0, sizeof *out);
  if (!model) return;
  out->weights = model->file_size;
  out->overhead = model->arena.bytes;
}
