/* struct rayo_model: a model file's copy, its parsed header and graph. */
#ifndef RAYO_MODEL_INTERNAL_H
#define RAYO_MODEL_INTERNAL_H

#include <stddef.h>

#include "core/arena.h"
#include "format/pcwt.h"
#include "graph/graph.h"
#include "platform/platform.h"
#include "rayo/model.h"

struct rayo_model {
  const rayo_u8 *file; /* the weights point into it: `map`'s bytes */
  size_t file_size;
  rayo_file_map map; /* owned: a copy of the caller's bytes, or the mapped file */
  rayo_arena arena;  /* owns the header, the graph and the packed weights */
  rayo_pcwt pcwt;
  rayo_graph graph;
  const char *metadata;     /* JSON text */
  const rayo_u8 *tokenizer; /* the tokeniser's table in the file, or NULL */
  size_t tokenizer_size;
};

#endif
