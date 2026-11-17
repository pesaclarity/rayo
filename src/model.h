/* struct rayo_model: a model file's copy, its parsed header and graph. */
#ifndef RAYO_MODEL_INTERNAL_H
#define RAYO_MODEL_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "core/arena.h"
#include "format/pcwt.h"
#include "graph/graph.h"
#include "rayo/model.h"

struct rayo_model {
  uint8_t *file; /* owned, RAYO_ALIGN-aligned copy; the weights point into it */
  size_t file_size;
  rayo_arena arena; /* owns the header, the graph and the packed weights */
  rayo_pcwt pcwt;
  rayo_graph graph;
  const char *metadata; /* JSON text */
};

#endif
