#include "ops/scratch.h"

void rayo_scratch_add(size_t *total, size_t bytes) {
  *total += rayo_align_up(bytes ? bytes : 1);
}

void *rayo_scratch_take(unsigned char **at, size_t bytes) {
  void *p = *at;
  *at += rayo_align_up(bytes ? bytes : 1);
  return p;
}
