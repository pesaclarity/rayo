#include "ops/scratch.h"

#include "core/compat.h"

void rayo_scratch_add(size_t *total, size_t bytes) {
  const size_t aligned = rayo_align_up(bytes ? bytes : 1);
  /* Saturates: a plan over RAYO_MAX_ARENA_BYTES is refused, so an op's
   * sizes never wrap around to a small scratch, on 32-bit systems too. */
  *total = aligned == 0 || *total > RAYO_SIZE_MAX - aligned ? RAYO_SIZE_MAX : *total + aligned;
}

size_t rayo_scratch_bytes(size_t a, size_t b, size_t c) {
  if ((b != 0 && a > RAYO_SIZE_MAX / b) || (c != 0 && a * b > RAYO_SIZE_MAX / c)) {
    return RAYO_SIZE_MAX;
  }
  return a * b * c;
}

void *rayo_scratch_take(unsigned char **at, size_t bytes) {
  void *p = *at;
  *at += rayo_align_up(bytes ? bytes : 1);
  return p;
}
