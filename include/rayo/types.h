/* rayo/types.h: element types and shapes. */
#ifndef RAYO_TYPES_H
#define RAYO_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define RAYO_MAX_RANK 4

typedef enum rayo_dtype {
  RAYO_DTYPE_F32 = 0,
  RAYO_DTYPE_I32 = 1,
  RAYO_DTYPE_I8 = 2,
} rayo_dtype;

/* Row-major (C order) dimensions, outermost first. */
typedef struct rayo_shape {
  int32_t rank;
  int64_t dims[RAYO_MAX_RANK];
} rayo_shape;

#endif
