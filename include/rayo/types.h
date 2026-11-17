/* rayo/types.h: fixed-size types, element types and shapes, in C89.
 *
 * Every size and count rayo handles fits rayo_i32 (a dimension is at most
 * 2^31 - 1) or size_t (bytes). */
#ifndef RAYO_TYPES_H
#define RAYO_TYPES_H

#include <limits.h>
#include <stddef.h>

#include "rayo/config.h"

typedef signed char rayo_i8;
typedef unsigned char rayo_u8;
typedef short rayo_i16;
typedef unsigned short rayo_u16;
#if INT_MAX == 2147483647
typedef int rayo_i32;
typedef unsigned int rayo_u32;
#else
typedef long rayo_i32;
typedef unsigned long rayo_u32;
#endif
typedef float rayo_f32;
typedef size_t rayo_size;
typedef int rayo_bool;

#define RAYO_TRUE 1
#define RAYO_FALSE 0
#define RAYO_I32_MAX 2147483647

/* Compile-time checks: a negative array size fails the build. */
typedef char rayo_check_i8[sizeof(rayo_i8) == 1 ? 1 : -1];
typedef char rayo_check_i16[sizeof(rayo_i16) == 2 ? 1 : -1];
typedef char rayo_check_i32[sizeof(rayo_i32) == 4 ? 1 : -1];
typedef char rayo_check_u32[sizeof(rayo_u32) == 4 ? 1 : -1];
typedef char rayo_check_f32[sizeof(rayo_f32) == 4 ? 1 : -1];
typedef char rayo_check_char[CHAR_BIT == 8 ? 1 : -1];

typedef enum rayo_dtype { RAYO_DTYPE_F32 = 0, RAYO_DTYPE_I32 = 1, RAYO_DTYPE_I8 = 2 } rayo_dtype;

/* Row-major (C order) dimensions, outermost first. */
typedef struct rayo_shape {
  rayo_i32 rank;
  rayo_i32 dims[RAYO_MAX_RANK];
} rayo_shape;

#endif
