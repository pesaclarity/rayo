/* What C89 lacks that rayo uses, kept in one place. */
#ifndef RAYO_CORE_COMPAT_H
#define RAYO_CORE_COMPAT_H

#include <stddef.h>

#include "rayo/types.h"

/* `restrict` where the compiler has it as an extension, else nothing. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RAYO_RESTRICT __restrict
#else
#define RAYO_RESTRICT
#endif

/* printf-style checking where the compiler has it. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_PRINTF(f, a) __attribute__((format(printf, f, a)))
#else
#define RAYO_PRINTF(f, a)
#endif

#define RAYO_SIZE_MAX ((size_t) - 1)

#endif
