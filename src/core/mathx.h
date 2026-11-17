/* Float maths C89 has only in double: exp, sqrt, and erf, which C89 lacks
 * altogether. Each is computed in double and rounded once to float. */
#ifndef RAYO_CORE_MATHX_H
#define RAYO_CORE_MATHX_H

/* exp(x), from C89's double exp. */
float rayo_expf(float x);

/* The square root, exactly as sqrtf (double sqrt is correctly rounded, so
 * rounding it to float is too). */
float rayo_sqrtf(float x);

/* erf(x): fdlibm's double erf (Sun's permissive licence, as in FreeBSD and
 * musl), rounded to float. tools/erf_check compares it with the C library's
 * erf on every float in [0, 4]. */
float rayo_erff(float x);

/* The largest float, for maxima: -RAYO_FLT_BIG stands in for -infinity. */
#define RAYO_FLT_BIG 3.402823466e+38f

#endif
