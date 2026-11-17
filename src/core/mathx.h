/* Float maths C89 has only in double: exp, sqrt, and erf, which C89 lacks
 * altogether. Each is computed in double and rounded once to float. */
#ifndef RAYO_CORE_MATHX_H
#define RAYO_CORE_MATHX_H

/* exp(x), from C89's double exp. */
float rayo_expf(float x);

/* The square root, exactly as sqrtf (double sqrt is correctly rounded, so
 * rounding it to float is too). */
float rayo_sqrtf(float x);

/* erf(x): piecewise polynomials fitted in double (error below 6e-15
 * against erf to 40 digits), rounded to float. On every float in
 * [-6, 6] it is the correctly rounded value, or within half an ulp of it
 * where the polynomials meet; it differs from glibc's erff (which is off by
 * one ulp on about 5% of inputs) only where glibc is the less accurate. */
float rayo_erff(float x);

/* The largest float, for maxima: -RAYO_FLT_BIG stands in for -infinity. */
#define RAYO_FLT_BIG 3.402823466e+38f

#endif
