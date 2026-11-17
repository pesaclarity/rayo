#include "core/mathx.h"

#include <math.h>

/* erf from fdlibm's s_erf.c, as in FreeBSD's msun and musl, rewritten in
 * C89 for float arguments; the method and the coefficients are fdlibm's.
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * fdlibm's erf is within one ulp in double; rounding it to float gives the
 * float erf. Where fdlibm reads the high word of x to pick an interval,
 * this compares |x| with the same bounds as doubles. */

/* [0, 0.84375): erf(x) = x + x * pp(x^2) / qq(x^2). */
static const double efx = 1.28379167095512586316e-01;
static const double pp[5] = {1.28379167095512558561e-01, -3.25042107247001499370e-01,
                             -2.84817495755985104766e-02, -5.77027029648944159157e-03,
                             -2.37630166566501626084e-05};
static const double qq[6] = {1.0,
                             3.97917223959155352819e-01,
                             6.50222499887672944485e-02,
                             5.08130628187576562776e-03,
                             1.32494738004321644526e-04,
                             -3.96022827877536812320e-06};

/* [0.84375, 1.25): erf(x) = erx + pa(s) / qa(s), s = x - 1, erx is
 * 0.84506291151 rounded to float. */
static const double erx = 8.45062911510467529297e-01;
static const double pa[7] = {-2.36211856075265944077e-03, 4.14856118683748331666e-01,
                             -3.72207876035701323847e-01, 3.18346619901161753674e-01,
                             -1.10894694282396677476e-01, 3.54783043256182359371e-02,
                             -2.16637559486879084300e-03};
static const double qa[7] = {1.0,
                             1.06420880400844228286e-01,
                             5.40397917702171048937e-01,
                             7.18286544141962662868e-02,
                             1.26171219808761642112e-01,
                             1.36370839120290507362e-02,
                             1.19844998467991074170e-02};

/* [1.25, 1/0.35) and [1/0.35, 6): erf(x) = 1 - exp(-x^2 - 0.5625 + R/S) / x,
 * with R and S polynomials in s = 1 / x^2. */
static const double ra[8] = {-9.86494403484714822705e-03, -6.93858572707181764372e-01,
                             -1.05586262253232909814e+01, -6.23753324503260060396e+01,
                             -1.62396669462573470355e+02, -1.84605092906711035994e+02,
                             -8.12874355063065934246e+01, -9.81432934416914548592e+00};
static const double sa[9] = {1.0,
                             1.96512716674392571292e+01,
                             1.37657754143519042600e+02,
                             4.34565877475229228821e+02,
                             6.45387271733267880336e+02,
                             4.29008140027567833386e+02,
                             1.08635005541779435134e+02,
                             6.57024977031928170135e+00,
                             -6.04244152148580987438e-02};
static const double rb[7] = {-9.86494292470009928597e-03, -7.99283237680523006574e-01,
                             -1.77579549177547519889e+01, -1.60636384855821916062e+02,
                             -6.37566443368389627722e+02, -1.02509513161107724954e+03,
                             -4.83519191608651397019e+02};
static const double sb[8] = {1.0,
                             3.03380607434824582924e+01,
                             3.25792512996573918826e+02,
                             1.53672958608443695994e+03,
                             3.19985821950859553908e+03,
                             2.55305040643316442583e+03,
                             4.74528541206955367215e+02,
                             -2.24409524465858183362e+01};

/* The interval bounds: fdlibm's high words 0x3e300000 (2^-28),
 * 0x3feb0000, 0x3ff40000, 0x4006db6e (just above 1/0.35) and 0x40180000. */
#define ERF_TINY 3.7252902984619140625e-09
#define ERF_SMALL 0.84375
#define ERF_MIDDLE 1.25
#define ERF_FAR 2.8571434020996094
#define ERF_ONE 6.0

/* c[0] + t * (c[1] + t * (... + t * c[n - 1])), in fdlibm's order. */
static double poly(const double *c, int n, double t) {
  double v = c[n - 1];
  int i;
  for (i = n - 2; i >= 0; i--) v = c[i] + t * v;
  return v;
}

/* erf(a) for 0 <= a < 6 in double. fdlibm splits x into z, its high word,
 * to keep z * z exact: a float has no bits in the low word, so z is a. */
static double erf_positive(double a) {
  double s;
  if (a < ERF_SMALL) {
    if (a < ERF_TINY) return a + efx * a;
    s = a * a;
    return a + a * (poly(pp, 5, s) / poly(qq, 6, s));
  }
  if (a < ERF_MIDDLE) {
    s = a - 1.0;
    return erx + poly(pa, 7, s) / poly(qa, 7, s);
  }
  s = 1.0 / (a * a);
  if (a < ERF_FAR) {
    s = poly(ra, 8, s) / poly(sa, 9, s);
  } else {
    s = poly(rb, 7, s) / poly(sb, 8, s);
  }
  return 1.0 - exp(-a * a - 0.5625) * exp(s) / a;
}

float rayo_erff(float xf) {
  const double x = (double)xf;
  const double a = x < 0.0 ? -x : x;
  double r;
  if (x != x) return xf;
  r = a < ERF_ONE ? erf_positive(a) : 1.0;
  return (float)(x < 0.0 ? -r : r);
}

float rayo_expf(float x) {
  return (float)exp((double)x);
}

float rayo_sqrtf(float x) {
  return (float)sqrt((double)x);
}
