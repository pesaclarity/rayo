/* rayo_erff against the C library's erf on every float in [0, 4], a sample
 * of [4, 7], the negatives, and either side of every interval bound, old
 * and new. C99, to use the C library's erf, erff and nextafterf; rayo's own
 * src/core/mathx.c is compiled as C89 and linked in (tool/ci/erf.sh).
 *
 * The reference is erf in double rounded once to float, which is within
 * half an ulp of the float erf except where the double erf's own error
 * crosses a rounding boundary. Exits non-zero when rayo_erff is more than
 * one ulp from it anywhere, when it is not monotonic, or when it is not odd. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/mathx.h"

static uint32_t bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

static float from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

/* The distance in ulps between two non-negative floats. */
static uint32_t ulps(float a, float b) {
  const uint32_t x = bits(a), y = bits(b);
  return x > y ? x - y : y - x;
}

static float reference(float x) {
  return (float)erf((double)x);
}

struct tally {
  unsigned long n, off, off_libc;
  uint32_t worst, worst_libc;
  float worst_at;
  unsigned long not_monotonic, not_odd;
};

static void one(struct tally *t, float x, float *last) {
  const float r = rayo_erff(x);
  const float want = reference(x);
  const uint32_t d = ulps(r, want);
  const uint32_t dl = ulps(erff(x), want);
  t->n++;
  if (d) t->off++;
  if (d > t->worst) {
    t->worst = d;
    t->worst_at = x;
  }
  if (dl) t->off_libc++;
  if (dl > t->worst_libc) t->worst_libc = dl;
  if (last && r < *last) t->not_monotonic++;
  if (last) *last = r;
  if (rayo_erff(-x) != -r) t->not_odd++;
}

static void report(const char *name, const struct tally *t) {
  printf(
      "%s: %lu floats; rayo_erff off by 1+ ulp at %lu (worst %u ulp at %.9g), "
      "the C library's erff at %lu (worst %u ulp); not monotonic %lu, not odd %lu\n",
      name, t->n, t->off, (unsigned)t->worst, (double)t->worst_at, t->off_libc,
      (unsigned)t->worst_libc, t->not_monotonic, t->not_odd);
}

/* Steps across [bound - 8 ulp, bound + 8 ulp]: each step of rayo_erff must
 * be the reference's step within one ulp, so no bound leaves a jump. */
static int across(double bound, const char *what) {
  float x = (float)bound;
  int i, bad = 0;
  for (i = 0; i < 8; i++) x = nextafterf(x, 0.0f);
  for (i = 0; i < 16; i++) {
    const float y = nextafterf(x, 10.0f);
    const long step = (long)bits(rayo_erff(y)) - (long)bits(rayo_erff(x));
    const long want = (long)bits(reference(y)) - (long)bits(reference(x));
    if (step < 0 || step - want > 1 || want - step > 1) {
      printf("  %s bound %.17g: step %.9g -> %.9g is %ld ulp, the reference %ld\n", what, bound,
             (double)x, (double)y, step, want);
      bad = 1;
    }
    x = y;
  }
  return bad;
}

int main(void) {
  /* The old fitted polynomials met at 1, 1.75, 2.5, 3.25 and 4; fdlibm's
   * intervals at 2^-28, 0.84375, 1.25, 1/0.35 and 6. */
  static const double old_bounds[] = {1.0, 1.75, 2.5, 3.25, 4.0};
  static const double new_bounds[] = {3.7252902984619140625e-09, 0.84375, 1.25, 2.8571434020996094,
                                      6.0};
  struct tally all, beyond;
  float last = 0.0f;
  uint32_t u;
  const uint32_t four = bits(4.0f), seven = bits(7.0f);
  int bad = 0;
  size_t i;
  memset(&all, 0, sizeof all);
  memset(&beyond, 0, sizeof beyond);
  for (u = 0; u <= four; u++) one(&all, from_bits(u), &last);
  report("[0, 4], every float", &all);
  for (u = four; u <= seven; u += 64) one(&beyond, from_bits(u), NULL);
  report("[4, 7], every 64th float", &beyond);
  for (i = 0; i < sizeof old_bounds / sizeof old_bounds[0]; i++)
    bad |= across(old_bounds[i], "old");
  for (i = 0; i < sizeof new_bounds / sizeof new_bounds[0]; i++)
    bad |= across(new_bounds[i], "new");
  printf("bounds: %s\n", bad ? "a jump" : "no jump at any old or new bound");
  if (rayo_erff(INFINITY) != 1.0f || rayo_erff(-INFINITY) != -1.0f || !isnan(rayo_erff(NAN))) {
    printf("erf of infinity or NaN is wrong\n");
    bad = 1;
  }
  bad |= all.worst > 1 || beyond.worst > 1 || all.not_monotonic || all.not_odd || beyond.not_odd;
  return bad;
}
