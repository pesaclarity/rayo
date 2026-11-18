/* Choosing the next token (rayo/text.h): repetition penalty, temperature,
 * top-k, top-p and min-p, from a seeded random number generator
 * (xorshift128, 32-bit words, as C89 has no 64-bit integers). All working
 * memory is allocated when the sampler is made. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/message.h"
#include "rayo/text.h"

typedef struct candidate {
  float logit;
  rayo_i32 id;
} candidate;

struct rayo_sampler {
  rayo_sampler_options o;
  rayo_i32 vocab;
  rayo_u32 s[4];
  candidate *c;     /* vocab of them */
  float *penalised; /* vocab */
};

static rayo_u32 mix(rayo_u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static rayo_u32 next_u32(rayo_sampler *r) {
  rayo_u32 t = r->s[3];
  const rayo_u32 s0 = r->s[0];
  r->s[3] = r->s[2];
  r->s[2] = r->s[1];
  r->s[1] = s0;
  t ^= t << 11;
  t ^= t >> 8;
  r->s[0] = t ^ s0 ^ (s0 >> 19);
  return r->s[0];
}

/* Uniform in [0, 1). */
static double uniform(rayo_sampler *r) {
  return (double)(next_u32(r) >> 8) / 16777216.0;
}

rayo_status rayo_sampler_create(rayo_i32 vocab, const rayo_sampler_options *options,
                                rayo_sampler **out, rayo_message *msg) {
  rayo_sampler *r;
  int i;
  if (!out || vocab < 1) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "bad argument");
  *out = NULL;
  r = calloc(1, sizeof *r);
  if (!r) return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  if (options) r->o = *options;
  r->vocab = vocab;
  r->c = malloc((size_t)vocab * sizeof *r->c);
  r->penalised = malloc((size_t)vocab * sizeof *r->penalised);
  if (!r->c || !r->penalised) {
    rayo_sampler_destroy(r);
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory");
  }
  for (i = 0; i < 4; i++) r->s[i] = mix(r->o.seed + 0x9E3779B9u * (rayo_u32)(i + 1));
  if (!(r->s[0] | r->s[1] | r->s[2] | r->s[3])) r->s[0] = 1;
  *out = r;
  return RAYO_OK;
}

void rayo_sampler_destroy(rayo_sampler *sampler) {
  if (!sampler) return;
  free(sampler->c);
  free(sampler->penalised);
  free(sampler);
}

/* Moves the k largest of c[0 .. n) to the front, largest first (a heap of
 * the k largest, then sorted). */
static void sift(candidate *h, size_t k, size_t i) {
  for (;;) {
    size_t l = 2 * i + 1, small = i;
    candidate t;
    if (l < k && h[l].logit < h[small].logit) small = l;
    if (l + 1 < k && h[l + 1].logit < h[small].logit) small = l + 1;
    if (small == i) return;
    t = h[i];
    h[i] = h[small];
    h[small] = t;
    i = small;
  }
}

static void top(candidate *c, size_t n, size_t k) {
  size_t i;
  for (i = k / 2; i > 0; i--) sift(c, k, i - 1);
  for (i = k; i < n; i++) {
    if (c[i].logit > c[0].logit) {
      c[0] = c[i];
      sift(c, k, 0);
    }
  }
  /* Heap sort the k: the smallest goes to the end each time. */
  for (i = k; i > 1; i--) {
    candidate t = c[0];
    c[0] = c[i - 1];
    c[i - 1] = t;
    sift(c, i - 1, 0);
  }
}

rayo_i32 rayo_sampler_next(rayo_sampler *sampler, const float *logits, const rayo_i32 *recent,
                           size_t recent_count) {
  rayo_sampler *r = sampler;
  const size_t n = r ? (size_t)r->vocab : 0;
  const float *l = logits;
  size_t i, k, keep;
  double sum, u, acc;
  float best;
  if (!r || !logits) return -1;
  if (r->o.repetition_penalty > 0.0f && r->o.repetition_penalty != 1.0f && recent_count) {
    memcpy(r->penalised, logits, n * sizeof *logits);
    for (i = 0; i < recent_count; i++) {
      const rayo_i32 id = recent[i];
      float *v;
      if (id < 0 || (size_t)id >= n) continue;
      v = &r->penalised[id];
      /* Each id once, however often it is recent: its value is compared
       * with the logit's, so a second visit finds it changed. */
      if (*v != logits[id]) continue;
      *v = *v > 0.0f ? *v / r->o.repetition_penalty : *v * r->o.repetition_penalty;
    }
    l = r->penalised;
  }
  if (r->o.temperature <= 0.0f) {
    rayo_i32 arg = 0;
    for (i = 1; i < n; i++) {
      if (l[i] > l[arg]) arg = (rayo_i32)i;
    }
    return arg;
  }
  for (i = 0; i < n; i++) {
    r->c[i].logit = l[i];
    r->c[i].id = (rayo_i32)i;
  }
  k = r->o.top_k > 0 && (size_t)r->o.top_k < n ? (size_t)r->o.top_k : n;
  top(r->c, n, k);
  /* Probabilities with the temperature, in double. */
  best = r->c[0].logit;
  sum = 0.0;
  for (i = 0; i < k; i++) {
    const double p = exp(((double)r->c[i].logit - best) / r->o.temperature);
    r->c[i].logit = (float)p;
    sum += p;
  }
  keep = k;
  if (r->o.min_p > 0.0f) {
    for (i = 1; i < keep; i++) {
      if (r->c[i].logit < r->o.min_p * r->c[0].logit) {
        keep = i;
        break;
      }
    }
  }
  if (r->o.top_p > 0.0f && r->o.top_p < 1.0f) {
    acc = 0.0;
    for (i = 0; i < keep; i++) {
      acc += r->c[i].logit / sum;
      if (acc >= r->o.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  sum = 0.0;
  for (i = 0; i < keep; i++) sum += r->c[i].logit;
  u = uniform(r) * sum;
  acc = 0.0;
  for (i = 0; i < keep; i++) {
    acc += r->c[i].logit;
    if (u < acc) return r->c[i].id;
  }
  return r->c[keep - 1].id;
}
