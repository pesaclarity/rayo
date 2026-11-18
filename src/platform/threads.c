/* A pool of worker threads: pthreads on POSIX systems, Win32 threads on
 * Windows, none otherwise (one thread).
 *
 * A run hands out its parts (`grain` indices each, a few at a time) from a
 * shared counter, so a fast core takes more of them than a slow one and
 * nobody waits for a thread that is late or asleep: the caller returns as
 * soon as every part is done. Between runs the workers spin briefly (a
 * decoder's ops come microseconds apart, and waking a sleeping thread costs
 * about as much as a small op), with the CPU's pause hint, and then sleep
 * on a condition variable, so an idle pool uses no CPU. */
#if !defined(RAYO_NO_PLATFORM) && !defined(RAYO_NO_THREADS) && \
    (defined(__unix__) || defined(__APPLE__))
#define RAYO_THREADS_POSIX 1
#define _POSIX_C_SOURCE 200112L
#elif !defined(RAYO_NO_PLATFORM) && !defined(RAYO_NO_THREADS) && defined(_WIN32)
#define RAYO_THREADS_WIN32 1
#endif

#include <stdlib.h>

#include "platform/platform.h"

#if defined(RAYO_THREADS_POSIX)
#include <pthread.h>
#include <time.h>
typedef pthread_mutex_t rayo_mutex;
typedef pthread_cond_t rayo_cond;
typedef pthread_t rayo_thread;
#define RAYO_HAS_THREADS 1
#elif defined(RAYO_THREADS_WIN32)
#include <windows.h>
typedef CRITICAL_SECTION rayo_mutex;
typedef CONDITION_VARIABLE rayo_cond;
typedef HANDLE rayo_thread;
#define RAYO_HAS_THREADS 1
#endif

#if defined(RAYO_HAS_THREADS)

/* Atomic counters: the compiler's builtins, Win32's Interlocked functions,
 * or (any other compiler) a mutex of their own. All sequentially
 * consistent. */
#if defined(__GNUC__) || defined(__clang__)
#define RAYO_ATOMIC_BUILTINS 1
#elif defined(RAYO_THREADS_WIN32)
#define RAYO_ATOMIC_WIN32 1
#endif

struct rayo_pool {
  int threads; /* in all, the caller's included */
  rayo_thread *workers;
  rayo_mutex lock;
  rayo_cond wake; /* a new job, or quit */
  rayo_cond done; /* the job's last part finished */
#if !defined(RAYO_ATOMIC_BUILTINS) && !defined(RAYO_ATOMIC_WIN32)
  rayo_mutex atomic;
#endif
  /* Even while a job may be taken, odd while the caller writes the next
   * one; changed only by the caller. */
  long generation;
  /* The job: written by the caller while `generation` is odd and no worker
   * is `active`, then only read. */
  rayo_task_fn *fn;
  void *context;
  size_t n, grain;
  long units, chunk; /* parts of `grain`, taken `chunk` at a time */
  long next;         /* the next part to take */
  long finished;     /* parts done */
  long active;       /* workers between seeing a job and leaving it */
  /* Under `lock`. */
  int sleepers;      /* workers waiting on `wake` */
  rayo_bool waiting; /* the caller waits on `done` */
  long quit;
};

#if defined(RAYO_ATOMIC_BUILTINS)
static long atomic_get(rayo_pool *pool, long *p) {
  (void)pool;
  return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}
static void atomic_set(rayo_pool *pool, long *p, long v) {
  (void)pool;
  __atomic_store_n(p, v, __ATOMIC_SEQ_CST);
}
/* Adds v; the value before. */
static long atomic_add(rayo_pool *pool, long *p, long v) {
  (void)pool;
  return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}
#elif defined(RAYO_ATOMIC_WIN32)
static long atomic_get(rayo_pool *pool, long *p) {
  (void)pool;
  return InterlockedCompareExchange((LONG volatile *)p, 0, 0);
}
static void atomic_set(rayo_pool *pool, long *p, long v) {
  (void)pool;
  InterlockedExchange((LONG volatile *)p, v);
}
static long atomic_add(rayo_pool *pool, long *p, long v) {
  (void)pool;
  return InterlockedExchangeAdd((LONG volatile *)p, v);
}
#else
static long atomic_get(rayo_pool *pool, long *p) {
  long v;
  pthread_mutex_lock(&pool->atomic);
  v = *p;
  pthread_mutex_unlock(&pool->atomic);
  return v;
}
static void atomic_set(rayo_pool *pool, long *p, long v) {
  pthread_mutex_lock(&pool->atomic);
  *p = v;
  pthread_mutex_unlock(&pool->atomic);
}
static long atomic_add(rayo_pool *pool, long *p, long v) {
  long old;
  pthread_mutex_lock(&pool->atomic);
  old = *p;
  *p = old + v;
  pthread_mutex_unlock(&pool->atomic);
  return old;
}
#endif

/* The CPU's hint that this thread is spinning (fewer resources taken
 * from a sibling thread, less power), where the compiler has a way to
 * say it. */
static void relax(void) {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
  __asm__ __volatile__("pause");
#elif defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 7)
  __asm__ __volatile__("yield");
#endif
#elif defined(RAYO_THREADS_WIN32)
  YieldProcessor();
#endif
}

/* A clock in microseconds, for how long to spin. */
static double now_us(void) {
#if defined(RAYO_THREADS_POSIX)
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return 0.0;
  return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
#else
  LARGE_INTEGER c, f;
  QueryPerformanceCounter(&c);
  QueryPerformanceFrequency(&f);
  return (double)c.QuadPart * 1e6 / (double)f.QuadPart;
#endif
}

/* How long a worker spins after a job before it sleeps, and how long the
 * caller spins for the last parts before it sleeps, in microseconds: long
 * enough to cover the gaps between a decoder's ops, short enough that the
 * gaps between tokens (and an idle app) cost nothing. */
#define WORKER_SPIN_US 300.0
#define CALLER_SPIN_US 2000.0

/* Spins until *p differs from `old` (or quit, for workers), for at most
 * `us` microseconds; whether it changed. */
static rayo_bool spin_while(rayo_pool *pool, long *p, long old, double us) {
  double until = -1.0;
  long i;
  for (i = 0;; i++) {
    if (atomic_get(pool, p) != old || atomic_get(pool, &pool->quit)) return RAYO_TRUE;
    relax();
    if ((i & 63) == 63) {
      const double t = now_us();
      if (until < 0.0) until = t + us;
      if (t >= until) return RAYO_FALSE;
    }
  }
}

#if defined(RAYO_THREADS_POSIX)
static void lock(rayo_mutex *m) {
  pthread_mutex_lock(m);
}
static void unlock(rayo_mutex *m) {
  pthread_mutex_unlock(m);
}
static void wait_on(rayo_cond *c, rayo_mutex *m) {
  pthread_cond_wait(c, m);
}
static void wake_all(rayo_cond *c) {
  pthread_cond_broadcast(c);
}
static void wake_one(rayo_cond *c) {
  pthread_cond_signal(c);
}
#else
static void lock(rayo_mutex *m) {
  EnterCriticalSection(m);
}
static void unlock(rayo_mutex *m) {
  LeaveCriticalSection(m);
}
static void wait_on(rayo_cond *c, rayo_mutex *m) {
  SleepConditionVariableCS(c, m, INFINITE);
}
static void wake_all(rayo_cond *c) {
  WakeAllConditionVariable(c);
}
static void wake_one(rayo_cond *c) {
  WakeConditionVariable(c);
}
#endif

/* Takes parts of the current job until none is left. */
static void take_parts(rayo_pool *pool) {
  const long units = pool->units, chunk = pool->chunk;
  for (;;) {
    const long u = atomic_add(pool, &pool->next, chunk);
    size_t begin, end;
    long count;
    if (u >= units) return;
    count = units - u < chunk ? units - u : chunk;
    begin = (size_t)u * pool->grain;
    end = (size_t)(u + count) * pool->grain;
    if (end > pool->n) end = pool->n;
    pool->fn(pool->context, begin, end);
    if (atomic_add(pool, &pool->finished, count) + count == units) {
      lock(&pool->lock);
      if (pool->waiting) wake_one(&pool->done);
      unlock(&pool->lock);
    }
  }
}

static void work(rayo_pool *pool) {
  long seen = 0;
  for (;;) {
    long g = atomic_get(pool, &pool->generation);
    if (g == seen || (g & 1)) {
      if (!spin_while(pool, &pool->generation, g, WORKER_SPIN_US)) {
        lock(&pool->lock);
        pool->sleepers++;
        while (atomic_get(pool, &pool->generation) == g && !atomic_get(pool, &pool->quit)) {
          wait_on(&pool->wake, &pool->lock);
        }
        pool->sleepers--;
        unlock(&pool->lock);
      }
      if (atomic_get(pool, &pool->quit)) return;
      continue;
    }
    if (atomic_get(pool, &pool->quit)) return;
    /* The job of generation g, if the caller has not moved on: it waits
     * for `active` to drop to 0 after making the generation odd. */
    atomic_add(pool, &pool->active, 1);
    if (atomic_get(pool, &pool->generation) == g) take_parts(pool);
    atomic_add(pool, &pool->active, -1);
    seen = g;
  }
}

typedef struct worker_arg {
  rayo_pool *pool;
} worker_arg;

#if defined(RAYO_THREADS_POSIX)
static void *worker_main(void *arg) {
  worker_arg a = *(worker_arg *)arg;
  free(arg);
  work(a.pool);
  return NULL;
}
#else
static DWORD WINAPI worker_main(LPVOID arg) {
  worker_arg a = *(worker_arg *)arg;
  free(arg);
  work(a.pool);
  return 0;
}
#endif

static rayo_bool start(rayo_pool *pool, int index) {
  worker_arg *arg = malloc(sizeof *arg);
  if (!arg) return RAYO_FALSE;
  arg->pool = pool;
#if defined(RAYO_THREADS_POSIX)
  if (pthread_create(&pool->workers[index - 1], NULL, worker_main, arg) != 0) {
    free(arg);
    return RAYO_FALSE;
  }
#else
  pool->workers[index - 1] = CreateThread(NULL, 0, worker_main, arg, 0, NULL);
  if (!pool->workers[index - 1]) {
    free(arg);
    return RAYO_FALSE;
  }
#endif
  return RAYO_TRUE;
}

static void stop(rayo_pool *pool, int started) {
  int i;
  lock(&pool->lock);
  atomic_set(pool, &pool->quit, 1);
  wake_all(&pool->wake);
  unlock(&pool->lock);
  for (i = 0; i < started; i++) {
#if defined(RAYO_THREADS_POSIX)
    pthread_join(pool->workers[i], NULL);
#else
    WaitForSingleObject(pool->workers[i], INFINITE);
    CloseHandle(pool->workers[i]);
#endif
  }
#if defined(RAYO_THREADS_POSIX)
  pthread_cond_destroy(&pool->wake);
  pthread_cond_destroy(&pool->done);
  pthread_mutex_destroy(&pool->lock);
#if !defined(RAYO_ATOMIC_BUILTINS)
  pthread_mutex_destroy(&pool->atomic);
#endif
#else
  DeleteCriticalSection(&pool->lock);
#endif
  free(pool->workers);
  free(pool);
}

rayo_pool *rayo_pool_create(int threads) {
  rayo_pool *pool;
  int i;
  if (threads <= 1) return NULL;
  if (threads > 256) threads = 256;
  pool = calloc(1, sizeof *pool);
  if (!pool) return NULL;
  pool->workers = calloc((size_t)threads, sizeof *pool->workers);
  if (!pool->workers) {
    free(pool);
    return NULL;
  }
  pool->threads = threads;
#if defined(RAYO_THREADS_POSIX)
  if (pthread_mutex_init(&pool->lock, NULL) != 0) {
    free(pool->workers);
    free(pool);
    return NULL;
  }
#if !defined(RAYO_ATOMIC_BUILTINS)
  pthread_mutex_init(&pool->atomic, NULL);
#endif
  pthread_cond_init(&pool->wake, NULL);
  pthread_cond_init(&pool->done, NULL);
#else
  InitializeCriticalSection(&pool->lock);
  InitializeConditionVariable(&pool->wake);
  InitializeConditionVariable(&pool->done);
#endif
  for (i = 1; i < threads; i++) {
    if (!start(pool, i)) {
      stop(pool, i - 1);
      return NULL;
    }
  }
  return pool;
}

void rayo_pool_destroy(rayo_pool *pool) {
  if (pool) stop(pool, pool->threads - 1);
}

int rayo_pool_threads(const rayo_pool *pool) {
  return pool ? pool->threads : 1;
}

void rayo_pool_run(rayo_pool *pool, size_t n, size_t grain, rayo_task_fn *fn, void *context) {
  long units;
  if (grain == 0) grain = 1;
  if (!pool || n <= grain) {
    if (n) fn(context, 0, n);
    return;
  }
  units = (long)((n + grain - 1) / grain);
  /* No worker may still be reading the last job while this one is
   * written: the odd generation turns away any that comes late. */
  atomic_add(pool, &pool->generation, 1);
  while (atomic_get(pool, &pool->active) != 0) relax();
  pool->fn = fn;
  pool->context = context;
  pool->n = n;
  pool->grain = grain;
  pool->units = units;
  /* About four takes a thread: few enough to cost nothing, enough to even
   * out cores of different speeds. */
  pool->chunk = units / (4L * pool->threads);
  if (pool->chunk < 1) pool->chunk = 1;
  atomic_set(pool, &pool->next, 0);
  atomic_set(pool, &pool->finished, 0);
  atomic_add(pool, &pool->generation, 1);
  lock(&pool->lock);
  if (pool->sleepers > 0) wake_all(&pool->wake);
  unlock(&pool->lock);
  take_parts(pool);
  while (atomic_get(pool, &pool->finished) != units) {
    const long f = atomic_get(pool, &pool->finished);
    if (spin_while(pool, &pool->finished, f, CALLER_SPIN_US)) continue;
    lock(&pool->lock);
    pool->waiting = RAYO_TRUE;
    while (atomic_get(pool, &pool->finished) != units) wait_on(&pool->done, &pool->lock);
    pool->waiting = RAYO_FALSE;
    unlock(&pool->lock);
  }
}

#else /* one thread */

rayo_pool *rayo_pool_create(int threads) {
  (void)threads;
  return NULL;
}

void rayo_pool_destroy(rayo_pool *pool) {
  (void)pool;
}

int rayo_pool_threads(const rayo_pool *pool) {
  (void)pool;
  return 1;
}

void rayo_pool_run(rayo_pool *pool, size_t n, size_t grain, rayo_task_fn *fn, void *context) {
  (void)pool;
  (void)grain;
  if (n) fn(context, 0, n);
}

#endif
