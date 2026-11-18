/* A pool of worker threads: pthreads on POSIX systems, Win32 threads on
 * Windows, none otherwise (one thread). The workers sleep on a condition
 * variable between runs, so an idle pool uses no CPU. */
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

struct rayo_pool {
  int threads; /* in all, the caller's included */
  rayo_thread *workers;
  rayo_mutex lock;
  rayo_cond wake; /* a new job, or quit */
  rayo_cond done; /* the last part of a job finished */
  /* The job, under `lock`. */
  unsigned long generation;
  rayo_task_fn *fn;
  void *context;
  size_t n, grain;
  int pending; /* workers still on this job */
  rayo_bool quit;
};

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

/* Part `part` of `parts` of [0, n), cut at multiples of `grain`. */
static void part_of(size_t n, size_t grain, int parts, int part, size_t *begin, size_t *end) {
  const size_t units = (n + grain - 1) / grain;
  const size_t per = units / (size_t)parts, extra = units % (size_t)parts;
  const size_t p = (size_t)part;
  const size_t first = p * per + (p < extra ? p : extra);
  const size_t count = per + (p < extra ? 1 : 0);
  *begin = first * grain;
  *end = (first + count) * grain;
  if (*begin > n) *begin = n;
  if (*end > n) *end = n;
}

typedef struct worker_arg {
  rayo_pool *pool;
  int index;
} worker_arg;

static void work(rayo_pool *pool, int index) {
  unsigned long seen = 0;
  for (;;) {
    rayo_task_fn *fn;
    void *context;
    size_t n, grain, begin, end;
    lock(&pool->lock);
    while (pool->generation == seen && !pool->quit) wait_on(&pool->wake, &pool->lock);
    if (pool->quit) {
      unlock(&pool->lock);
      return;
    }
    seen = pool->generation;
    fn = pool->fn;
    context = pool->context;
    n = pool->n;
    grain = pool->grain;
    unlock(&pool->lock);
    part_of(n, grain, pool->threads, index, &begin, &end);
    if (begin < end) fn(context, begin, end);
    lock(&pool->lock);
    if (--pool->pending == 0) wake_one(&pool->done);
    unlock(&pool->lock);
  }
}

#if defined(RAYO_THREADS_POSIX)
static void *worker_main(void *arg) {
  worker_arg a = *(worker_arg *)arg;
  free(arg);
  work(a.pool, a.index);
  return NULL;
}
#else
static DWORD WINAPI worker_main(LPVOID arg) {
  worker_arg a = *(worker_arg *)arg;
  free(arg);
  work(a.pool, a.index);
  return 0;
}
#endif

static rayo_bool start(rayo_pool *pool, int index) {
  worker_arg *arg = malloc(sizeof *arg);
  if (!arg) return RAYO_FALSE;
  arg->pool = pool;
  arg->index = index;
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
  pool->quit = RAYO_TRUE;
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
  size_t begin, end;
  if (grain == 0) grain = 1;
  if (!pool || n <= grain) {
    if (n) fn(context, 0, n);
    return;
  }
  lock(&pool->lock);
  pool->fn = fn;
  pool->context = context;
  pool->n = n;
  pool->grain = grain;
  pool->pending = pool->threads - 1;
  pool->generation++;
  wake_all(&pool->wake);
  unlock(&pool->lock);
  part_of(n, grain, pool->threads, 0, &begin, &end);
  if (begin < end) fn(context, begin, end);
  lock(&pool->lock);
  while (pool->pending > 0) wait_on(&pool->done, &pool->lock);
  unlock(&pool->lock);
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
