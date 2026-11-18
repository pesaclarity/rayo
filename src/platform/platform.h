/* What rayo asks of the operating system beyond C89: a file mapped into
 * memory, threads, and how many CPU cores are worth using. POSIX
 * (Linux, Android, macOS, iOS) and Win32 each have their own version;
 * anything else, or a build with RAYO_NO_PLATFORM defined, gets plain C89:
 * the file read with fread and one thread. Only the files of this directory
 * use the system's headers. */
#ifndef RAYO_PLATFORM_PLATFORM_H
#define RAYO_PLATFORM_PLATFORM_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/types.h"

typedef enum rayo_map_kind {
  RAYO_MAP_NONE,
  RAYO_MAP_HEAP, /* read into memory of ours (rayo_aligned_alloc) */
  RAYO_MAP_MMAP, /* POSIX mmap, read-only */
  RAYO_MAP_WIN32 /* MapViewOfFile, read-only */
} rayo_map_kind;

/* A file's bytes, read-only. Mapped pages are read from the file as they
 * are touched and can be dropped again under memory pressure, so weights
 * that are mapped cost no copy and count only as they are used. */
typedef struct rayo_file_map {
  const rayo_u8 *data; /* page-aligned when mapped, RAYO_ALIGN-aligned when read */
  size_t size;
  rayo_map_kind kind;
  void *handle; /* Win32: the mapping object */
} rayo_file_map;

/* Maps (or, with `read` set or no mapping here, reads) the file at `path`. */
rayo_status rayo_map_file(const char *path, rayo_bool read, rayo_file_map *out, rayo_message *msg);

/* Releases `map` (a zeroed one is fine). */
void rayo_unmap_file(rayo_file_map *map);

/* A pool of worker threads. rayo_pool_run splits [0, n) into one part per
 * thread, the caller's thread taking the first, and returns when all are
 * done. The parts depend only on n, `grain` and the pool's size, and each
 * index is computed by exactly one thread, so results do not depend on the
 * timing. */
typedef struct rayo_pool rayo_pool;

typedef void rayo_task_fn(void *context, size_t begin, size_t end);

/* A pool of `threads` threads in all (the caller's included): NULL for one
 * thread, or when this build has none, or when they cannot be made. */
rayo_pool *rayo_pool_create(int threads);

void rayo_pool_destroy(rayo_pool *pool);

/* The threads a pool runs on (1 for NULL). */
int rayo_pool_threads(const rayo_pool *pool);

/* fn(context, begin, end) over [0, n), in parts that are multiples of
 * `grain` (but the last); with a NULL pool, fn(context, 0, n). */
void rayo_pool_run(rayo_pool *pool, size_t n, size_t grain, rayo_task_fn *fn, void *context);

/* The cores worth running on: the performance ("big") cores where the
 * system says which they are (Android and Linux: cpu_capacity or the
 * highest clock in /sys; macOS: hw.perflevel0 or the physical cores), else
 * the cores online; at least 1. */
RAYO_API int rayo_cpu_fast_cores(void);

#endif
