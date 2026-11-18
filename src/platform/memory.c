/* Memory from the system in whole pages (a session's arena), handing pages
 * back when they are zeroed, hints about how a mapped file is read, and
 * what the process holds: mmap and /proc on Linux and Android, mmap and
 * task_info on macOS, VirtualAlloc on Windows, the C library elsewhere. */
#if !defined(RAYO_NO_PLATFORM) && defined(__APPLE__)
#define RAYO_MEM_APPLE 1
#define RAYO_MEM_MMAP 1
#define _DARWIN_C_SOURCE 1
#elif !defined(RAYO_NO_PLATFORM) && defined(__unix__)
#define RAYO_MEM_LINUX 1
#define RAYO_MEM_MMAP 1
#define _DEFAULT_SOURCE 1 /* MAP_ANONYMOUS and madvise, under -std=c89 */
#define _BSD_SOURCE 1
#elif !defined(RAYO_NO_PLATFORM) && defined(_WIN32)
#define RAYO_MEM_WIN32 1
#endif

#include <stdio.h>
#include <string.h>

#include "core/alloc.h"
#include "core/compat.h"
#include "platform/platform.h"
#include "rayo/model.h"

#if defined(RAYO_MEM_MMAP)
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif
#if defined(RAYO_MEM_APPLE)
#include <mach/mach.h>
#endif
#if defined(RAYO_MEM_WIN32)
#include <windows.h>
#endif

#if defined(RAYO_MEM_MMAP) || defined(RAYO_MEM_WIN32)
static size_t page_size(void) {
#if defined(RAYO_MEM_MMAP)
  const long n = sysconf(_SC_PAGESIZE);
  return n > 0 ? (size_t)n : 4096;
#else
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize > 0 ? (size_t)info.dwPageSize : 4096;
#endif
}

/* `size` rounded up to whole pages; 0 on overflow. */
static size_t whole_pages(size_t size) {
  const size_t page = page_size();
  return size > RAYO_SIZE_MAX - page ? 0 : (size + page - 1) / page * page;
}
#endif

void *rayo_pages_alloc(size_t size) {
#if defined(RAYO_MEM_MMAP)
  const size_t n = whole_pages(size);
  void *p;
  if (size == 0 || n == 0) return NULL;
  p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return p == MAP_FAILED ? NULL : p;
#elif defined(RAYO_MEM_WIN32)
  const size_t n = whole_pages(size);
  if (size == 0 || n == 0) return NULL;
  return VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
  return rayo_aligned_alloc(size);
#endif
}

void rayo_pages_free(void *p, size_t size) {
  if (!p) return;
#if defined(RAYO_MEM_MMAP)
  munmap(p, whole_pages(size));
#elif defined(RAYO_MEM_WIN32)
  (void)size;
  VirtualFree(p, 0, MEM_RELEASE);
#else
  (void)size;
  rayo_aligned_free(p);
#endif
}

void rayo_pages_zero(void *p, size_t size) {
  unsigned char *b = (unsigned char *)p;
#if defined(RAYO_MEM_MMAP) || defined(RAYO_MEM_WIN32)
  const size_t page = page_size();
  unsigned char *first = (unsigned char *)rayo_align_pointer(b, page);
  size_t whole;
  if (!p || size == 0) return;
  if ((size_t)(first - b) >= size || size - (size_t)(first - b) < page) {
    memset(b, 0, size);
    return;
  }
  whole = (size - (size_t)(first - b)) / page * page;
  memset(b, 0, (size_t)(first - b));
  memset(first + whole, 0, size - (size_t)(first - b) - whole);
#if defined(RAYO_MEM_MMAP)
  /* Fresh zero pages over the old ones: the old ones go back to the system
   * (MADV_DONTNEED would on Linux, not on macOS). */
  if (mmap(first, whole, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) ==
      MAP_FAILED) {
    memset(first, 0, whole);
  }
#else
  if (!VirtualFree(first, whole, MEM_DECOMMIT) ||
      !VirtualAlloc(first, whole, MEM_COMMIT, PAGE_READWRITE)) {
    memset(first, 0, whole);
  }
#endif
#else
  if (b) memset(b, 0, size);
#endif
}

void rayo_map_advise(const rayo_file_map *map, rayo_advice advice) {
#if defined(RAYO_MEM_MMAP)
  int how = MADV_NORMAL;
  if (!map || map->kind != RAYO_MAP_MMAP) return;
  switch (advice) {
    case RAYO_ADVICE_SEQUENTIAL:
      how = MADV_SEQUENTIAL;
      break;
    case RAYO_ADVICE_WILLNEED:
      how = MADV_WILLNEED;
      break;
    case RAYO_ADVICE_RANDOM:
      how = MADV_RANDOM;
      break;
    case RAYO_ADVICE_NORMAL:
      break;
  }
  /* A hint: whether the system takes it changes no result. */
  (void)madvise((void *)map->data, map->size, how);
#else
  (void)map;
  (void)advice;
#endif
}

#if defined(RAYO_MEM_LINUX)
/* The kB value of each wanted field of a /proc file ("Rss:  123 kB"). */
static int read_fields(const char *path, const char *const *names, size_t *values, int count) {
  char line[256];
  int found = 0, i;
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  while (fgets(line, sizeof line, f)) {
    for (i = 0; i < count; i++) {
      const size_t n = strlen(names[i]);
      unsigned long kb;
      if (strncmp(line, names[i], n) == 0 && sscanf(line + n, "%lu", &kb) == 1) {
        values[i] = (size_t)kb * 1024;
        found++;
      }
    }
  }
  fclose(f);
  return found;
}
#endif

RAYO_API rayo_status rayo_process_memory(rayo_process_memory_info *out) {
  if (!out) return RAYO_ERR_INVALID_ARGUMENT;
  memset(out, 0, sizeof *out);
#if defined(RAYO_MEM_LINUX)
  {
    static const char *const rollup[3] = {"Rss:", "Anonymous:", "Swap:"};
    static const char *const status[1] = {"VmHWM:"};
    size_t v[3] = {0, 0, 0}, hwm = 0;
    if (read_fields("/proc/self/smaps_rollup", rollup, v, 3) >= 2) {
      out->resident_bytes = v[0];
      out->private_bytes = v[1];
      out->file_bytes = v[0] > v[1] ? v[0] - v[1] : 0;
    } else {
      /* Kernels before 4.14 (older Android phones): statm's resident and
       * shared (file-backed) pages. */
      unsigned long size, resident, shared;
      FILE *f = fopen("/proc/self/statm", "r");
      const size_t page = page_size();
      if (!f) return RAYO_ERR_UNSUPPORTED;
      if (fscanf(f, "%lu %lu %lu", &size, &resident, &shared) != 3) {
        fclose(f);
        return RAYO_ERR_UNSUPPORTED;
      }
      fclose(f);
      out->resident_bytes = (size_t)resident * page;
      out->file_bytes = (size_t)shared * page;
      out->private_bytes = out->resident_bytes - out->file_bytes;
    }
    if (read_fields("/proc/self/status", status, &hwm, 1) == 1) out->peak_resident_bytes = hwm;
    return RAYO_OK;
  }
#elif defined(RAYO_MEM_APPLE)
  {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
      return RAYO_ERR_UNSUPPORTED;
    }
    out->private_bytes = (size_t)info.internal;
    out->file_bytes = (size_t)info.external;
    out->resident_bytes = (size_t)info.resident_size;
    out->peak_resident_bytes = (size_t)info.resident_size_peak;
    return RAYO_OK;
  }
#else
  return RAYO_ERR_UNSUPPORTED;
#endif
}
