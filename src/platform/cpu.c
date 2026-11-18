/* How many cores are worth using. On Android and Linux, the cores whose
 * capacity (or highest clock) in /sys is at least half the fastest one's:
 * a phone's big and prime cores, not its little ones, which are slower
 * and share the same memory bandwidth. On macOS, the performance cores
 * (hw.perflevel0.physicalcpu), else the physical ones. Else the cores
 * online, else 1. */
#if !defined(RAYO_NO_PLATFORM) && defined(__APPLE__)
#define RAYO_CPU_APPLE 1
#define _DARWIN_C_SOURCE 1
#elif !defined(RAYO_NO_PLATFORM) && defined(__unix__)
#define RAYO_CPU_POSIX 1
#define _POSIX_C_SOURCE 200112L
#elif !defined(RAYO_NO_PLATFORM) && defined(_WIN32)
#define RAYO_CPU_WIN32 1
#endif

#include <stdio.h>

#include "platform/platform.h"

#if defined(RAYO_CPU_APPLE)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(RAYO_CPU_POSIX)
#include <unistd.h>
#elif defined(RAYO_CPU_WIN32)
#include <windows.h>
#endif

#if defined(RAYO_CPU_POSIX)
/* A whole number from the file `path`, or -1. */
static long read_number(const char *path) {
  FILE *f = fopen(path, "r");
  long v = -1;
  if (!f) return -1;
  if (fscanf(f, "%ld", &v) != 1) v = -1;
  fclose(f);
  return v;
}

/* The cores whose `what` in /sys is at least half the largest; 0 if none
 * says. */
static int fast_by(const char *what, int cores) {
  char path[96];
  long best = 0, v;
  int i, n = 0;
  for (i = 0; i < cores && i < 1024; i++) {
    sprintf(path, "/sys/devices/system/cpu/cpu%d/%s", i, what);
    v = read_number(path);
    if (v > best) best = v;
  }
  if (best <= 0) return 0;
  for (i = 0; i < cores && i < 1024; i++) {
    sprintf(path, "/sys/devices/system/cpu/cpu%d/%s", i, what);
    v = read_number(path);
    if (v >= 0 && 2 * v >= best) n++;
  }
  return n;
}
#endif

RAYO_API int rayo_cpu_fast_cores(void) {
#if defined(RAYO_CPU_APPLE)
  int v = 0;
  size_t size = sizeof v;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &size, NULL, 0) == 0 && v > 0) return v;
  size = sizeof v;
  if (sysctlbyname("hw.physicalcpu", &v, &size, NULL, 0) == 0 && v > 0) return v;
  return 1;
#elif defined(RAYO_CPU_POSIX)
  const long online = sysconf(_SC_NPROCESSORS_ONLN);
  const int cores = online > 0 ? (int)(online < 1024 ? online : 1024) : 1;
  int n = fast_by("cpu_capacity", cores);
  if (n == 0) n = fast_by("cpufreq/cpuinfo_max_freq", cores);
  return n > 0 ? n : cores;
#elif defined(RAYO_CPU_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 1;
#else
  return 1;
#endif
}
