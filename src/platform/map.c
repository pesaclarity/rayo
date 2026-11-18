/* Mapping a file: mmap on POSIX systems, MapViewOfFile on Windows, fread
 * otherwise (or when asked). */
#if !defined(RAYO_NO_PLATFORM) && (defined(__unix__) || defined(__APPLE__))
#define RAYO_MAP_POSIX 1
#define _POSIX_C_SOURCE 200112L
#elif !defined(RAYO_NO_PLATFORM) && defined(_WIN32)
#define RAYO_MAP_WINDOWS 1
#endif

#include <stdio.h>
#include <string.h>

#include "core/alloc.h"
#include "core/message.h"
#include "platform/platform.h"

#if defined(RAYO_MAP_POSIX)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(RAYO_MAP_WINDOWS)
#include <windows.h>
#endif

static rayo_status read_whole(const char *path, rayo_file_map *out, rayo_message *msg) {
  FILE *f = fopen(path, "rb");
  long n;
  rayo_u8 *p;
  if (!f) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot open %s", path);
  if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot read %s", path);
  }
  if (n == 0) {
    fclose(f);
    return rayo_fail(msg, RAYO_ERR_FORMAT, "an empty file");
  }
  p = rayo_aligned_alloc((size_t)n);
  if (!p) {
    fclose(f);
    return rayo_fail(msg, RAYO_ERR_OUT_OF_MEMORY, "out of memory for %lu bytes", (unsigned long)n);
  }
  if (fread(p, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    rayo_aligned_free(p);
    return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot read %s", path);
  }
  fclose(f);
  out->data = p;
  out->size = (size_t)n;
  out->kind = RAYO_MAP_HEAP;
  return RAYO_OK;
}

rayo_status rayo_map_file(const char *path, rayo_bool read, rayo_file_map *out, rayo_message *msg) {
  memset(out, 0, sizeof *out);
  if (!path) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "null argument");
#if defined(RAYO_MAP_POSIX)
  if (!read) {
    struct stat st;
    void *p;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot open %s", path);
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || (double)st.st_size > (double)RAYO_SIZE_MAX) {
      close(fd);
      return rayo_fail(msg, RAYO_ERR_FORMAT, "%s: an empty file or not a file", path);
    }
    p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* the mapping keeps the file */
    if (p != MAP_FAILED) {
      out->data = (const rayo_u8 *)p;
      out->size = (size_t)st.st_size;
      out->kind = RAYO_MAP_MMAP;
      return RAYO_OK;
    }
  }
#elif defined(RAYO_MAP_WINDOWS)
  if (!read) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    LARGE_INTEGER size;
    if (file == INVALID_HANDLE_VALUE) {
      return rayo_fail(msg, RAYO_ERR_INVALID_ARGUMENT, "cannot open %s", path);
    }
    if (GetFileSizeEx(file, &size) && size.QuadPart > 0 &&
        (double)size.QuadPart <= (double)RAYO_SIZE_MAX) {
      HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
      void *p = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : NULL;
      if (p) {
        CloseHandle(file);
        out->data = (const rayo_u8 *)p;
        out->size = (size_t)size.QuadPart;
        out->kind = RAYO_MAP_WIN32;
        out->handle = mapping;
        return RAYO_OK;
      }
      if (mapping) CloseHandle(mapping);
    }
    CloseHandle(file);
  }
#endif
  (void)read;
  return read_whole(path, out, msg);
}

void rayo_unmap_file(rayo_file_map *map) {
  if (!map) return;
  switch (map->kind) {
    case RAYO_MAP_HEAP:
      rayo_aligned_free((void *)map->data);
      break;
    case RAYO_MAP_MMAP:
#if defined(RAYO_MAP_POSIX)
      munmap((void *)map->data, map->size);
#endif
      break;
    case RAYO_MAP_WIN32:
#if defined(RAYO_MAP_WINDOWS)
      UnmapViewOfFile(map->data);
      CloseHandle((HANDLE)map->handle);
#endif
      break;
    case RAYO_MAP_NONE:
      break;
  }
  memset(map, 0, sizeof *map);
}
