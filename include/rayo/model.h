/* rayo/model.h: a loaded model (weights and graph). */
#ifndef RAYO_MODEL_H
#define RAYO_MODEL_H

#include <stddef.h>

#include "rayo/error.h"
#include "rayo/export.h"
#include "rayo/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rayo_model rayo_model;

/* Memory a model or a session holds, in bytes. */
typedef struct rayo_memory_info {
  size_t weights;  /* the model's file (copied, or mapped), and packed weights */
  size_t arena;    /* a session's activations and scratch, planned once */
  size_t overhead; /* bookkeeping: graph, plans, tables */
  size_t states;   /* of `arena`: a session's states (a decoder's key/value cache), in full */
} rayo_memory_info;

/* What the whole process holds in memory now, in bytes, as the system
 * counts it. */
typedef struct rayo_process_memory_info {
  /* Resident memory of the process's own (anonymous: the heap, a session's
   * arena and cache, stacks), which the system must keep while it lives.
   * Linux and Android: smaps_rollup's Anonymous (or statm's resident less
   * shared, on kernels before 4.14); macOS: task_info's internal. */
  size_t private_bytes;
  /* Resident pages of mapped files (the weights, the program itself), which
   * the system can drop under pressure and read again. */
  size_t file_bytes;
  size_t resident_bytes;      /* both */
  size_t peak_resident_bytes; /* the most resident so far (0 if unknown) */
} rayo_process_memory_info;

/* Fills `out` for this process. RAYO_ERR_UNSUPPORTED (and zeros) where the
 * system does not say (Windows and plain C89 builds, for now). */
RAYO_API rayo_status rayo_process_memory(rayo_process_memory_info *out);

/* Reads a model from the `size` bytes at `data`, a plain-weights (PCWT) file
 * with a graph (docs/FORMAT.md). rayo copies the bytes it needs: the caller
 * may free `data` as soon as this returns. On success `*out` is a new model
 * the caller owns and frees with rayo_model_destroy. The bytes are
 * untrusted: every size and offset is checked. */
RAYO_API rayo_status rayo_model_create(const void *data, size_t size, rayo_model **out,
                                       rayo_message *msg);

/* How a mapped model's file will be read, a hint to the system's
 * read-ahead (madvise): RAYO_ACCESS_DEFAULT is rayo's choice, measured on a
 * decoder (docs/RUNNING-SLMS.md). No hint where the file is read into
 * memory or the system takes none. */
typedef enum rayo_access {
  RAYO_ACCESS_DEFAULT = 0,
  RAYO_ACCESS_NORMAL,     /* the system's own read-ahead */
  RAYO_ACCESS_SEQUENTIAL, /* front to back (MADV_SEQUENTIAL) */
  RAYO_ACCESS_WILLNEED,   /* all of it, soon: read ahead now (MADV_WILLNEED) */
  RAYO_ACCESS_RANDOM      /* no read-ahead (MADV_RANDOM) */
} rayo_access;

/* Options for rayo_model_create_with. Zero them (memset) before setting
 * what you need, so options added later keep their defaults. */
typedef struct rayo_model_options {
  /* Ops of the program's own (rayo/op.h) the graph may name, besides the
   * built-in ones; their names must differ from those. The ops must stay
   * valid while the model lives. */
  const struct rayo_custom_op *const *ops;
  int op_count;
  /* rayo_model_load only: 1 reads the file into memory of rayo's instead
   * of mapping it. */
  int read_file;
  /* rayo_model_load only: a rayo_access hint for the mapped file. */
  int access;
  /* What the file must match (rayo/verify.h), checked before rayo reads
   * any of it: a mismatch is RAYO_ERR_INTEGRITY. NULL: no check. */
  const struct rayo_integrity *integrity;
} rayo_model_options;

/* rayo_model_create with `options` (NULL: the defaults). */
RAYO_API rayo_status rayo_model_create_with(const void *data, size_t size,
                                            const rayo_model_options *options, rayo_model **out,
                                            rayo_message *msg);

/* Loads the model file at `path` (`options` may be NULL). The file is
 * mapped (mmap, MapViewOfFile) and its weights are used where they are, with
 * no copy: the system reads their pages as they are touched and may drop
 * them again under memory pressure, so a model costs what a run touches.
 * Where the system cannot map files (or with options->read_file), the file
 * is read into memory instead. The file must not change while the model
 * lives. */
RAYO_API rayo_status rayo_model_load(const char *path, const rayo_model_options *options,
                                     rayo_model **out, rayo_message *msg);

/* Frees `model` (NULL is fine). Destroy its sessions first. */
RAYO_API void rayo_model_destroy(rayo_model *model);

/* The header's "model" name, such as "example-classifier". Owned by the model. */
RAYO_API const char *rayo_model_name(const rayo_model *model);

/* The header's "metadata" object as JSON text ("{}" when absent). Owned by
 * the model. */
RAYO_API const char *rayo_model_metadata_json(const rayo_model *model);

RAYO_API int rayo_model_input_count(const rayo_model *model);
RAYO_API int rayo_model_output_count(const rayo_model *model);

/* The name, dtype and largest shape of input or output `index`; `data` is
 * NULL. RAYO_ERR_INVALID_ARGUMENT when `index` is out of range. */
RAYO_API rayo_status rayo_model_input_info(const rayo_model *model, int index,
                                           rayo_tensor_view *out);
RAYO_API rayo_status rayo_model_output_info(const rayo_model *model, int index,
                                            rayo_tensor_view *out);

RAYO_API void rayo_model_memory(const rayo_model *model, rayo_memory_info *out);

#ifdef __cplusplus
}
#endif

#endif
