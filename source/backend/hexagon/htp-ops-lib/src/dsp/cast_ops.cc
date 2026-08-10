#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/worker_pool.h"

extern "C" {

typedef enum {
  HTP_OPS_CAST_FP16_TO_FP16 = 0,
  HTP_OPS_CAST_INT32_TO_INT32 = 1,
  HTP_OPS_CAST_INT32_TO_FP16 = 2,
  HTP_OPS_CAST_FP16_TO_INT32 = 3,
} HtpOpsCastType;

#define HTP_OPS_CAST_MT_MIN_ELEMS 2048

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int task_id;
  int n_tasks;
  int size;
  int castType;
  uint8_t* dst;
  const uint8_t* src;
} HtpOpsCastTaskState;

static inline void htp_ops_cast_compute_range(uint8_t* dst, const uint8_t* src,
                                              int start, int end, int castType) {
  switch (castType) {
    case HTP_OPS_CAST_FP16_TO_FP16: {
      memcpy(dst + start * (int)sizeof(__fp16), src + start * (int)sizeof(__fp16),
             (end - start) * sizeof(__fp16));
      break;
    }
    case HTP_OPS_CAST_INT32_TO_INT32: {
      memcpy(dst + start * (int)sizeof(int32_t), src + start * (int)sizeof(int32_t),
             (end - start) * sizeof(int32_t));
      break;
    }
    case HTP_OPS_CAST_INT32_TO_FP16: {
      const int32_t* src_i32 = (const int32_t*)src;
      __fp16* dst_fp16 = (__fp16*)dst;
      for (int i = start; i < end; ++i) {
        dst_fp16[i] = (__fp16)src_i32[i];
      }
      break;
    }
    case HTP_OPS_CAST_FP16_TO_INT32: {
      const __fp16* src_fp16 = (const __fp16*)src;
      int32_t* dst_i32 = (int32_t*)dst;
      for (int i = start; i < end; ++i) {
        dst_i32[i] = (int32_t)src_fp16[i];
      }
      break;
    }
    default:
      break;
  }
}

static void htp_ops_cast_worker_loop(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsCastTaskState* state = (HtpOpsCastTaskState*)data;
  const int total_blocks = (state->size + HTP_OPS_CAST_MT_MIN_ELEMS - 1) / HTP_OPS_CAST_MT_MIN_ELEMS;
  const int blocks_per_task = (total_blocks + state->n_tasks - 1) / state->n_tasks;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if ((int)task_id >= state->n_tasks) {
      break;
    }
    const int start = (int)task_id * blocks_per_task * HTP_OPS_CAST_MT_MIN_ELEMS;
    if (start >= state->size) {
      break;
    }
    int end = start + blocks_per_task * HTP_OPS_CAST_MT_MIN_ELEMS;
    if (end > state->size) {
      end = state->size;
    }
    htp_ops_cast_compute_range(state->dst, state->src, start, end, state->castType);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline int htp_ops_cast_pick_task_count(int size) {
  unsigned int worker_cap = g_max_num_workers;
  if (worker_cap <= 1) {
    return 1;
  }
  int task_count = (size + HTP_OPS_CAST_MT_MIN_ELEMS - 1) / HTP_OPS_CAST_MT_MIN_ELEMS;
  if (task_count < 2) {
    return 1;
  }
  if (task_count > (int)worker_cap) {
    task_count = (int)worker_cap;
  }
  return task_count;
}

AEEResult htp_ops_cast(uint8_t* dst, const uint8_t* src, int32_t size, int32_t castType) {
  if (dst == NULL || src == NULL || size < 0) {
    return -1;
  }
  if (size == 0) {
    return 0;
  }
  if (castType < HTP_OPS_CAST_FP16_TO_FP16 || castType > HTP_OPS_CAST_FP16_TO_INT32) {
    return -1;
  }

  const int n_tasks = htp_ops_cast_pick_task_count(size);
  if (n_tasks <= 1) {
    htp_ops_cast_compute_range(dst, src, 0, size, castType);
    return 0;
  }

  HtpOpsCastTaskState state = {};
  state.task_id = 0;
  state.n_tasks = n_tasks;
  state.size = size;
  state.castType = castType;
  state.dst = dst;
  state.src = src;

  worker_pool_job_t job;
  job.fptr = htp_ops_cast_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return 0;
}

}  // extern "C"
