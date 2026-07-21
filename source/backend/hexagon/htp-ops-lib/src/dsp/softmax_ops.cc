#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_math.h"
#include "dsp/worker_pool.h"
#include "htp_ops.h"

extern "C" {

static inline uint16_t htp_ops_softmax_fp16_max_bits(const __fp16* src, int32_t size) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16* ptr = src;
  int i = 0;

  __fp16 bestScalar = src[0];
  HVX_Vector best_v = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)ptr);
    best_v = Q6_Vhf_vmax_VhfVhf(best_v, v);
    ptr += vec_len;
  }
  if (i < size && size - i <= vec_len) {
    uint16_t tail[vec_len] __attribute__((aligned(128)));
    for (int j = 0; j < vec_len; ++j) {
      tail[j] = 0xfc00;
    }
    memcpy(tail, src + i, (size_t)(size - i) * sizeof(__fp16));
    best_v = Q6_Vhf_vmax_VhfVhf(best_v, vmem(tail));
    i = size;
  }

  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 64));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 32));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 16));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 8));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 4));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 2));

  uint16_t tmp[vec_len] __attribute__((aligned(128)));
  vmemu((HVX_Vector*)tmp) = best_v;
  uint16_t bestBits = tmp[0];
  bestScalar = *(__fp16*)&bestBits;

  for (; i < size; ++i) {
    const __fp16 value = src[i];
    if (value > bestScalar) {
      bestScalar = value;
      bestBits = ((const uint16_t*)src)[i];
    }
  }
  return bestBits;
}

static inline void htp_ops_softmax_fp16_row_range(__fp16* dst, const __fp16* src, int32_t channel,
                                                  int32_t inside, int32_t begin, int32_t end) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int32_t vecEnd = begin + ((end - begin) / vec_len) * vec_len;
  const HVX_Vector log2e_v = Q6_Vh_vsplat_R(0x3dc5);
  const HVX_Vector two_v = Q6_Vh_vsplat_R(0x4000);
  int32_t i = begin;
  for (; i < vecEnd; i += vec_len) {
    HVX_Vector max_v = vmemu((const HVX_Vector*)(src + i));
    for (int32_t c = 1; c < channel; ++c) {
      HVX_Vector x = vmemu((const HVX_Vector*)(src + (size_t)c * inside + i));
      max_v = Q6_Vhf_vmax_VhfVhf(max_v, x);
    }

    HVX_Vector sum_v = Q6_V_vzero();
    for (int32_t c = 0; c < channel; ++c) {
      __fp16* dstPtr = dst + (size_t)c * inside + i;
      HVX_Vector x = vmemu((const HVX_Vector*)(src + (size_t)c * inside + i));
      HVX_Vector diff = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x, max_v));
      HVX_Vector expArg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(diff, log2e_v));
      HVX_Vector expValue = hvx_my_exp2_vhf(expArg);
      vmemu((HVX_Vector*)dstPtr) = expValue;
      sum_v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum_v, expValue));
    }

    HVX_Vector invSum = hvx_my_inv_vhf(sum_v);
    HVX_Vector dy = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sum_v, invSum));
    HVX_Vector twoMinusDy = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(two_v, dy));
    invSum = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(invSum, twoMinusDy));
    for (int32_t c = 0; c < channel; ++c) {
      __fp16* dstPtr = dst + (size_t)c * inside + i;
      HVX_Vector x = vmemu((const HVX_Vector*)dstPtr);
      HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x, invSum));
      vmemu((HVX_Vector*)dstPtr) = y;
    }
  }

  for (; i < end; ++i) {
    const __fp16* srcRow = src + i;
    __fp16* dstRow = dst + i;

    float maxValue = (float)srcRow[0];
    for (int32_t c = 1; c < channel; ++c) {
      const float value = (float)srcRow[c * inside];
      if (value > maxValue) {
        maxValue = value;
      }
    }

    float sumValue = 0.0f;
    for (int32_t c = 0; c < channel; ++c) {
      const float value = expf((float)srcRow[c * inside] - maxValue);
      dstRow[c * inside] = (__fp16)value;
      sumValue += value;
    }

    const float invSum = 1.0f / sumValue;
    for (int32_t c = 0; c < channel; ++c) {
      dstRow[c * inside] = (__fp16)((float)dstRow[c * inside] * invSum);
    }
  }
}

static inline void htp_ops_softmax_fp16_row(__fp16* dst, const __fp16* src, int32_t channel, int32_t inside) {
  htp_ops_softmax_fp16_row_range(dst, src, channel, inside, 0, inside);
}

typedef struct {
  __fp16* dst;
  const __fp16* src;
  int32_t channel;
  int32_t inside;
  worker_synctoken_t sync_ctx;
} HtpOpsSoftmaxTaskState;

typedef struct {
  HtpOpsSoftmaxTaskState* state;
  int32_t begin;
  int32_t end;
} HtpOpsSoftmaxTask;

static void htp_ops_softmax_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSoftmaxTask* task = (HtpOpsSoftmaxTask*)data;
  HtpOpsSoftmaxTaskState* state = task->state;
  htp_ops_softmax_fp16_row_range(state->dst, state->src, state->channel, state->inside,
                                 task->begin, task->end);
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static bool htp_ops_softmax_fp16_row_try_parallel(__fp16* dst, const __fp16* src,
                                                  int32_t channel, int32_t inside) {
  if (g_max_num_workers <= 1 || inside < 2 || inside <= 64 || (int64_t)channel * inside < 4096) {
    return false;
  }
  int task_count = (int)g_max_num_workers;
  if (task_count > inside) {
    task_count = inside;
  }
  if (task_count <= 1) {
    return false;
  }
  HtpOpsSoftmaxTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.channel = channel;
  state.inside = inside;
  HtpOpsSoftmaxTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsSoftmaxTask, task_count);
  worker_pool_job_t job;
  job.fptr = htp_ops_softmax_worker;
  worker_pool_synctoken_init(&state.sync_ctx, task_count);
  const int rows_per_task = (inside + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    const int begin = i * rows_per_task;
    int end = begin + rows_per_task;
    if (end > inside) {
      end = inside;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
  return true;
}

typedef struct {
  __fp16* dst;
  const __fp16* src;
  int32_t channel;
  int32_t inside;
  int32_t plane;
  worker_synctoken_t sync_ctx;
} HtpOpsSoftmaxOutsideTaskState;

typedef struct {
  HtpOpsSoftmaxOutsideTaskState* state;
  int32_t begin;
  int32_t end;
} HtpOpsSoftmaxOutsideTask;

static void htp_ops_softmax_outside_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSoftmaxOutsideTask* task = (HtpOpsSoftmaxOutsideTask*)data;
  HtpOpsSoftmaxOutsideTaskState* state = task->state;
  for (int32_t o = task->begin; o < task->end; ++o) {
    htp_ops_softmax_fp16_row_range(state->dst + (size_t)o * state->plane,
                                   state->src + (size_t)o * state->plane,
                                   state->channel, state->inside, 0, state->inside);
  }
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static bool htp_ops_softmax_outside_try_parallel(__fp16* dst, const __fp16* src,
                                                 int32_t outside, int32_t channel,
                                                 int32_t inside, int32_t plane) {
  if (g_max_num_workers <= 1 || outside < 2 || inside > 64 || (int64_t)outside * channel * inside < 4096) {
    return false;
  }
  int task_count = (int)g_max_num_workers;
  if (task_count > outside) {
    task_count = outside;
  }
  if (task_count <= 1) {
    return false;
  }
  HtpOpsSoftmaxOutsideTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.channel = channel;
  state.inside = inside;
  state.plane = plane;
  HtpOpsSoftmaxOutsideTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsSoftmaxOutsideTask, task_count);
  worker_pool_job_t job;
  job.fptr = htp_ops_softmax_outside_worker;
  worker_pool_synctoken_init(&state.sync_ctx, task_count);
  const int rows_per_task = (outside + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    const int begin = i * rows_per_task;
    int end = begin + rows_per_task;
    if (end > outside) {
      end = outside;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
  return true;
}

static inline float htp_ops_softmax_reduce_sum_f32(HVX_Vector sum0, HVX_Vector sum1) {
  HVX_Vector sum = Q6_Vsf_vadd_VsfVsf(sum0, sum1);
  sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, 64)));
  sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, 32)));
  sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, 16)));
  sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, 8)));
  sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, 4)));

  float tmp[32] __attribute__((aligned(128)));
  *(HVX_Vector*)tmp = sum;
  return tmp[0];
}

static inline void htp_ops_softmax_fp16_row_inside1_hvx(__fp16* dst, const __fp16* src, int32_t channel) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = channel & -vec_len;
  const uint16_t maxBits = htp_ops_softmax_fp16_max_bits(src, channel);
  const HVX_Vector max_v = Q6_Vh_vsplat_R(maxBits);
  const HVX_Vector log2e_v = Q6_Vh_vsplat_R(0x3dc5);

  int i = 0;
  HVX_Vector sum0 = Q6_V_vzero();
  HVX_Vector sum1 = Q6_V_vzero();
  for (; i < vec_end; i += vec_len) {
    HVX_Vector x = vmemu((const HVX_Vector*)(src + i));
    HVX_Vector diff = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x, max_v));
    HVX_Vector expArg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(diff, log2e_v));
    HVX_Vector expValue = hvx_my_exp2_vhf(expArg);
    vmemu((HVX_Vector*)(dst + i)) = expValue;
    HVX_VectorPair expSf = hvx_my_vhf_to_wsf(expValue);
    sum0 = Q6_Vsf_vadd_VsfVsf(sum0, Q6_V_lo_W(expSf));
    sum1 = Q6_Vsf_vadd_VsfVsf(sum1, Q6_V_hi_W(expSf));
  }
  const int tailStart = i;
  if (tailStart < channel) {
    const uint32_t tailBytes = (uint32_t)((channel - tailStart) * sizeof(__fp16));
    const HVX_VectorPred qTail = Q6_Q_vsetq_R((int)tailBytes);
    HVX_Vector x = Q6_V_vzero();
    memcpy(&x, src + tailStart, tailBytes);
    HVX_Vector diff = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x, max_v));
    HVX_Vector expArg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(diff, log2e_v));
    HVX_Vector expValue = hvx_my_exp2_vhf(expArg);
    expValue = Q6_V_vmux_QVV(qTail, expValue, Q6_V_vzero());
    vstu_variable(dst + tailStart, tailBytes, expValue);
    HVX_VectorPair expSf = hvx_my_vhf_to_wsf(expValue);
    sum0 = Q6_Vsf_vadd_VsfVsf(sum0, Q6_V_lo_W(expSf));
    sum1 = Q6_Vsf_vadd_VsfVsf(sum1, Q6_V_hi_W(expSf));
  }
  float sumValue = htp_ops_softmax_reduce_sum_f32(sum0, sum1);

  const __fp16 invSum = (__fp16)(1.0f / sumValue);
  const uint16_t invSumBits = *(const uint16_t*)&invSum;
  const HVX_Vector invSum_v = Q6_Vh_vsplat_R(invSumBits);
  i = 0;
  for (; i < vec_end; i += vec_len) {
    HVX_Vector x = vmemu((const HVX_Vector*)(dst + i));
    HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x, invSum_v));
    vmemu((HVX_Vector*)(dst + i)) = y;
  }
  if (tailStart < channel) {
    const uint32_t tailBytes = (uint32_t)((channel - tailStart) * sizeof(__fp16));
    HVX_Vector x = Q6_V_vzero();
    memcpy(&x, dst + tailStart, tailBytes);
    HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x, invSum_v));
    vstu_variable(dst + tailStart, tailBytes, y);
  }
}

typedef struct {
  __fp16* dst;
  const __fp16* src;
  int32_t channel;
  int32_t plane;
  worker_synctoken_t sync_ctx;
} HtpOpsSoftmaxInside1TaskState;

typedef struct {
  HtpOpsSoftmaxInside1TaskState* state;
  int32_t begin;
  int32_t end;
} HtpOpsSoftmaxInside1Task;

static void htp_ops_softmax_inside1_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsSoftmaxInside1Task* task = (HtpOpsSoftmaxInside1Task*)data;
  HtpOpsSoftmaxInside1TaskState* state = task->state;
  for (int32_t o = task->begin; o < task->end; ++o) {
    htp_ops_softmax_fp16_row_inside1_hvx(state->dst + (size_t)o * state->plane,
                                         state->src + (size_t)o * state->plane,
                                         state->channel);
  }
  worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static bool htp_ops_softmax_inside1_try_parallel(__fp16* dst, const __fp16* src,
                                                 int32_t outside, int32_t channel, int32_t plane) {
  if (g_max_num_workers <= 1 || outside < 2 || (int64_t)outside * channel < 4096) {
    return false;
  }
  int task_count = (int)g_max_num_workers;
  if (task_count > outside) {
    task_count = outside;
  }
  if (task_count <= 1) {
    return false;
  }
  HtpOpsSoftmaxInside1TaskState state = {};
  state.dst = dst;
  state.src = src;
  state.channel = channel;
  state.plane = plane;
  HtpOpsSoftmaxInside1Task* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsSoftmaxInside1Task, task_count);
  worker_pool_job_t job;
  job.fptr = htp_ops_softmax_inside1_worker;
  worker_pool_synctoken_init(&state.sync_ctx, task_count);
  const int rows_per_task = (outside + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    const int begin = i * rows_per_task;
    int end = begin + rows_per_task;
    if (end > outside) {
      end = outside;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
  return true;
}

AEEResult htp_ops_softmax(uint8_t* dst, const uint8_t* src, int32_t outside, int32_t channel,
                          int32_t inside, int32_t bytes) {
  if (dst == nullptr || src == nullptr || outside <= 0 || channel <= 0 || inside <= 0) {
    return AEE_EBADPARM;
  }
  if (bytes != 2) {
    return AEE_EBADPARM;
  }

  const int32_t plane = channel * inside;
  const __fp16* srcFp16 = (const __fp16*)src;
  __fp16* dstFp16 = (__fp16*)dst;
  if (inside == 1) {
    if (!htp_ops_softmax_inside1_try_parallel(dstFp16, srcFp16, outside, channel, plane)) {
      for (int32_t o = 0; o < outside; ++o) {
        htp_ops_softmax_fp16_row_inside1_hvx(dstFp16 + o * plane, srcFp16 + o * plane, channel);
      }
    }
    return AEE_SUCCESS;
  }
  if (htp_ops_softmax_outside_try_parallel(dstFp16, srcFp16, outside, channel, inside, plane)) {
    return AEE_SUCCESS;
  }
  for (int32_t o = 0; o < outside; ++o) {
    if (!htp_ops_softmax_fp16_row_try_parallel(dstFp16 + o * plane, srcFp16 + o * plane,
                                               channel, inside)) {
      htp_ops_softmax_fp16_row(dstFp16 + o * plane, srcFp16 + o * plane, channel, inside);
    }
  }
  return AEE_SUCCESS;
}

}  // extern "C"
