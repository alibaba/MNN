#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stddef.h>
#include <stdint.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx_utils.h"
#include "dsp/worker_pool.h"

extern "C" {

static inline uint16_t htp_ops_topk_fp16_max_bits(const __fp16* src, int32_t size) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = size & -vec_len;
  const __fp16* ptr = src;
  int i = 0;

  __fp16 best_scalar = src[0];
  HVX_Vector best_v = Q6_Vh_vsplat_R(((const uint16_t*)src)[0]);
  for (; i < vec_end; i += vec_len) {
    HVX_Vector v = vmemu((const HVX_Vector*)ptr);
    best_v = Q6_Vhf_vmax_VhfVhf(best_v, v);
    ptr += vec_len;
  }

  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 64));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 32));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 16));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 8));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 4));
  best_v = Q6_Vhf_vmax_VhfVhf(best_v, Q6_V_vror_VR(best_v, 2));

  __attribute__((aligned(128))) uint16_t tmp[vec_len];
  vmemu((HVX_Vector*)tmp) = best_v;
  uint16_t best_bits = tmp[0];
  best_scalar = *(__fp16*)&best_bits;

  for (; i < size; ++i) {
    const __fp16 value = src[i];
    if (value > best_scalar) {
      best_scalar = value;
      best_bits = ((const uint16_t*)src)[i];
    }
  }
  return best_bits;
}

AEEResult htp_ops_topkv2_k1_fp16(uint8_t* values, uint8_t* indices, uint8_t* input, int32_t rowSize, int32_t rows) {
  if (values == nullptr || indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  __fp16* valueOut = (__fp16*)values;
  int32_t* indexOut = (int32_t*)indices;
  for (int r = 0; r < rows; ++r) {
    const __fp16* row = src + (size_t)r * rowSize;
    const uint16_t bestBits = htp_ops_topk_fp16_max_bits(row, rowSize);
    const uint16_t* rowBits = (const uint16_t*)row;
    int32_t bestIndex = 0;
    for (int i = 0; i < rowSize; ++i) {
      if (rowBits[i] == bestBits) {
        bestIndex = i;
        break;
      }
    }
    ((uint16_t*)valueOut)[r] = bestBits;
    indexOut[r] = bestIndex;
  }
  return 0;
}

// Single-row argmax: one pass over the row, fully HVX. Per 4x128B tile the running max is a
// register; the scalar store-back (and the leftmost-equal find) only happens when a tile raises
// it, so the common path is just loads + horizontal vmax. Tiles are 4K apart with an L2 prefetch
// of the next tile to hide DDR latency. Ties resolve to the lowest index (matches CPU ArgMax).
static inline int32_t htp_ops_argmax_fp16_row(const __fp16* __restrict row, int32_t rowSize) {
  const int vec_len = 128 / (int)sizeof(__fp16);
  const int vec_end = rowSize & -vec_len;
  constexpr int UNROLL = 4;
  __fp16 bestValue = row[0];
  int32_t bestIndex = 0;

  int i = 0;
  for (; i + UNROLL * vec_len <= vec_end; i += UNROLL * vec_len) {
    // Prefetch the tile after next (1KB stride keeps the L2 prefetch lines streaming).
    if (i + 2 * UNROLL * vec_len < rowSize) {
      l2fetch(row + i + 2 * UNROLL * vec_len, 128, 128, UNROLL, 0);
    }
    HVX_Vector v[UNROLL];
    for (int k = 0; k < UNROLL; ++k) {
      v[k] = vmemu((const HVX_Vector*)(row + i + k * vec_len));
    }
    HVX_Vector vmax = Q6_Vhf_vmax_VhfVhf(v[0], v[1]);
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, v[2]);
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, v[3]);
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 64));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 32));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 16));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 8));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 4));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 2));
    __attribute__((aligned(128))) uint16_t tmp[vec_len];
    vmemu((HVX_Vector*)tmp) = vmax;
    const __fp16 chunkMax = *(__fp16*)&tmp[0];
    if (chunkMax > bestValue) {
      bestValue = chunkMax;
      for (int k = 0; k < UNROLL; ++k) {
        const HVX_Vector m = Q6_V_vmux_QVV(
            Q6_Q_vcmp_eq_VhVh(v[k], Q6_Vh_vsplat_R(((const uint16_t*)&chunkMax)[0])), Q6_V_vsplat_R(-1),
            Q6_V_vzero());
        __attribute__((aligned(128))) uint16_t lanes[vec_len];
        vmemu((HVX_Vector*)lanes) = m;
        for (int l = 0; l < vec_len; ++l) {
          if (lanes[l] == 0xffffu) {
            bestIndex = i + k * vec_len + l;
            goto found_block;
          }
        }
      }
    }
  found_block:;
  }
  for (; i < vec_end; i += vec_len) {
    const HVX_Vector v = vmemu((const HVX_Vector*)(row + i));
    HVX_Vector vmax = Q6_Vhf_vmax_VhfVhf(v, Q6_V_vror_VR(v, 64));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 32));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 16));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 8));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 4));
    vmax = Q6_Vhf_vmax_VhfVhf(vmax, Q6_V_vror_VR(vmax, 2));
    __attribute__((aligned(128))) uint16_t tmp2[vec_len];
    vmemu((HVX_Vector*)tmp2) = vmax;
    const __fp16 chunkMax = *(__fp16*)&tmp2[0];
    if (chunkMax > bestValue) {
      bestValue = chunkMax;
      const HVX_Vector m = Q6_V_vmux_QVV(
          Q6_Q_vcmp_eq_VhVh(v, Q6_Vh_vsplat_R(((const uint16_t*)&chunkMax)[0])), Q6_V_vsplat_R(-1),
          Q6_V_vzero());
      __attribute__((aligned(128))) uint16_t lanes[vec_len];
      vmemu((HVX_Vector*)lanes) = m;
      for (int l = 0; l < vec_len; ++l) {
        if (lanes[l] == 0xffffu) {
          bestIndex = i + l;
          break;
        }
      }
    }
  }
  for (; i < rowSize; ++i) {
    const __fp16 value = row[i];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

typedef struct {
  const __fp16* src;
  int32_t* indexOut;
  int32_t rowBegin;
  int32_t rowEnd;
  int32_t rowSize;
  worker_synctoken_t* sync;
} HtpOpsArgmaxTask;

static void htp_ops_argmax_worker(void* data, int worker_index) {
  (void)worker_index;
  HtpOpsArgmaxTask* task = (HtpOpsArgmaxTask*)data;
  for (int32_t r = task->rowBegin; r < task->rowEnd; ++r) {
    task->indexOut[r] = htp_ops_argmax_fp16_row(task->src + (size_t)r * task->rowSize, task->rowSize);
  }
  worker_pool_synctoken_jobdone(task->sync);
}

// Row-wise argmax over fp16 rows, split across the HTP worker pool when there are enough rows.
// The caller thread also takes a share of the rows: worker_pool_submit from a pool thread would
// run inline (serializing the fan-out), so the caller must never be counted into the pool.
AEEResult htp_ops_argmax_fp16(uint8_t* indices, const uint8_t* input, int32_t rowSize, int32_t rows) {
  if (indices == nullptr || input == nullptr || rowSize <= 0 || rows <= 0) {
    return -1;
  }
  const __fp16* src = (const __fp16*)input;
  int32_t* indexOut = (int32_t*)indices;
  int taskCount = (int)g_max_num_workers;
#ifdef MNN_DSP_ARGMAX_FORCE_SERIAL
  taskCount = 1;
#endif
  if (taskCount > rows) {
    taskCount = rows;
  }
  if (taskCount <= 1 || (int64_t)rowSize * rows < 4096) {
    for (int r = 0; r < rows; ++r) {
      indexOut[r] = htp_ops_argmax_fp16_row(src + (size_t)r * rowSize, rowSize);
    }
    return 0;
  }
  // taskCount pool workers, each on its own thread, plus the caller row range. The caller's
  // rows go LAST: they run while the pool drains, hiding the submit/sync round-trips.
  const int rowsPerTask = (rows + taskCount) / (taskCount + 1);
  int callerBegin = taskCount * rowsPerTask;
  int callerEnd = rows;
  // When rows don't divide evenly the tail is caller-only; clamp so the pool covers the rest.
  if (callerBegin > callerEnd) {
    callerBegin = callerEnd;
  }
  worker_synctoken_t sync;
  worker_pool_synctoken_init(&sync, (unsigned int)taskCount);
  HtpOpsArgmaxTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsArgmaxTask, taskCount);
  worker_pool_job_t job;
  job.fptr = htp_ops_argmax_worker;
  for (int i = 0; i < taskCount; ++i) {
    const int begin = i * rowsPerTask;
    int end = begin + rowsPerTask;
    if (end > callerBegin) {
      end = callerBegin;
    }
    tasks[i].src = src;
    tasks[i].indexOut = indexOut;
    tasks[i].rowBegin = begin;
    tasks[i].rowEnd = end;
    tasks[i].rowSize = rowSize;
    tasks[i].sync = &sync;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  for (int r = callerBegin; r < callerEnd; ++r) {
    indexOut[r] = htp_ops_argmax_fp16_row(src + (size_t)r * rowSize, rowSize);
  }
  worker_pool_synctoken_wait(&sync);
  return 0;
}

} // extern "C"
