#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/hvx_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "transpose_hvx.h"

extern "C" {

typedef enum {
  HTP_OPS_CONVERT_NC4HW4_TO_NCHW = 0,
  HTP_OPS_CONVERT_NCHW_TO_NC4HW4 = 1,
  HTP_OPS_CONVERT_COPY = 2,
  HTP_OPS_CONVERT_NC4HW4_FLATTEN_CW = 3,
} HtpOpsConvertType;

static inline uint8_t* htp_ops_tensor_convert_alloc_bytes(size_t bytes) {
  uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
  if (vtcm_ptr == NULL) {
    return NULL;
  }
  return vtcm_seq_alloc(&vtcm_ptr, bytes);
}

static inline HVX_Vector* htp_ops_tensor_convert_alloc_workspace() {
  return (HVX_Vector*)htp_ops_tensor_convert_alloc_bytes(128 * 128);
}

static inline bool htp_ops_tensor_convert_aligned_128(const void* ptr) {
  return (((uintptr_t)ptr) & 127) == 0;
}

static inline void htp_ops_tensor_convert_copy_hvx(uint8_t* dst, const uint8_t* src, size_t bytes) {
  size_t offset = 0;
  const size_t vecBytes = 128;
  for (; offset + vecBytes <= bytes; offset += vecBytes) {
    vmemu((HVX_Vector*)(dst + offset)) = vmemu((const HVX_Vector*)(src + offset));
  }
  if (offset < bytes) {
    memcpy(dst + offset, src + offset, bytes - offset);
  }
}

typedef struct {
  uint8_t* dst;
  const uint8_t* src;
  size_t bytes;
  size_t grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsTensorConvertCopyTaskState;

static void htp_ops_tensor_convert_copy_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsTensorConvertCopyTaskState* state = (HtpOpsTensorConvertCopyTaskState*)data;
  while (true) {
    const size_t taskId = (size_t)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const size_t begin = taskId * state->grain;
    if (begin >= state->bytes) {
      break;
    }
    size_t end = begin + state->grain;
    if (end > state->bytes) {
      end = state->bytes;
    }
    htp_ops_tensor_convert_copy_hvx(state->dst + begin, state->src + begin, end - begin);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline void htp_ops_tensor_convert_copy_parallel(uint8_t* dst, const uint8_t* src, size_t bytes) {
  if (bytes < 16384 || g_max_num_workers <= 1) {
    htp_ops_tensor_convert_copy_hvx(dst, src, bytes);
    return;
  }
  int nTasks = (int)g_max_num_workers;
  const size_t minBytesPerTask = 8192;
  const size_t bySize = (bytes + minBytesPerTask - 1) / minBytesPerTask;
  if ((size_t)nTasks > bySize) {
    nTasks = (int)bySize;
  }
  if (nTasks <= 1) {
    htp_ops_tensor_convert_copy_hvx(dst, src, bytes);
    return;
  }
  HtpOpsTensorConvertCopyTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.bytes = bytes;
  state.grain = ((bytes + (size_t)nTasks - 1) / (size_t)nTasks + 127) & ~(size_t)127;
  state.task_id = 0;

  worker_pool_job_t job;
  job.fptr = htp_ops_tensor_convert_copy_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static inline int htp_ops_tensor_convert_pick_task_count(size_t bytes, int units) {
  if (units <= 1 || g_max_num_workers <= 1 || bytes < 2048) {
    return 1;
  }
  int n = (int)g_max_num_workers;
  if (n > units) {
    n = units;
  }
  return n;
}

static inline HVX_Vector* htp_ops_tensor_convert_pack_workspaces() {
  int slots = (int)g_max_num_workers;
  if (slots < 1) {
    slots = 1;
  }
  return (HVX_Vector*)vtcm_manager_reserve_area("tensor_convert_pack_tiles",
                                                (size_t)slots * 64 * sizeof(HVX_Vector),
                                                128);
}

typedef struct {
  uint8_t* dst;
  const uint8_t* src;
  int batch;
  int area;
  int channel;
  int dstC4;
  int srcC4;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
  HVX_Vector* workspaces;
} HtpOpsTensorConvertFlattenCWTaskState;

static inline void htp_ops_tensor_convert_flatten_cw_fp16_range(
    HtpOpsTensorConvertFlattenCWTaskState* state, int cpStart, int cpEnd) {
  const int batch = state->batch;
  const int area = state->area;
  const int channel = state->channel;
  const int dstC = channel * area;
  const int pack = 64;
  const int srcArea = batch * area;
  const uint16_t* srcBase = (const uint16_t*)state->src;
  uint16_t* dstBase = (uint16_t*)state->dst;
  for (int cpOut = cpStart; cpOut < cpEnd; ++cpOut) {
    const int cOutBegin = cpOut * pack;
    const int cOutEnd = cOutBegin + pack < dstC ? cOutBegin + pack : dstC;
    int cOut = cOutBegin;
    while (cOut < cOutEnd) {
      const int srcC = cOut / area;
      const int srcInner = cOut - srcC * area;
      const int srcCp = srcC / pack;
      const int srcK = srcC - srcCp * pack;
      const int innerRemain = area - srcInner;
      const int packRemain = cOutEnd - cOut;
      const int innerCount = innerRemain < packRemain ? innerRemain : packRemain;
      for (int ni = 0; ni < batch; ++ni) {
        const uint16_t* srcY = srcBase + (size_t)srcCp * srcArea * pack +
                               (size_t)(ni * area + srcInner) * pack + srcK;
        uint16_t* dstY = dstBase + (size_t)(cpOut * batch + ni) * pack + (cOut - cOutBegin);
        int x = 0;
        for (; x + 3 < innerCount; x += 4) {
          dstY[x + 0] = srcY[(x + 0) * pack];
          dstY[x + 1] = srcY[(x + 1) * pack];
          dstY[x + 2] = srcY[(x + 2) * pack];
          dstY[x + 3] = srcY[(x + 3) * pack];
        }
        for (; x < innerCount; ++x) {
          dstY[x] = srcY[x * pack];
        }
      }
      cOut += innerCount;
    }
  }
}

static inline void htp_ops_tensor_convert_flatten_cw_area20_fp16_units_range(
    HtpOpsTensorConvertFlattenCWTaskState* state, int unitStart, int unitEnd,
    HVX_Vector* workspace) {
  const int pack = 64;
  const int area = 20;
  const int batch = state->batch;
  const int srcArea = batch * area;
  const uint16_t* srcBase = (const uint16_t*)state->src;
  uint16_t* dstBase = (uint16_t*)state->dst;
  HVX_Vector* v = workspace;
  for (int unit = unitStart; unit < unitEnd; ++unit) {
    const int srcCp = unit / batch;
    const int ni = unit - srcCp * batch;
    const int validChannels = state->channel - srcCp * pack < pack ? state->channel - srcCp * pack : pack;
    const uint16_t* src = srcBase + (size_t)srcCp * srcArea * pack + (size_t)ni * area * pack;
    for (int a = 0; a < area; ++a) {
      v[a] = vmemu((const HVX_Vector*)(src + (size_t)a * pack));
    }
    for (int a = area; a < pack; ++a) {
      v[a] = Q6_V_vzero();
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < validChannels; ++k) {
      const int cOut = (srcCp * pack + k) * area;
      const int dstCp = cOut / pack;
      const int dstOffset = cOut - dstCp * pack;
      const int first = pack - dstOffset < area ? pack - dstOffset : area;
      const uint16_t* value = (const uint16_t*)&v[k];
      uint16_t* dst = dstBase + (size_t)(dstCp * batch + ni) * pack + dstOffset;
      memcpy(dst, value, (size_t)first * sizeof(uint16_t));
      if (first < area) {
        uint16_t* dstNext = dstBase + (size_t)((dstCp + 1) * batch + ni) * pack;
        memcpy(dstNext, value + first, (size_t)(area - first) * sizeof(uint16_t));
      }
    }
  }
}

static void htp_ops_tensor_convert_flatten_cw_area20_fp16_worker(void* data, int worker_id) {
  HtpOpsTensorConvertFlattenCWTaskState* state = (HtpOpsTensorConvertFlattenCWTaskState*)data;
  HVX_Vector* workspace = state->workspaces + (size_t)worker_id * 64;
  const int unitCount = state->batch * state->srcC4;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int unitStart = taskId * state->grain;
    if (unitStart >= unitCount) {
      break;
    }
    int unitEnd = unitStart + state->grain;
    if (unitEnd > unitCount) {
      unitEnd = unitCount;
    }
    htp_ops_tensor_convert_flatten_cw_area20_fp16_units_range(state, unitStart, unitEnd, workspace);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void htp_ops_tensor_convert_flatten_cw_fp16_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsTensorConvertFlattenCWTaskState* state = (HtpOpsTensorConvertFlattenCWTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int cpStart = taskId * state->grain;
    if (cpStart >= state->dstC4) {
      break;
    }
    int cpEnd = cpStart + state->grain;
    if (cpEnd > state->dstC4) {
      cpEnd = state->dstC4;
    }
    htp_ops_tensor_convert_flatten_cw_fp16_range(state, cpStart, cpEnd);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_tensor_convert_flatten_cw_fp16(uint8_t* dst, const uint8_t* src,
                                                          int batch, int area, int channel) {
  const int pack = 64;
  const int dstC = channel * area;
  const int dstC4 = (dstC + pack - 1) / pack;
  const int srcC4 = (channel + pack - 1) / pack;
  const size_t dstBytes = (size_t)batch * dstC4 * pack * sizeof(uint16_t);
  if (dstC != dstC4 * pack) {
    memset(dst, 0, dstBytes);
  }

  const size_t workBytes = (size_t)batch * dstC * sizeof(uint16_t);
  if (area == 20) {
    const int unitCount = batch * srcC4;
    const int nTasks = htp_ops_tensor_convert_pick_task_count(workBytes, unitCount);
    HtpOpsTensorConvertFlattenCWTaskState state = {};
    state.dst = dst;
    state.src = src;
    state.batch = batch;
    state.area = area;
    state.channel = channel;
    state.dstC4 = dstC4;
    state.srcC4 = srcC4;
    state.grain = (unitCount + nTasks - 1) / nTasks;
    state.task_id = 0;
    state.workspaces = htp_ops_tensor_convert_pack_workspaces();
    if (state.workspaces == NULL) {
      return false;
    }
    if (nTasks <= 1) {
      htp_ops_tensor_convert_flatten_cw_area20_fp16_units_range(&state, 0, unitCount, state.workspaces);
      return true;
    }
    worker_pool_job_t job;
    job.fptr = htp_ops_tensor_convert_flatten_cw_area20_fp16_worker;
    job.dptr = &state;
    worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
    for (int i = 0; i < nTasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
    return true;
  }

  const int nTasks = htp_ops_tensor_convert_pick_task_count(workBytes, dstC4);
  HtpOpsTensorConvertFlattenCWTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.batch = batch;
  state.area = area;
  state.channel = channel;
  state.dstC4 = dstC4;
  state.srcC4 = srcC4;
  state.grain = (dstC4 + nTasks - 1) / nTasks;
  state.task_id = 0;
  if (nTasks <= 1) {
    htp_ops_tensor_convert_flatten_cw_fp16_range(&state, 0, dstC4);
    return true;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_tensor_convert_flatten_cw_fp16_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dst;
  const uint8_t* src;
  int batch;
  int area;
  int channel;
  int c4;
  int direction;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
  HVX_Vector* workspaces;
} HtpOpsTensorConvertPackTaskState;

static inline void htp_ops_tensor_convert_nc4hw4_to_nchw_area_z(
    uint8_t* dstBase, const uint8_t* srcBase, int batch, int area, int channel, int z, HVX_Vector* v) {
  const int pack = 64;
  const int c4 = (channel + pack - 1) / pack;
  const int cp = z % c4;
  const int ni = z / c4;
  const int validChannels = channel - cp * pack < pack ? channel - cp * pack : pack;
  const __fp16* srcZ = (const __fp16*)srcBase + (size_t)(cp * batch + ni) * area * pack;
  __fp16* dstZ = (__fp16*)dstBase + (size_t)(ni * channel + cp * pack) * area;
  const int rowMain = area & (~63);
  for (int row = 0; row < rowMain; row += 64) {
    for (int k = 0; k < 64; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + (row + k) * pack));
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < validChannels; ++k) {
      vmemu((HVX_Vector*)(dstZ + k * area + row)) = v[k];
    }
  }
  if (rowMain < area) {
    const int rowTail = area - rowMain;
    for (int k = 0; k < rowTail; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + (rowMain + k) * pack));
    }
    for (int k = rowTail; k < 64; ++k) {
      v[k] = Q6_V_vzero();
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < validChannels; ++k) {
      vstu_variable(dstZ + k * area + rowMain, (uint32_t)((size_t)rowTail * sizeof(__fp16)), v[k]);
    }
  }
}

static inline void htp_ops_tensor_convert_nchw_to_nc4hw4_area_z(
    uint8_t* dstBase, const uint8_t* srcBase, int batch, int area, int channel, int z, int zCount,
    HVX_Vector* v) {
  const int pack = 64;
  const int c4 = (channel + pack - 1) / pack;
  const int cp = z % c4;
  const int ni = z / c4;
  const int validChannels = channel - cp * pack < pack ? channel - cp * pack : pack;
  const __fp16* srcZ = (const __fp16*)srcBase + (size_t)(ni * channel + cp * pack) * area;
  __fp16* dstZ = (__fp16*)dstBase + (size_t)(cp * batch + ni) * area * pack;
  const int colMain = area & (~63);
  for (int col = 0; col < colMain; col += 64) {
    for (int k = 0; k < validChannels; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + k * area + col));
    }
    for (int k = validChannels; k < 64; ++k) {
      v[k] = Q6_V_vzero();
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < 64; ++k) {
      vmemu((HVX_Vector*)(dstZ + (col + k) * pack)) = v[k];
    }
  }
  if (colMain < area) {
    const int colTail = area - colMain;
    if (validChannels == 64 && z + 1 < zCount) {
      for (int k = 0; k < 64; ++k) {
        v[k] = vmemu((const HVX_Vector*)(srcZ + k * area + colMain));
      }
      hvx_transpose_64x64_first(v, validChannels);
      for (int k = 0; k < colTail; ++k) {
        vmemu((HVX_Vector*)(dstZ + (colMain + k) * pack)) = v[k];
      }
    } else {
      const size_t tailBytes = (size_t)colTail * sizeof(__fp16);
      const HVX_Vector zero = Q6_V_vzero();
      const HVX_VectorPred qTail = Q6_Q_vsetq_R((int)tailBytes);
      int row = 0;
      for (; row + 1 < validChannels; ++row) {
        HVX_Vector load = vmemu((const HVX_Vector*)(srcZ + row * area + colMain));
        v[row] = Q6_V_vmux_QVV(qTail, load, zero);
      }
      for (; row < validChannels; ++row) {
        v[row] = zero;
        memcpy(&v[row], srcZ + row * area + colMain, tailBytes);
      }
      for (int row = validChannels; row < 64; ++row) {
        v[row] = zero;
      }
      hvx_transpose_64x64_first(v, colTail);
      for (int col = 0; col < colTail; ++col) {
        vmemu((HVX_Vector*)(dstZ + (colMain + col) * pack)) = v[col];
      }
    }
  }
}

static inline void htp_ops_tensor_convert_area1_to_nchw_batch_range(HtpOpsTensorConvertPackTaskState* state,
                                                                    int batchStart, int batchEnd) {
  const int pack = 64;
  const int cFull = state->channel / pack;
  const int cRemain = state->channel - cFull * pack;
  const __fp16* srcBase = (const __fp16*)state->src;
  __fp16* dstBase = (__fp16*)state->dst;
  int ni = batchStart;
  for (; ni + 7 < batchEnd; ni += 8) {
    __fp16* dst0 = dstBase + (size_t)(ni + 0) * state->channel;
    __fp16* dst1 = dstBase + (size_t)(ni + 1) * state->channel;
    __fp16* dst2 = dstBase + (size_t)(ni + 2) * state->channel;
    __fp16* dst3 = dstBase + (size_t)(ni + 3) * state->channel;
    __fp16* dst4 = dstBase + (size_t)(ni + 4) * state->channel;
    __fp16* dst5 = dstBase + (size_t)(ni + 5) * state->channel;
    __fp16* dst6 = dstBase + (size_t)(ni + 6) * state->channel;
    __fp16* dst7 = dstBase + (size_t)(ni + 7) * state->channel;
    if (cFull > 1) {
      const __fp16* fetch = srcBase + (size_t)ni * pack;
      l2fetch(fetch, (uint32_t)(state->batch * pack * sizeof(__fp16)),
              (uint32_t)(8 * pack * sizeof(__fp16)), (uint32_t)cFull, 0);
    }
    for (int cp = 0; cp < cFull; ++cp) {
      const __fp16* src = srcBase + (size_t)(cp * state->batch + ni) * pack;
      HVX_Vector v0 = vmemu((const HVX_Vector*)src);
      HVX_Vector v1 = vmemu((const HVX_Vector*)(src + pack));
      HVX_Vector v2 = vmemu((const HVX_Vector*)(src + 2 * pack));
      HVX_Vector v3 = vmemu((const HVX_Vector*)(src + 3 * pack));
      HVX_Vector v4 = vmemu((const HVX_Vector*)(src + 4 * pack));
      HVX_Vector v5 = vmemu((const HVX_Vector*)(src + 5 * pack));
      HVX_Vector v6 = vmemu((const HVX_Vector*)(src + 6 * pack));
      HVX_Vector v7 = vmemu((const HVX_Vector*)(src + 7 * pack));
      vmemu((HVX_Vector*)(dst0 + (size_t)cp * pack)) = v0;
      vmemu((HVX_Vector*)(dst1 + (size_t)cp * pack)) = v1;
      vmemu((HVX_Vector*)(dst2 + (size_t)cp * pack)) = v2;
      vmemu((HVX_Vector*)(dst3 + (size_t)cp * pack)) = v3;
      vmemu((HVX_Vector*)(dst4 + (size_t)cp * pack)) = v4;
      vmemu((HVX_Vector*)(dst5 + (size_t)cp * pack)) = v5;
      vmemu((HVX_Vector*)(dst6 + (size_t)cp * pack)) = v6;
      vmemu((HVX_Vector*)(dst7 + (size_t)cp * pack)) = v7;
    }
    if (cRemain > 0) {
      const __fp16* src = srcBase + (size_t)(cFull * state->batch + ni) * pack;
      const uint32_t tailBytes = (uint32_t)(cRemain * sizeof(__fp16));
      vstu_variable(dst0 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)src));
      vstu_variable(dst1 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + pack)));
      vstu_variable(dst2 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 2 * pack)));
      vstu_variable(dst3 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 3 * pack)));
      vstu_variable(dst4 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 4 * pack)));
      vstu_variable(dst5 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 5 * pack)));
      vstu_variable(dst6 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 6 * pack)));
      vstu_variable(dst7 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 7 * pack)));
    }
  }
  for (; ni + 3 < batchEnd; ni += 4) {
    __fp16* dst0 = dstBase + (size_t)(ni + 0) * state->channel;
    __fp16* dst1 = dstBase + (size_t)(ni + 1) * state->channel;
    __fp16* dst2 = dstBase + (size_t)(ni + 2) * state->channel;
    __fp16* dst3 = dstBase + (size_t)(ni + 3) * state->channel;
    if (cFull > 1) {
      const __fp16* fetch = srcBase + (size_t)ni * pack;
      l2fetch(fetch, (uint32_t)(state->batch * pack * sizeof(__fp16)),
              (uint32_t)(4 * pack * sizeof(__fp16)), (uint32_t)cFull, 0);
    }
    for (int cp = 0; cp < cFull; ++cp) {
      const __fp16* src = srcBase + (size_t)(cp * state->batch + ni) * pack;
      HVX_Vector v0 = vmemu((const HVX_Vector*)src);
      HVX_Vector v1 = vmemu((const HVX_Vector*)(src + pack));
      HVX_Vector v2 = vmemu((const HVX_Vector*)(src + 2 * pack));
      HVX_Vector v3 = vmemu((const HVX_Vector*)(src + 3 * pack));
      vmemu((HVX_Vector*)(dst0 + (size_t)cp * pack)) = v0;
      vmemu((HVX_Vector*)(dst1 + (size_t)cp * pack)) = v1;
      vmemu((HVX_Vector*)(dst2 + (size_t)cp * pack)) = v2;
      vmemu((HVX_Vector*)(dst3 + (size_t)cp * pack)) = v3;
    }
    if (cRemain > 0) {
      const __fp16* src = srcBase + (size_t)(cFull * state->batch + ni) * pack;
      const uint32_t tailBytes = (uint32_t)(cRemain * sizeof(__fp16));
      vstu_variable(dst0 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)src));
      vstu_variable(dst1 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + pack)));
      vstu_variable(dst2 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 2 * pack)));
      vstu_variable(dst3 + (size_t)cFull * pack, tailBytes, vmemu((const HVX_Vector*)(src + 3 * pack)));
    }
  }
  for (; ni < batchEnd; ++ni) {
    __fp16* dst = dstBase + (size_t)ni * state->channel;
    for (int cp = 0; cp < cFull; ++cp) {
      const __fp16* src = srcBase + (size_t)(cp * state->batch + ni) * pack;
      vmemu((HVX_Vector*)(dst + (size_t)cp * pack)) = vmemu((const HVX_Vector*)src);
    }
    if (cRemain > 0) {
      const __fp16* src = srcBase + (size_t)(cFull * state->batch + ni) * pack;
      vstu_variable(dst + (size_t)cFull * pack, (uint32_t)(cRemain * sizeof(__fp16)),
                    vmemu((const HVX_Vector*)src));
    }
  }
}

static inline void htp_ops_tensor_convert_area1_to_nc4hw4_cp_range(HtpOpsTensorConvertPackTaskState* state,
                                                                   int cpStart, int cpEnd) {
  const int pack = 64;
  const int cFull = state->channel / pack;
  const int cRemain = state->channel - cFull * pack;
  const __fp16* srcBase = (const __fp16*)state->src;
  __fp16* dstBase = (__fp16*)state->dst;
  for (int cp = cpStart; cp < cpEnd; ++cp) {
    const bool tail = cp >= cFull;
    const int valid = tail ? cRemain : pack;
    if (valid <= 0) {
      continue;
    }
    const __fp16* srcCp = srcBase + (size_t)cp * pack;
    __fp16* dstCp = dstBase + (size_t)cp * state->batch * pack;
    if (state->batch > 4) {
      l2fetch(srcCp, (uint32_t)(state->channel * sizeof(__fp16)),
              (uint32_t)(valid * sizeof(__fp16)), (uint32_t)state->batch, 0);
    }
    int ni = 0;
    if (tail) {
      HVX_Vector zero = Q6_V_vzero();
      const HVX_VectorPred qTail = Q6_Q_vsetq_R(cRemain * (int)sizeof(__fp16));
      for (; ni + 3 < state->batch - 1; ni += 4) {
        const __fp16* src0 = srcCp + (size_t)(ni + 0) * state->channel;
        const __fp16* src1 = srcCp + (size_t)(ni + 1) * state->channel;
        const __fp16* src2 = srcCp + (size_t)(ni + 2) * state->channel;
        const __fp16* src3 = srcCp + (size_t)(ni + 3) * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
        HVX_Vector v1 = vmemu((const HVX_Vector*)src1);
        HVX_Vector v2 = vmemu((const HVX_Vector*)src2);
        HVX_Vector v3 = vmemu((const HVX_Vector*)src3);
        vmemu((HVX_Vector*)dst) = Q6_V_vmux_QVV(qTail, v0, zero);
        vmemu((HVX_Vector*)(dst + pack)) = Q6_V_vmux_QVV(qTail, v1, zero);
        vmemu((HVX_Vector*)(dst + 2 * pack)) = Q6_V_vmux_QVV(qTail, v2, zero);
        vmemu((HVX_Vector*)(dst + 3 * pack)) = Q6_V_vmux_QVV(qTail, v3, zero);
      }
      for (; ni + 1 < state->batch; ++ni) {
        const __fp16* src = srcCp + (size_t)ni * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        HVX_Vector v = vmemu((const HVX_Vector*)src);
        vmemu((HVX_Vector*)dst) = Q6_V_vmux_QVV(qTail, v, zero);
      }
      for (; ni < state->batch; ++ni) {
        const __fp16* src = srcCp + (size_t)ni * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        vmemu((HVX_Vector*)dst) = zero;
        memcpy(dst, src, (size_t)cRemain * sizeof(__fp16));
      }
    } else {
      for (; ni + 7 < state->batch; ni += 8) {
        const __fp16* src0 = srcCp + (size_t)(ni + 0) * state->channel;
        const __fp16* src1 = srcCp + (size_t)(ni + 1) * state->channel;
        const __fp16* src2 = srcCp + (size_t)(ni + 2) * state->channel;
        const __fp16* src3 = srcCp + (size_t)(ni + 3) * state->channel;
        const __fp16* src4 = srcCp + (size_t)(ni + 4) * state->channel;
        const __fp16* src5 = srcCp + (size_t)(ni + 5) * state->channel;
        const __fp16* src6 = srcCp + (size_t)(ni + 6) * state->channel;
        const __fp16* src7 = srcCp + (size_t)(ni + 7) * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
        HVX_Vector v1 = vmemu((const HVX_Vector*)src1);
        HVX_Vector v2 = vmemu((const HVX_Vector*)src2);
        HVX_Vector v3 = vmemu((const HVX_Vector*)src3);
        HVX_Vector v4 = vmemu((const HVX_Vector*)src4);
        HVX_Vector v5 = vmemu((const HVX_Vector*)src5);
        HVX_Vector v6 = vmemu((const HVX_Vector*)src6);
        HVX_Vector v7 = vmemu((const HVX_Vector*)src7);
        vmemu((HVX_Vector*)dst) = v0;
        vmemu((HVX_Vector*)(dst + pack)) = v1;
        vmemu((HVX_Vector*)(dst + 2 * pack)) = v2;
        vmemu((HVX_Vector*)(dst + 3 * pack)) = v3;
        vmemu((HVX_Vector*)(dst + 4 * pack)) = v4;
        vmemu((HVX_Vector*)(dst + 5 * pack)) = v5;
        vmemu((HVX_Vector*)(dst + 6 * pack)) = v6;
        vmemu((HVX_Vector*)(dst + 7 * pack)) = v7;
      }
      for (; ni + 3 < state->batch; ni += 4) {
        const __fp16* src0 = srcCp + (size_t)(ni + 0) * state->channel;
        const __fp16* src1 = srcCp + (size_t)(ni + 1) * state->channel;
        const __fp16* src2 = srcCp + (size_t)(ni + 2) * state->channel;
        const __fp16* src3 = srcCp + (size_t)(ni + 3) * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
        HVX_Vector v1 = vmemu((const HVX_Vector*)src1);
        HVX_Vector v2 = vmemu((const HVX_Vector*)src2);
        HVX_Vector v3 = vmemu((const HVX_Vector*)src3);
        vmemu((HVX_Vector*)dst) = v0;
        vmemu((HVX_Vector*)(dst + pack)) = v1;
        vmemu((HVX_Vector*)(dst + 2 * pack)) = v2;
        vmemu((HVX_Vector*)(dst + 3 * pack)) = v3;
      }
      for (; ni < state->batch; ++ni) {
        const __fp16* src = srcCp + (size_t)ni * state->channel;
        __fp16* dst = dstCp + (size_t)ni * pack;
        vmemu((HVX_Vector*)dst) = vmemu((const HVX_Vector*)src);
      }
    }
  }
}

static inline void htp_ops_tensor_convert_c1_to_nc4hw4_range(HtpOpsTensorConvertPackTaskState* state,
                                                             int elemStart, int elemEnd) {
  const int pack = 64;
  const uint16_t* srcBase = (const uint16_t*)state->src;
  uint16_t* dstBase = (uint16_t*)state->dst;
  const HVX_Vector zero = Q6_V_vzero();
  if (state->batch == 1) {
    int elem = elemStart;
    for (; elem + 3 < elemEnd; elem += 4) {
      uint16_t* dst0 = dstBase + (size_t)(elem + 0) * pack;
      uint16_t* dst1 = dstBase + (size_t)(elem + 1) * pack;
      uint16_t* dst2 = dstBase + (size_t)(elem + 2) * pack;
      uint16_t* dst3 = dstBase + (size_t)(elem + 3) * pack;
      vmemu((HVX_Vector*)dst0) = zero;
      vmemu((HVX_Vector*)dst1) = zero;
      vmemu((HVX_Vector*)dst2) = zero;
      vmemu((HVX_Vector*)dst3) = zero;
      dst0[0] = srcBase[elem + 0];
      dst1[0] = srcBase[elem + 1];
      dst2[0] = srcBase[elem + 2];
      dst3[0] = srcBase[elem + 3];
    }
    for (; elem < elemEnd; ++elem) {
      uint16_t* dst = dstBase + (size_t)elem * pack;
      vmemu((HVX_Vector*)dst) = zero;
      dst[0] = srcBase[elem];
    }
    return;
  }
  for (int elem = elemStart; elem < elemEnd; ++elem) {
    const int ni = elem / state->area;
    const int wi = elem - ni * state->area;
    const size_t srcIdx = (size_t)ni * state->area + wi;
    const size_t dstIdx = ((size_t)ni * state->area + wi) * pack;
    vmemu((HVX_Vector*)(dstBase + dstIdx)) = zero;
    dstBase[dstIdx] = srcBase[srcIdx];
  }
}

static inline void htp_ops_tensor_convert_pack_area_range(HtpOpsTensorConvertPackTaskState* state,
                                                          int zStart, int zEnd, HVX_Vector* workspace) {
  const int zCount = state->batch * state->c4;
  for (int z = zStart; z < zEnd; ++z) {
    if (state->direction == 0) {
      htp_ops_tensor_convert_nc4hw4_to_nchw_area_z(state->dst, state->src,
                                                   state->batch, state->area, state->channel, z, workspace);
    } else {
      htp_ops_tensor_convert_nchw_to_nc4hw4_area_z(state->dst, state->src,
                                                   state->batch, state->area, state->channel, z, zCount, workspace);
    }
  }
}

static inline void htp_ops_tensor_convert_c4_1_area_blocks_range(HtpOpsTensorConvertPackTaskState* state,
                                                                 int unitStart, int unitEnd,
                                                                 HVX_Vector* v) {
  const int pack = 64;
  const int area = state->area;
  const int channel = state->channel;
  const int blockCount = (area + pack - 1) / pack;
  const bool srcBaseAligned = htp_ops_tensor_convert_aligned_128(state->src);
  const bool dstBaseAligned = htp_ops_tensor_convert_aligned_128(state->dst);
  const bool nchwAreaAligned = (area & (pack - 1)) == 0;
  for (int unit = unitStart; unit < unitEnd; ++unit) {
    const int z = unit / blockCount;
    const int block = unit - z * blockCount;
    const int cp = z % state->c4;
    const int ni = z / state->c4;
    const int validChannels = channel - cp * pack < pack ? channel - cp * pack : pack;
    const int row = block * pack;
    int count = area - row;
    if (count > pack) {
      count = pack;
    }
    if (count <= 0 || validChannels <= 0) {
      continue;
    }
    if (state->direction == 0) {
      const __fp16* src = (const __fp16*)state->src + ((size_t)(cp * state->batch + ni) * area + row) * pack;
      __fp16* dst = (__fp16*)state->dst + (size_t)(ni * channel + cp * pack) * area;
      if (srcBaseAligned) {
        if (channel <= 32 && count >= 16) {
          l2fetch(src, (uint32_t)(pack * sizeof(__fp16)),
                  (uint32_t)(pack * sizeof(__fp16)), (uint32_t)count, 0);
        }
        for (int k = 0; k < count; ++k) {
          v[k] = vmem((const HVX_Vector*)(src + (size_t)k * pack));
        }
      } else if (channel <= 32 && count >= 16) {
        l2fetch(src, (uint32_t)(pack * sizeof(__fp16)),
                (uint32_t)(pack * sizeof(__fp16)), (uint32_t)count, 0);
        for (int k = 0; k < count; ++k) {
          v[k] = vmemu((const HVX_Vector*)(src + (size_t)k * pack));
        }
      } else {
        for (int k = 0; k < count; ++k) {
          v[k] = vmemu((const HVX_Vector*)(src + (size_t)k * pack));
        }
      }
      for (int k = count; k < pack; ++k) {
        v[k] = Q6_V_vzero();
      }
      hvx_transpose_64x64_first(v, validChannels);
      if (count == pack) {
        if (dstBaseAligned && nchwAreaAligned) {
          for (int k = 0; k < validChannels; ++k) {
            vmem((HVX_Vector*)(dst + (size_t)k * area + row)) = v[k];
          }
        } else {
          for (int k = 0; k < validChannels; ++k) {
            vmemu((HVX_Vector*)(dst + (size_t)k * area + row)) = v[k];
          }
        }
      } else {
        const uint32_t tailBytes = (uint32_t)(count * sizeof(__fp16));
        for (int k = 0; k < validChannels; ++k) {
          vstu_variable(dst + (size_t)k * area + row, tailBytes, v[k]);
        }
      }
    } else {
      const __fp16* src = (const __fp16*)state->src + (size_t)(ni * channel + cp * pack) * area;
      __fp16* dst = (__fp16*)state->dst + ((size_t)(cp * state->batch + ni) * area + row) * pack;
      if (count == pack || (validChannels == pack && z + 1 < state->batch * state->c4)) {
        if (srcBaseAligned && nchwAreaAligned) {
          for (int k = 0; k < validChannels; ++k) {
            v[k] = vmem((const HVX_Vector*)(src + (size_t)k * area + row));
          }
        } else {
          for (int k = 0; k < validChannels; ++k) {
            v[k] = vmemu((const HVX_Vector*)(src + (size_t)k * area + row));
          }
        }
      } else {
        const size_t tailBytes = (size_t)count * sizeof(__fp16);
        for (int k = 0; k < validChannels; ++k) {
          v[k] = Q6_V_vzero();
          memcpy(&v[k], src + (size_t)k * area + row, tailBytes);
        }
      }
      for (int k = validChannels; k < pack; ++k) {
        v[k] = Q6_V_vzero();
      }
      hvx_transpose_64x64_first(v, count);
      if (dstBaseAligned) {
        for (int k = 0; k < count; ++k) {
          vmem((HVX_Vector*)(dst + (size_t)k * pack)) = v[k];
        }
      } else {
        for (int k = 0; k < count; ++k) {
          vmemu((HVX_Vector*)(dst + (size_t)k * pack)) = v[k];
        }
      }
    }
  }
}

static void htp_ops_tensor_convert_c4_1_area_worker(void* data, int worker_id) {
  HtpOpsTensorConvertPackTaskState* state = (HtpOpsTensorConvertPackTaskState*)data;
  const int blockCount = (state->area + 63) / 64;
  const int unitCount = state->batch * state->c4 * blockCount;
  HVX_Vector* workspace = state->workspaces + (size_t)worker_id * 64;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    if (begin >= unitCount) {
      break;
    }
    int end = begin + state->grain;
    if (end > unitCount) {
      end = unitCount;
    }
    htp_ops_tensor_convert_c4_1_area_blocks_range(state, begin, end, workspace);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void htp_ops_tensor_convert_pack_worker(void* data, int worker_id) {
  HtpOpsTensorConvertPackTaskState* state = (HtpOpsTensorConvertPackTaskState*)data;
  HVX_Vector* workspace = state->workspaces != NULL ? state->workspaces + (size_t)worker_id * 64 : NULL;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    const int total = state->area == 1 ? (state->direction == 0 ? state->batch : state->c4) : state->batch * state->c4;
    if (begin >= total) {
      break;
    }
    int end = begin + state->grain;
    if (end > total) {
      end = total;
    }
    if (state->area == 1) {
      if (state->direction == 0) {
        htp_ops_tensor_convert_area1_to_nchw_batch_range(state, begin, end);
      } else {
        htp_ops_tensor_convert_area1_to_nc4hw4_cp_range(state, begin, end);
      }
    } else {
      htp_ops_tensor_convert_pack_area_range(state, begin, end, workspace);
    }
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void htp_ops_tensor_convert_c1_to_nc4hw4_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsTensorConvertPackTaskState* state = (HtpOpsTensorConvertPackTaskState*)data;
  const int total = state->batch * state->area;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    if (begin >= total) {
      break;
    }
    int end = begin + state->grain;
    if (end > total) {
      end = total;
    }
    htp_ops_tensor_convert_c1_to_nc4hw4_range(state, begin, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_tensor_convert_pack_fp16(uint8_t* dst, const uint8_t* src,
                                                    int batch, int area, int channel, int direction) {
  const int pack = 64;
  const int c4 = (channel + pack - 1) / pack;
  if (direction == 1 && channel == 1 && area > 1) {
    const int elemCount = batch * area;
    const size_t workBytes = (size_t)elemCount * pack * sizeof(uint16_t);
    const int nTasks = htp_ops_tensor_convert_pick_task_count(workBytes, elemCount);
    HtpOpsTensorConvertPackTaskState state = {};
    state.dst = dst;
    state.src = src;
    state.batch = batch;
    state.area = area;
    state.channel = channel;
    state.c4 = c4;
    state.direction = direction;
    state.grain = (elemCount + nTasks - 1) / nTasks;
    state.task_id = 0;
    if (nTasks <= 1) {
      htp_ops_tensor_convert_c1_to_nc4hw4_range(&state, 0, elemCount);
      return true;
    }
    worker_pool_job_t job;
    job.fptr = htp_ops_tensor_convert_c1_to_nc4hw4_worker;
    job.dptr = &state;
    worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
    for (int i = 0; i < nTasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
    return true;
  }
  if (area > pack) {
    const int blockCount = (area + pack - 1) / pack;
    const int unitCount = batch * c4 * blockCount;
    const size_t workBytes = (size_t)batch * area * c4 * pack * sizeof(uint16_t);
    const int nTasks = htp_ops_tensor_convert_pick_task_count(workBytes, unitCount);
    HtpOpsTensorConvertPackTaskState state = {};
    state.dst = dst;
    state.src = src;
    state.batch = batch;
    state.area = area;
    state.channel = channel;
    state.c4 = c4;
    state.direction = direction;
    state.grain = (unitCount + nTasks - 1) / nTasks;
    state.task_id = 0;
    state.workspaces = htp_ops_tensor_convert_pack_workspaces();
    if (state.workspaces == NULL) {
      return false;
    }
    if (nTasks <= 1) {
      htp_ops_tensor_convert_c4_1_area_blocks_range(&state, 0, unitCount, state.workspaces);
      return true;
    }
    worker_pool_job_t job;
    job.fptr = htp_ops_tensor_convert_c4_1_area_worker;
    job.dptr = &state;
    worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
    for (int i = 0; i < nTasks; ++i) {
      worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&(state.sync_ctx));
    return true;
  }
  const int total = area == 1 ? (direction == 0 ? batch : c4) : batch * c4;
  const size_t workBytes = (size_t)batch * area * c4 * pack * sizeof(uint16_t);
  const int nTasks = htp_ops_tensor_convert_pick_task_count(workBytes, total);
  HtpOpsTensorConvertPackTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.batch = batch;
  state.area = area;
  state.channel = channel;
  state.c4 = c4;
  state.direction = direction;
  state.grain = (total + nTasks - 1) / nTasks;
  state.task_id = 0;
  if (area != 1) {
    state.workspaces = htp_ops_tensor_convert_pack_workspaces();
    if (state.workspaces == NULL) {
      return false;
    }
  }
  if (nTasks <= 1) {
    if (area == 1) {
      if (direction == 0) {
        htp_ops_tensor_convert_area1_to_nchw_batch_range(&state, 0, total);
      } else {
        htp_ops_tensor_convert_area1_to_nc4hw4_cp_range(&state, 0, total);
      }
    } else {
      htp_ops_tensor_convert_pack_area_range(&state, 0, total, state.workspaces);
    }
    return true;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_tensor_convert_pack_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

AEEResult htp_ops_tensor_convert(uint8_t* dst, uint8_t* src,
                                 int32_t batch, int32_t area, int32_t c,
                                 int32_t bytes, int32_t convertType) {
  if (batch <= 0 || area <= 0 || c <= 0) {
    return 0;
  }
  if (bytes != 1 && bytes != 2 && bytes != 4) {
    return -1;
  }

  const uint8_t* srcBase = src;
  uint8_t* dstBase = dst;

  int pack = 4;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif
  int c4 = (c + pack - 1) / pack;
  size_t srcBytes = 0;
  size_t dstBytes = 0;
  if (convertType == HTP_OPS_CONVERT_NC4HW4_TO_NCHW) {
    srcBytes = (size_t)batch * area * c4 * pack * bytes;
    dstBytes = (size_t)batch * area * c * bytes;
  } else if (convertType == HTP_OPS_CONVERT_NCHW_TO_NC4HW4) {
    srcBytes = (size_t)batch * area * c * bytes;
    dstBytes = (size_t)batch * area * c4 * pack * bytes;
  } else if (convertType == HTP_OPS_CONVERT_NC4HW4_FLATTEN_CW) {
    int dstC4 = (c * area + pack - 1) / pack;
    srcBytes = (size_t)batch * area * c4 * pack * bytes;
    dstBytes = (size_t)batch * dstC4 * pack * bytes;
  } else {
    srcBytes = (size_t)batch * area * c * bytes;
    dstBytes = srcBytes;
  }
  if (convertType == HTP_OPS_CONVERT_NC4HW4_FLATTEN_CW) {
    if (bytes == 2 && pack == 64 &&
        htp_ops_tensor_convert_flatten_cw_fp16(dstBase, srcBase, batch, area, c)) {
      return 0;
    }
    const int srcArea = batch * area;
    const int dstC = c * area;
    const int dstC4 = (dstC + pack - 1) / pack;
    if (dstC != dstC4 * pack) {
      memset(dstBase, 0, dstBytes);
    }
    uint8_t* vtcmPtr = (bytes == 2 && pack == 64)
                           ? htp_ops_tensor_convert_alloc_bytes((size_t)__HVX_LENGTH__ +
                                                                (size_t)dstC4 * pack * sizeof(int32_t))
                           : NULL;
    if (bytes == 2 && pack == 64 && vtcmPtr != NULL) {
      HVX_Vector* v = (HVX_Vector*)vtcmPtr;
      __fp16* tile = (__fp16*)v;
      int32_t* srcOffset = (int32_t*)(vtcmPtr + __HVX_LENGTH__);
      for (int cp = 0; cp < dstC4; ++cp) {
        const int cOutBase = cp * pack;
        int srcC = cOutBase / area;
        int srcInner = cOutBase - srcC * area;
        for (int k = 0; k < pack; ++k) {
          int cOut = cOutBase + k;
          if (cOut < dstC) {
            srcOffset[cp * pack + k] = (int32_t)((size_t)(srcC / pack) * srcArea * pack +
                                                 (size_t)srcInner * pack + (srcC % pack));
            ++srcInner;
            if (srcInner == area) {
              srcInner = 0;
              ++srcC;
            }
          } else {
            srcOffset[cp * pack + k] = -1;
          }
        }
      }
      for (int ni = 0; ni < batch; ++ni) {
        const int32_t batchSrcOffset = ni * area * pack;
        for (int cp = 0; cp < dstC4; ++cp) {
          const int32_t* offsetBase = srcOffset + cp * pack;
          for (int k = 0; k < pack; ++k) {
            const int32_t baseOffset = offsetBase[k];
            if (baseOffset >= 0) {
              tile[k] = *((const __fp16*)srcBase + (size_t)(baseOffset + batchSrcOffset));
            } else {
              tile[k] = (__fp16)0.0f;
            }
          }
          size_t dstIdx = (size_t)(cp * batch + ni) * pack;
          vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = v[0];
        }
      }
    } else {
      for (int ni = 0; ni < batch; ++ni) {
        for (int cOut = 0; cOut < dstC; ++cOut) {
          int srcC = cOut / area;
          int srcInner = cOut - srcC * area;
          int srcHw = ni * area + srcInner;
          size_t srcIdx = (size_t)(srcC / pack) * srcArea * pack + (size_t)srcHw * pack + (srcC % pack);
          size_t dstIdx = (size_t)((cOut / pack) * batch + ni) * pack + (cOut % pack);
          if (bytes == 4) {
            *((float*)dstBase + dstIdx) = *((const float*)srcBase + srcIdx);
          } else if (bytes == 2) {
            *((__fp16*)dstBase + dstIdx) = *((const __fp16*)srcBase + srcIdx);
          } else {
            *(dstBase + dstIdx) = *(srcBase + srcIdx);
          }
        }
      }
    }
  } else if (convertType == HTP_OPS_CONVERT_COPY) {
    if (dstBase != srcBase) {
      htp_ops_tensor_convert_copy_parallel(dstBase, srcBase, srcBytes);
    }
  } else if (convertType == HTP_OPS_CONVERT_NC4HW4_TO_NCHW) {
    if (bytes == 2 && pack == 64 && area == 1 && batch == 1 && c == c4 * pack) {
      if (dstBase != srcBase) {
        htp_ops_tensor_convert_copy_hvx(dstBase, srcBase, srcBytes);
      }
      return 0;
    }
    if (bytes == 2 && pack == 64 &&
        htp_ops_tensor_convert_pack_fp16(dstBase, srcBase, batch, area, c, 0)) {
      return 0;
    }
    if (bytes == 2 && pack == 64 && area == 1) {
      const int cFull = c / pack;
      const int cRemain = c - cFull * pack;
      if (batch == 1 && cRemain == 0) {
        memcpy(dstBase, srcBase, srcBytes);
        return 0;
      }
      for (int ni = 0; ni < batch; ++ni) {
        for (int cp = 0; cp < cFull; ++cp) {
          size_t srcIdx = (size_t)(cp * batch + ni) * pack;
          size_t dstIdx = (size_t)ni * c + (size_t)cp * pack;
          vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = vmemu((HVX_Vector*)((const __fp16*)srcBase + srcIdx));
        }
        if (cRemain > 0) {
          size_t srcIdx = (size_t)(cFull * batch + ni) * pack;
          size_t dstIdx = (size_t)ni * c + (size_t)cFull * pack;
          memcpy((__fp16*)dstBase + dstIdx, (const __fp16*)srcBase + srcIdx, (size_t)cRemain * sizeof(__fp16));
        }
      }
    } else if (bytes == 2 && pack == 64 && c % pack == 0) {
      if (area == 1) {
        if (batch == 1) {
          memcpy(dstBase, srcBase, srcBytes);
          return 0;
        }
        for (int ni = 0; ni < batch; ++ni) {
          for (int cp = 0; cp < c4; ++cp) {
            size_t srcIdx = (size_t)(cp * batch + ni) * pack;
            size_t dstIdx = (size_t)ni * c + (size_t)cp * pack;
            vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = vmemu((HVX_Vector*)((const __fp16*)srcBase + srcIdx));
          }
        }
      } else {
        HVX_Vector* v = htp_ops_tensor_convert_alloc_workspace();
        if (v == NULL) {
          return -1;
        }
        for (int ni = 0; ni < batch; ++ni) {
          for (int cp = 0; cp < c4; ++cp) {
            int w_main = area / pack;
            int w_rem = area % pack;
            for (int w_blk = 0; w_blk < w_main; ++w_blk) {
              int wi = w_blk * pack;
              for (int k = 0; k < pack; ++k) {
                size_t srcIdx = (size_t)(cp * batch + ni) * area * pack + (size_t)(wi + k) * pack;
                v[k] = vmemu((HVX_Vector*)((const __fp16*)srcBase + srcIdx));
              }
              hvx_transpose_64x64(v);
              for (int k = 0; k < pack; ++k) {
                int ci = cp * pack + k;
                if (ci < c) {
                  size_t dstIdx = (size_t)(ni * c + ci) * area + (size_t)wi;
                  vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = v[k];
                }
              }
            }
            if (w_rem > 0) {
              int wi = w_main * pack;
              for (int w_i = wi; w_i < area; ++w_i) {
                for (int ci = cp * pack; ci < cp * pack + pack && ci < c; ++ci) {
                  size_t srcIdx = (size_t)(cp * batch + ni) * area * pack + (size_t)w_i * pack + (ci % pack);
                  size_t dstIdx = (size_t)(ni * c + ci) * area + (size_t)w_i;
                  *((__fp16*)dstBase + dstIdx) = *((const __fp16*)srcBase + srcIdx);
                }
              }
            }
          }
        }
      }
    } else {
      for (int ni = 0; ni < batch; ++ni) {
        for (int wi = 0; wi < area; ++wi) {
          for (int ci = 0; ci < c; ++ci) {
            size_t srcIdx = (size_t)((ci / pack) * batch + ni) * area * pack + (size_t)wi * pack + (ci % pack);
            size_t dstIdx = (size_t)(ni * c + ci) * area + (size_t)wi;
            if (bytes == 4) {
              *((float*)dstBase + dstIdx) = *((const float*)srcBase + srcIdx);
            } else if (bytes == 2) {
              *((__fp16*)dstBase + dstIdx) = *((const __fp16*)srcBase + srcIdx);
            } else {
              *(dstBase + dstIdx) = *(srcBase + srcIdx);
            }
          }
        }
      }
    }
  } else {
    if (bytes == 2 && pack == 64 && area == 1 && batch == 1 && c == c4 * pack) {
      if (dstBase != srcBase) {
        htp_ops_tensor_convert_copy_hvx(dstBase, srcBase, srcBytes);
      }
      return 0;
    }
    if (bytes == 2 && pack == 64 &&
        htp_ops_tensor_convert_pack_fp16(dstBase, srcBase, batch, area, c, 1)) {
      return 0;
    }
    if (bytes == 2 && pack == 64 && area == 1) {
      const int cFull = c / pack;
      const int cRemain = c - cFull * pack;
      if (batch == 1 && cRemain == 0) {
        memcpy(dstBase, srcBase, srcBytes);
        return 0;
      }
      for (int ni = 0; ni < batch; ++ni) {
        for (int cp = 0; cp < cFull; ++cp) {
          size_t srcIdx = (size_t)ni * c + (size_t)cp * pack;
          size_t dstIdx = (size_t)(cp * batch + ni) * pack;
          vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = vmemu((HVX_Vector*)((const __fp16*)srcBase + srcIdx));
        }
        if (cRemain > 0) {
          size_t srcIdx = (size_t)ni * c + (size_t)cFull * pack;
          size_t dstIdx = (size_t)(cFull * batch + ni) * pack;
          HVX_Vector vZero = Q6_V_vzero();
          vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = vZero;
          memcpy((__fp16*)dstBase + dstIdx, (const __fp16*)srcBase + srcIdx, (size_t)cRemain * sizeof(__fp16));
        }
      }
    } else if (bytes == 2 && pack == 64 && c % pack == 0) {
      {
        HVX_Vector* v = htp_ops_tensor_convert_alloc_workspace();
        if (v == NULL) {
          return -1;
        }
        for (int ni = 0; ni < batch; ++ni) {
          for (int cp = 0; cp < c4; ++cp) {
            int w_main = area / pack;
            int w_rem = area % pack;
            for (int w_blk = 0; w_blk < w_main; ++w_blk) {
              int wi = w_blk * pack;
              for (int k = 0; k < pack; ++k) {
                int ci = cp * pack + k;
                if (ci < c) {
                  size_t srcIdx = (size_t)(ni * c + ci) * area + (size_t)wi;
                  v[k] = vmemu((HVX_Vector*)((const __fp16*)srcBase + srcIdx));
                } else {
                  v[k] = Q6_V_vzero();
                }
              }
              hvx_transpose_64x64(v);
              for (int k = 0; k < pack; ++k) {
                size_t dstIdx = (size_t)(cp * batch + ni) * area * pack + (size_t)(wi + k) * pack;
                vmemu((HVX_Vector*)((__fp16*)dstBase + dstIdx)) = v[k];
              }
            }
            if (w_rem > 0) {
              int wi = w_main * pack;
              for (int w_i = wi; w_i < area; ++w_i) {
                for (int ci = cp * pack; ci < cp * pack + pack && ci < c; ++ci) {
                  size_t srcIdx = (size_t)(ni * c + ci) * area + (size_t)w_i;
                  size_t dstIdx = (size_t)(cp * batch + ni) * area * pack + (size_t)w_i * pack + (ci % pack);
                  *((__fp16*)dstBase + dstIdx) = *((const __fp16*)srcBase + srcIdx);
                }
              }
            }
          }
        }
      }
    } else {
      memset(dstBase, 0, dstBytes);
      for (int ni = 0; ni < batch; ++ni) {
        for (int wi = 0; wi < area; ++wi) {
          for (int ci = 0; ci < c; ++ci) {
            size_t srcIdx = (size_t)(ni * c + ci) * area + (size_t)wi;
            size_t dstIdx = (size_t)((ci / pack) * batch + ni) * area * pack + (size_t)wi * pack + (ci % pack);
            if (bytes == 4) {
              *((float*)dstBase + dstIdx) = *((const float*)srcBase + srcIdx);
            } else if (bytes == 2) {
              *((__fp16*)dstBase + dstIdx) = *((const __fp16*)srcBase + srcIdx);
            } else {
              *(dstBase + dstIdx) = *(srcBase + srcIdx);
            }
          }
        }
      }
    }
  }

  return 0;
}

}  // extern "C"
