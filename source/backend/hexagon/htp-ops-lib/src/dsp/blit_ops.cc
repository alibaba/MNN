#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <qurt_memory.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <remote.h>

#include <string.h>

#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
#include "dsp/mmap_mgr.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "transpose_hvx.h"

#include "region_ops.h"

extern "C" {


// ===== raster / unary / binary blit (Loop support) =====

static inline int htp_ops_raster_pick_task_count(size_t bytes, int units) {
  if (units <= 1 || g_max_num_workers <= 1 || bytes < 2048) {
    return 1;
  }
  int n = (int)g_max_num_workers;
  if (n > units) {
    n = units;
  }
  return n;
}

static inline bool htp_ops_prepare_transpose(const HtpOpsRasterRegion* region, int dims[4], int* keepDim) {
  int srcOne = -1;
  int dstOne = -1;
  *keepDim = -1;
  dims[0] = 1;
  dims[1] = 1;
  dims[2] = 0;
  dims[3] = 0;
  for (int i = 0; i < 3; ++i) {
    if (region->srcStride[i] == 1 && region->size[i] != 1) {
      if (srcOne >= 0) {
        return false;
      }
      srcOne = i;
      dims[1] = region->size[i];
      dims[3] = region->dstStride[i];
    } else if (region->dstStride[i] == 1 && region->size[i] != 1) {
      if (dstOne >= 0) {
        return false;
      }
      dstOne = i;
      dims[0] = region->size[i];
      dims[2] = region->srcStride[i];
    } else {
      *keepDim = i;
    }
  }
  return srcOne >= 0 && dstOne >= 0 && srcOne != dstOne && *keepDim >= 0 &&
         dims[2] > 0 && dims[3] > 0;
}

static inline void htp_ops_transpose_16_hvx_range(uint8_t* dstBase, const uint8_t* srcBase,
                                                  const HtpOpsRasterRegion* region,
                                                  const int dims[4], int keepDim,
                                                  int zStart, int zEnd) {
  const int rows = dims[0];
  const int cols = dims[1];
  const int srcStride = dims[2];
  const int dstStride = dims[3];
  const int rowMain = rows & (~63);
  const int colMain = cols & (~63);
  const __fp16* src = (const __fp16*)srcBase;
  __fp16* dst = (__fp16*)dstBase;
  HVX_Vector v[64];
  for (int z = zStart; z < zEnd; ++z) {
    const __fp16* srcZ = src + region->srcStride[keepDim] * z;
    __fp16* dstZ = dst + region->dstStride[keepDim] * z;
    for (int row = 0; row < rowMain; row += 64) {
      for (int col = 0; col < colMain; col += 64) {
        for (int k = 0; k < 64; ++k) {
          v[k] = vmemu((const HVX_Vector*)(srcZ + (row + k) * srcStride + col));
        }
        hvx_transpose_64x64(v);
        for (int k = 0; k < 64; ++k) {
          vmemu((HVX_Vector*)(dstZ + (col + k) * dstStride + row)) = v[k];
        }
      }
    }
    if (rowMain < rows && colMain > 0) {
      const int rowTail = rows - rowMain;
      for (int col = 0; col < colMain; col += 64) {
        for (int k = 0; k < rowTail; ++k) {
          v[k] = vmemu((const HVX_Vector*)(srcZ + (rowMain + k) * srcStride + col));
        }
        for (int k = rowTail; k < 64; ++k) {
          v[k] = Q6_V_vzero();
        }
        hvx_transpose_64x64(v);
        for (int k = 0; k < 64; ++k) {
          vstu_variable(dstZ + (col + k) * dstStride + rowMain, (uint32_t)((size_t)rowTail * sizeof(__fp16)), v[k]);
        }
      }
    }
    if (colMain < cols) {
      const int colTailBytes = (cols - colMain) * (int)sizeof(__fp16);
      for (int row = 0; row < rows; row += 64) {
        const int rowCount = rows - row > 64 ? 64 : rows - row;
        for (int k = 0; k < rowCount; ++k) {
          const __fp16* srcTail = srcZ + (row + k) * srcStride + colMain;
          if (row + k + 1 < rows) {
            v[k] = vmemu((const HVX_Vector*)srcTail);
          } else {
            v[k] = Q6_V_vzero();
            memcpy(&v[k], srcTail, (size_t)colTailBytes);
          }
        }
        for (int k = rowCount; k < 64; ++k) {
          v[k] = Q6_V_vzero();
        }
        hvx_transpose_64x64(v);
        for (int k = 0; k < cols - colMain; ++k) {
          if (rowCount == 64) {
            vmemu((HVX_Vector*)(dstZ + (colMain + k) * dstStride + row)) = v[k];
          } else {
            vstu_variable(dstZ + (colMain + k) * dstStride + row,
                          (uint32_t)((size_t)rowCount * sizeof(__fp16)), v[k]);
          }
        }
      }
    }
  }
}

static inline void htp_ops_transpose_16_hvx_block(uint8_t* dstBase, const uint8_t* srcBase,
                                                  const HtpOpsRasterRegion* region,
                                                  const int dims[4], int keepDim,
                                                  int z, int row, int col) {
  const int rows = dims[0];
  const int cols = dims[1];
  const int srcStride = dims[2];
  const int dstStride = dims[3];
  const int rowCount = rows - row > 64 ? 64 : rows - row;
  const int colCount = cols - col > 64 ? 64 : cols - col;
  const __fp16* srcZ = (const __fp16*)srcBase + region->srcStride[keepDim] * z;
  __fp16* dstZ = (__fp16*)dstBase + region->dstStride[keepDim] * z;
  HVX_Vector v[64];

  if (rowCount <= 8 || (row > 0 && row + rowCount == rows && rowCount <= 40)) {
    for (int c = 0; c < colCount; ++c) {
      __fp16* dstCol = dstZ + (col + c) * dstStride + row;
      for (int k = 0; k < rowCount; ++k) {
        dstCol[k] = srcZ[(row + k) * srcStride + col + c];
      }
    }
    return;
  }

  if (colCount == 64) {
    for (int k = 0; k < rowCount; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + (row + k) * srcStride + col));
    }
  } else {
    const int colTailBytes = colCount * (int)sizeof(__fp16);
    for (int k = 0; k < rowCount; ++k) {
      const __fp16* srcTail = srcZ + (row + k) * srcStride + col;
      if (row + k + 1 < rows) {
        v[k] = vmemu((const HVX_Vector*)srcTail);
      } else {
        v[k] = Q6_V_vzero();
        memcpy(&v[k], srcTail, (size_t)colTailBytes);
      }
    }
  }
  for (int k = rowCount; k < 64; ++k) {
    v[k] = Q6_V_vzero();
  }
  hvx_transpose_64x64(v);
  for (int k = 0; k < colCount; ++k) {
    __fp16* dstRow = dstZ + (col + k) * dstStride + row;
    if (rowCount == 64) {
      vmemu((HVX_Vector*)dstRow) = v[k];
    } else {
      vstu_variable(dstRow, (uint32_t)(rowCount * (int)sizeof(__fp16)), v[k]);
    }
  }
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int dims[4];
  int keepDim;
  int grain;
  int zCount;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterTransposeTaskState;

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int dims[4];
  int keepDim;
  int rowBlocks;
  int colBlocks;
  int totalBlocks;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterTransposeBlockTaskState;

static void htp_ops_transpose_16_hvx_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterTransposeTaskState* state = (HtpOpsRasterTransposeTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int zStart = taskId * state->grain;
    if (zStart >= state->zCount) {
      break;
    }
    int zEnd = zStart + state->grain;
    if (zEnd > state->zCount) {
      zEnd = state->zCount;
    }
    htp_ops_transpose_16_hvx_range(state->dstBase, state->srcBase, state->region,
                                   state->dims, state->keepDim, zStart, zEnd);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void htp_ops_transpose_16_hvx_block_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterTransposeBlockTaskState* state = (HtpOpsRasterTransposeBlockTaskState*)data;
  const int blocksPerZ = state->rowBlocks * state->colBlocks;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    int begin = taskId * state->grain;
    if (begin >= state->totalBlocks) {
      break;
    }
    int end = begin + state->grain;
    if (end > state->totalBlocks) {
      end = state->totalBlocks;
    }
    for (int index = begin; index < end; ++index) {
      const int z = index / blocksPerZ;
      const int rem = index - z * blocksPerZ;
      const int rowBlock = rem / state->colBlocks;
      const int colBlock = rem - rowBlock * state->colBlocks;
      htp_ops_transpose_16_hvx_block(state->dstBase, state->srcBase, state->region,
                                     state->dims, state->keepDim, z,
                                     rowBlock * 64, colBlock * 64);
    }
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline void htp_ops_transpose_16_hvx_blocks(uint8_t* dstBase, const uint8_t* srcBase,
                                                   const HtpOpsRasterRegion* region,
                                                   const int dims[4], int keepDim,
                                                   int zCount, int nTasks) {
  HtpOpsRasterTransposeBlockTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = region;
  state.keepDim = keepDim;
  state.rowBlocks = (dims[0] + 63) / 64;
  state.colBlocks = (dims[1] + 63) / 64;
  state.totalBlocks = zCount * state.rowBlocks * state.colBlocks;
  state.grain = (state.totalBlocks + nTasks - 1) / nTasks;
  state.task_id = 0;
  for (int i = 0; i < 4; ++i) {
    state.dims[i] = dims[i];
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_transpose_16_hvx_block_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static inline void htp_ops_transpose_16_hvx(uint8_t* dstBase, const uint8_t* srcBase, const HtpOpsRasterRegion* region) {
  int dims[4];
  int keepDim = -1;
  if (!htp_ops_prepare_transpose(region, dims, &keepDim)) {
    return;
  }
  const int zCount = region->size[keepDim];
  const size_t workBytes = (size_t)zCount * dims[0] * dims[1] * sizeof(__fp16);
  const int totalBlocks = zCount * ((dims[0] + 63) / 64) * ((dims[1] + 63) / 64);
  const int blockTasks = htp_ops_raster_pick_task_count(workBytes, totalBlocks);
  if (blockTasks > 1 && totalBlocks >= blockTasks * 2 && (zCount < blockTasks || dims[0] > 128 || dims[1] > 128)) {
    htp_ops_transpose_16_hvx_blocks(dstBase, srcBase, region, dims, keepDim, zCount, blockTasks);
    return;
  }
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, zCount);
  if (nTasks <= 1) {
    htp_ops_transpose_16_hvx_range(dstBase, srcBase, region, dims, keepDim, 0, zCount);
    return;
  }

  HtpOpsRasterTransposeTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = region;
  state.keepDim = keepDim;
  state.grain = (zCount + nTasks - 1) / nTasks;
  state.zCount = zCount;
  state.task_id = 0;
  for (int i = 0; i < 4; ++i) {
    state.dims[i] = dims[i];
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_transpose_16_hvx_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

typedef void (*HtpOpsRasterBlitProc)(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                                     ptrdiff_t dstStrideBytes, int32_t bytes);

static void htp_ops_blit_1(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                           ptrdiff_t dstStrideBytes, int32_t bytes) {
  (void)bytes;
  for (int32_t x = 0; x < size; ++x) {
    *dst = *src;
    src += srcStrideBytes;
    dst += dstStrideBytes;
  }
}

static void htp_ops_blit_2(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                           ptrdiff_t dstStrideBytes, int32_t bytes) {
  (void)bytes;
  for (int32_t x = 0; x < size; ++x) {
    *(uint16_t*)dst = *(const uint16_t*)src;
    src += srcStrideBytes;
    dst += dstStrideBytes;
  }
}

static void htp_ops_blit_4(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                           ptrdiff_t dstStrideBytes, int32_t bytes) {
  (void)bytes;
  for (int32_t x = 0; x < size; ++x) {
    *(uint32_t*)dst = *(const uint32_t*)src;
    src += srcStrideBytes;
    dst += dstStrideBytes;
  }
}

static void htp_ops_blit_128(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                             ptrdiff_t dstStrideBytes, int32_t bytes) {
  (void)bytes;
  for (int32_t x = 0; x < size; ++x) {
    HVX_Vector v = vmemu((const HVX_Vector*)src);
    vmemu((HVX_Vector*)dst) = v;
    src += srcStrideBytes;
    dst += dstStrideBytes;
  }
}

static void htp_ops_blit_default(uint8_t* dst, const uint8_t* src, int32_t size, ptrdiff_t srcStrideBytes,
                                 ptrdiff_t dstStrideBytes, int32_t bytes) {
  for (int32_t x = 0; x < size; ++x) {
    memcpy(dst, src, (size_t)bytes);
    src += srcStrideBytes;
    dst += dstStrideBytes;
  }
}

static inline HtpOpsRasterBlitProc htp_ops_select_blit_proc(int32_t bytes) {
  if (bytes == 1) {
    return htp_ops_blit_1;
  }
  if (bytes == 2) {
    return htp_ops_blit_2;
  }
  if (bytes == 4) {
    return htp_ops_blit_4;
  }
  if (bytes == __HVX_LENGTH__) {
    return htp_ops_blit_128;
  }
  return htp_ops_blit_default;
}

static inline void htp_ops_copy_bytes_scalar(uint8_t* dst, const uint8_t* src, size_t bytes) {
  if ((((uintptr_t)dst | (uintptr_t)src) & (sizeof(uint64_t) - 1)) == 0) {
    while (bytes >= sizeof(uint64_t)) {
      *(uint64_t*)dst = *(const uint64_t*)src;
      dst += sizeof(uint64_t);
      src += sizeof(uint64_t);
      bytes -= sizeof(uint64_t);
    }
  }
  if ((((uintptr_t)dst | (uintptr_t)src) & (sizeof(uint32_t) - 1)) == 0) {
    while (bytes >= sizeof(uint32_t)) {
      *(uint32_t*)dst = *(const uint32_t*)src;
      dst += sizeof(uint32_t);
      src += sizeof(uint32_t);
      bytes -= sizeof(uint32_t);
    }
  }
  if ((((uintptr_t)dst | (uintptr_t)src) & (sizeof(uint16_t) - 1)) == 0) {
    while (bytes >= sizeof(uint16_t)) {
      *(uint16_t*)dst = *(const uint16_t*)src;
      dst += sizeof(uint16_t);
      src += sizeof(uint16_t);
      bytes -= sizeof(uint16_t);
    }
  }
  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = src[i];
  }
}

static inline void htp_ops_copy_bytes_hvx(uint8_t* dst, const uint8_t* src, size_t rowBytes, bool allowOverread) {
  if (rowBytes >= (size_t)__HVX_LENGTH__) {
    const size_t vecCount = rowBytes / (size_t)__HVX_LENGTH__;
    const size_t tailBytes = rowBytes - vecCount * (size_t)__HVX_LENGTH__;
    for (size_t i = 0; i < vecCount; ++i) {
      HVX_Vector v = vmemu((const HVX_Vector*)(src + i * (size_t)__HVX_LENGTH__));
      vmemu((HVX_Vector*)(dst + i * (size_t)__HVX_LENGTH__)) = v;
    }
    if (tailBytes > 0) {
      uint8_t* tailDst = dst + vecCount * (size_t)__HVX_LENGTH__;
      const uint8_t* tailSrc = src + vecCount * (size_t)__HVX_LENGTH__;
      if (allowOverread) {
        HVX_Vector v = vmemu((const HVX_Vector*)tailSrc);
        vstu_variable(tailDst, (uint32_t)tailBytes, v);
      } else {
        memcpy(tailDst, tailSrc, tailBytes);
      }
    }
  } else {
    if (allowOverread && rowBytes >= 32) {
      HVX_Vector v = vmemu((const HVX_Vector*)src);
      vstu_variable(dst, (uint32_t)rowBytes, v);
    } else {
      htp_ops_copy_bytes_scalar(dst, src, rowBytes);
    }
  }
}

typedef struct {
  uint8_t* dst;
  const uint8_t* src;
  size_t bytes;
  size_t grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterLinearCopyTaskState;

static void htp_ops_linear_copy_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterLinearCopyTaskState* state = (HtpOpsRasterLinearCopyTaskState*)data;
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
    htp_ops_copy_bytes_hvx(state->dst + begin, state->src + begin, end - begin, false);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_parallel_linear_copy(uint8_t* dst, const uint8_t* src, size_t bytes) {
  if (g_max_num_workers <= 1 || bytes < 32768 || dst == src) {
    return false;
  }
  int nTasks = (int)g_max_num_workers;
  const size_t minBytesPerTask = 32768;
  const size_t bySize = (bytes + minBytesPerTask - 1) / minBytesPerTask;
  if ((size_t)nTasks > bySize) {
    nTasks = (int)bySize;
  }
  if (nTasks <= 1) {
    return false;
  }

  HtpOpsRasterLinearCopyTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.bytes = bytes;
  state.grain = ((bytes + (size_t)nTasks - 1) / (size_t)nTasks + 127) & ~(size_t)127;
  state.task_id = 0;

  worker_pool_job_t job;
  job.fptr = htp_ops_linear_copy_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int direction;
  int yCount;
  int zCount;
  int grain;
  bool splitZ;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterPackArea1TaskState;

typedef struct {
  uint8_t* dst;
  uint8_t** src;
  const HtpOpsRasterRegion* regions;
  int regionCount;
  int zCount;
  int batch;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterPackArea1GroupTaskState;

typedef struct {
  HtpOpsRasterPackArea1TaskState* state;
  int begin;
  int end;
} HtpOpsRasterPackArea1FixedTask;

static inline void htp_ops_pack_area1_blit_z_range(HtpOpsRasterPackArea1TaskState* state, int zStart, int zEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const ptrdiff_t unitBytes = 2;
  const int batch = r->size[1];
  if (state->direction == 0) {
    for (int z = zStart; z < zEnd; ++z) {
      const uint8_t* srcZ = state->srcBase + (ptrdiff_t)z * r->srcStride[0] * unitBytes;
      uint8_t* dstZ = state->dstBase + (ptrdiff_t)z * 64 * unitBytes;
      l2fetch(srcZ, 64 * unitBytes, 64 * unitBytes, batch, 0);
      int y = 0;
      for (; y + 3 < batch; y += 4) {
        HVX_Vector v0 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 0) * 64 * unitBytes));
        HVX_Vector v1 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 1) * 64 * unitBytes));
        HVX_Vector v2 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 2) * 64 * unitBytes));
        HVX_Vector v3 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 3) * 64 * unitBytes));
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 0) * r->dstStride[1] * unitBytes)) = v0;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 1) * r->dstStride[1] * unitBytes)) = v1;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 2) * r->dstStride[1] * unitBytes)) = v2;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 3) * r->dstStride[1] * unitBytes)) = v3;
      }
      for (; y < batch; ++y) {
        HVX_Vector v = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)y * 64 * unitBytes));
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)y * r->dstStride[1] * unitBytes)) = v;
      }
    }
  } else {
    for (int z = zStart; z < zEnd; ++z) {
      const uint8_t* srcZ = state->srcBase + (ptrdiff_t)z * 64 * unitBytes;
      uint8_t* dstZ = state->dstBase + (ptrdiff_t)z * r->dstStride[0] * unitBytes;
      l2fetch(srcZ, r->srcStride[1] * unitBytes, 64 * unitBytes, batch, 0);
      int y = 0;
      for (; y + 3 < batch; y += 4) {
        HVX_Vector v0 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 0) * r->srcStride[1] * unitBytes));
        HVX_Vector v1 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 1) * r->srcStride[1] * unitBytes));
        HVX_Vector v2 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 2) * r->srcStride[1] * unitBytes));
        HVX_Vector v3 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 3) * r->srcStride[1] * unitBytes));
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 0) * 64 * unitBytes)) = v0;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 1) * 64 * unitBytes)) = v1;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 2) * 64 * unitBytes)) = v2;
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 3) * 64 * unitBytes)) = v3;
      }
      for (; y < batch; ++y) {
        HVX_Vector v = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)y * r->srcStride[1] * unitBytes));
        vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)y * 64 * unitBytes)) = v;
      }
    }
  }
}

static inline void htp_ops_pack_area1_blit_range(HtpOpsRasterPackArea1TaskState* state, int yStart, int yEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const int inner = r->size[2];
  const ptrdiff_t unitBytes = 2;
  if (state->direction == 0) {
    for (int y = yStart; y < yEnd; ++y) {
      const uint8_t* srcY = state->srcBase + (ptrdiff_t)y * 64 * unitBytes;
      uint8_t* dstY = state->dstBase + (ptrdiff_t)y * r->dstStride[1] * unitBytes;
      if (inner == 64) {
        for (int z = 0; z < r->size[0]; ++z) {
          HVX_Vector v = vmemu((const HVX_Vector*)(srcY + (ptrdiff_t)z * r->srcStride[0] * unitBytes));
          vmemu((HVX_Vector*)(dstY + (ptrdiff_t)z * 64 * unitBytes)) = v;
        }
      } else {
        memcpy(dstY, srcY, (size_t)inner * unitBytes);
      }
    }
  } else {
    for (int y = yStart; y < yEnd; ++y) {
      const uint8_t* srcY = state->srcBase + (ptrdiff_t)y * r->srcStride[1] * unitBytes;
      uint8_t* dstY = state->dstBase + (ptrdiff_t)y * 64 * unitBytes;
      if (inner == 64) {
        for (int z = 0; z < r->size[0]; ++z) {
          HVX_Vector v = vmemu((const HVX_Vector*)(srcY + (ptrdiff_t)z * 64 * unitBytes));
          vmemu((HVX_Vector*)(dstY + (ptrdiff_t)z * r->dstStride[0] * unitBytes)) = v;
        }
      } else {
        memcpy(dstY, srcY, (size_t)inner * unitBytes);
      }
    }
  }
}

static void htp_ops_pack_area1_blit_fixed_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterPackArea1FixedTask* task = (HtpOpsRasterPackArea1FixedTask*)data;
  if (task->state->splitZ) {
    htp_ops_pack_area1_blit_z_range(task->state, task->begin, task->end);
  } else {
    htp_ops_pack_area1_blit_range(task->state, task->begin, task->end);
  }
  worker_pool_synctoken_jobdone(&(task->state->sync_ctx));
}

static inline void htp_ops_run_pack_area1_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                               const HtpOpsRasterRegion* r, int direction) {
  const int batch = r->size[1];
  const size_t workBytes = (size_t)r->size[0] * batch * r->size[2] * sizeof(__fp16);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, batch);
  HtpOpsRasterPackArea1TaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.direction = direction;
  state.yCount = batch;
  state.zCount = r->size[0];
  state.splitZ = (r->size[2] == 64 && r->size[0] > 1);
  const int total = state.splitZ ? state.zCount : batch;
  state.grain = (total + nTasks - 1) / nTasks;
  state.task_id = 0;

  if (nTasks <= 1) {
    if (state.splitZ) {
      htp_ops_pack_area1_blit_z_range(&state, 0, state.zCount);
    } else {
      htp_ops_pack_area1_blit_range(&state, 0, batch);
    }
    return;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_pack_area1_blit_fixed_worker;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  HtpOpsRasterPackArea1FixedTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsRasterPackArea1FixedTask, nTasks);
  for (int i = 0; i < nTasks; ++i) {
    const int begin = i * state.grain;
    int end = begin + state.grain;
    if (end > total) {
      end = total;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static inline void htp_ops_pack_area_transpose_nc4hw4_to_nchw_z(uint8_t* dstBase, const uint8_t* srcBase,
                                                                const HtpOpsRasterRegion* r, int z) {
  const int rows = r->size[2];
  const int cols = r->size[1];
  const int rowMain = rows & (~63);
  const __fp16* srcZ = (const __fp16*)srcBase + (ptrdiff_t)z * r->srcStride[0];
  __fp16* dstZ = (__fp16*)dstBase + (ptrdiff_t)z * r->dstStride[0];
  HVX_Vector v[64];
  for (int row = 0; row < rowMain; row += 64) {
    for (int k = 0; k < 64; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + (row + k) * 64));
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < cols; ++k) {
      vmemu((HVX_Vector*)(dstZ + k * r->dstStride[1] + row)) = v[k];
    }
  }
  if (rowMain < rows) {
    const int rowTail = rows - rowMain;
    for (int k = 0; k < rowTail; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + (rowMain + k) * 64));
    }
    for (int k = rowTail; k < 64; ++k) {
      v[k] = Q6_V_vzero();
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < cols; ++k) {
      vstu_variable(dstZ + k * r->dstStride[1] + rowMain, (uint32_t)((size_t)rowTail * sizeof(__fp16)), v[k]);
    }
  }
}

static inline void htp_ops_pack_area_transpose_nchw_to_nc4hw4_z(uint8_t* dstBase, const uint8_t* srcBase,
                                                                const HtpOpsRasterRegion* r, int z) {
  const int cols = r->size[2];
  const int colMain = cols & (~63);
  const __fp16* srcZ = (const __fp16*)srcBase + (ptrdiff_t)z * r->srcStride[0];
  __fp16* dstZ = (__fp16*)dstBase + (ptrdiff_t)z * r->dstStride[0];
  HVX_Vector v[64];
  for (int col = 0; col < colMain; col += 64) {
    for (int k = 0; k < 64; ++k) {
      v[k] = vmemu((const HVX_Vector*)(srcZ + k * r->srcStride[1] + col));
    }
    hvx_transpose_64x64(v);
    for (int k = 0; k < 64; ++k) {
      vmemu((HVX_Vector*)(dstZ + (col + k) * 64)) = v[k];
    }
  }
  if (colMain < cols) {
    const int colTail = cols - colMain;
    if (z + 1 < r->size[0]) {
      for (int k = 0; k < 64; ++k) {
        v[k] = vmemu((const HVX_Vector*)(srcZ + k * r->srcStride[1] + colMain));
      }
      hvx_transpose_64x64(v);
      for (int k = 0; k < colTail; ++k) {
        vmemu((HVX_Vector*)(dstZ + (colMain + k) * 64)) = v[k];
      }
    } else {
      for (int row = 0; row < 64; ++row) {
        const __fp16* srcRow = srcZ + row * r->srcStride[1];
        for (int col = colMain; col < cols; ++col) {
          dstZ[col * 64 + row] = srcRow[col];
        }
      }
    }
  }
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int direction;
  int zCount;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterPackAreaTaskState;

static inline void htp_ops_pack_area_blit_range(HtpOpsRasterPackAreaTaskState* state, int zStart, int zEnd) {
  for (int z = zStart; z < zEnd; ++z) {
    if (state->direction == 0) {
      htp_ops_pack_area_transpose_nc4hw4_to_nchw_z(state->dstBase, state->srcBase, state->region, z);
    } else {
      htp_ops_pack_area_transpose_nchw_to_nc4hw4_z(state->dstBase, state->srcBase, state->region, z);
    }
  }
}

static void htp_ops_pack_area_blit_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterPackAreaTaskState* state = (HtpOpsRasterPackAreaTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int zStart = taskId * state->grain;
    if (zStart >= state->zCount) {
      break;
    }
    int zEnd = zStart + state->grain;
    if (zEnd > state->zCount) {
      zEnd = state->zCount;
    }
    htp_ops_pack_area_blit_range(state, zStart, zEnd);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline void htp_ops_run_pack_area_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                              const HtpOpsRasterRegion* r, int direction) {
  const int zCount = r->size[0];
  const size_t workBytes = (size_t)zCount * r->size[1] * r->size[2] * sizeof(__fp16);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, zCount);
  HtpOpsRasterPackAreaTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.direction = direction;
  state.zCount = zCount;
  state.grain = (zCount + nTasks - 1) / nTasks;
  state.task_id = 0;
  if (nTasks <= 1) {
    htp_ops_pack_area_blit_range(&state, 0, zCount);
    return;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_pack_area_blit_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static inline bool htp_ops_try_pack_area1_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                               const HtpOpsRasterRegion* r, int32_t bytes) {
  if (bytes != 2 || r->srcStride[2] != 1 || r->dstStride[2] != 1 || r->size[1] <= 0 || r->size[2] <= 0) {
    return false;
  }
  const int batch = r->size[1];
  const int inner = r->size[2];

  // NC4HW4(area=1) -> NCHW: src=[cp,batch,64], dst=[batch,channel].
  if (r->srcStride[1] == 64 && r->dstStride[1] >= inner &&
      ((inner == 64 && r->srcStride[0] == batch * 64 && r->dstStride[0] == 64) ||
       (r->size[0] == 1 && r->srcStride[0] == 0 && r->dstStride[0] == 0))) {
    htp_ops_run_pack_area1_blit(dstBase, srcBase, r, 0);
    return true;
  }

  // NCHW -> NC4HW4(area=1): src=[batch,channel], dst=[cp,batch,64].
  if (r->dstStride[1] == 64 && r->srcStride[1] >= inner &&
      ((inner == 64 && r->srcStride[0] == 64 && r->dstStride[0] == batch * 64) ||
       (r->size[0] == 1 && r->srcStride[0] == 0 && r->dstStride[0] == 0))) {
    htp_ops_run_pack_area1_blit(dstBase, srcBase, r, 1);
    return true;
  }
  return false;
}

static inline bool htp_ops_try_pack_area_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                              const HtpOpsRasterRegion* r, int32_t bytes) {
  if (bytes != 2 || r->size[1] <= 0 || r->size[1] > 64 || r->size[2] <= 1) {
    return false;
  }
  const int area = r->size[2];
  if (r->srcStride[1] == 1 && r->srcStride[2] == 64 &&
      r->dstStride[1] == area && r->dstStride[2] == 1 &&
      ((r->size[1] == 64 && r->srcStride[0] == area * 64 && r->dstStride[0] == area * 64) ||
       (r->size[0] == 1 && r->srcStride[0] == 0 && r->dstStride[0] == 0))) {
    htp_ops_run_pack_area_blit(dstBase, srcBase, r, 0);
    return true;
  }
  if (r->size[1] == 64 && r->srcStride[1] == area && r->srcStride[2] == 1 &&
      r->dstStride[1] == 1 && r->dstStride[2] == 64 &&
      r->srcStride[0] == area * 64 && r->dstStride[0] == area * 64) {
    htp_ops_run_pack_area_blit(dstBase, srcBase, r, 1);
    return true;
  }
  return false;
}

static inline void htp_ops_pack_area1_group_range(HtpOpsRasterPackArea1GroupTaskState* state, int indexStart, int indexEnd) {
  const ptrdiff_t unitBytes = 2;
  for (int index = indexStart; index < indexEnd; ++index) {
    const int regionIndex = index / state->zCount;
    const int z = index - regionIndex * state->zCount;
    const HtpOpsRasterRegion* r = state->regions + regionIndex;
    const uint8_t* srcBase = state->src[r->srcIndex] + (ptrdiff_t)r->srcOffset * unitBytes;
    uint8_t* dstBase = state->dst + (ptrdiff_t)r->dstOffset * unitBytes;
    const uint8_t* srcZ = srcBase + (ptrdiff_t)z * r->srcStride[0] * unitBytes;
    uint8_t* dstZ = dstBase + (ptrdiff_t)z * r->dstStride[0] * unitBytes;
    int y = 0;
    for (; y + 1 < state->batch; y += 2) {
      HVX_Vector v0 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 0) * r->srcStride[1] * unitBytes));
      HVX_Vector v1 = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)(y + 1) * r->srcStride[1] * unitBytes));
      vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 0) * 64 * unitBytes)) = v0;
      vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)(y + 1) * 64 * unitBytes)) = v1;
    }
    for (; y < state->batch; ++y) {
      HVX_Vector v = vmemu((const HVX_Vector*)(srcZ + (ptrdiff_t)y * r->srcStride[1] * unitBytes));
      vmemu((HVX_Vector*)(dstZ + (ptrdiff_t)y * 64 * unitBytes)) = v;
    }
  }
}

static void htp_ops_pack_area1_group_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterPackArea1GroupTaskState* state = (HtpOpsRasterPackArea1GroupTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    int begin = taskId * state->grain;
    if (begin >= state->regionCount * state->zCount) {
      break;
    }
    int end = begin + state->grain;
    const int total = state->regionCount * state->zCount;
    if (end > total) {
      end = total;
    }
    htp_ops_pack_area1_group_range(state, begin, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_pack_area1_group_blit(uint8_t* dst, uint8_t** src, int srcNumber,
                                                     const HtpOpsRasterRegion* regions, int regionCount,
                                                     int32_t bytes) {
  if (bytes != 2 || regionCount < 8 || g_max_num_workers <= 1) {
    return false;
  }
  const HtpOpsRasterRegion* r0 = regions + 0;
  if (r0->srcIndex < 0 || r0->srcIndex >= srcNumber || src[r0->srcIndex] == NULL ||
      r0->size[0] <= 0 || r0->size[1] <= 0 || r0->size[2] != 64 ||
      r0->srcStride[0] != 64 || r0->dstStride[0] != r0->size[1] * 64 ||
      r0->srcStride[1] < 64 || r0->dstStride[1] != 64 ||
      r0->srcStride[2] != 1 || r0->dstStride[2] != 1) {
    return false;
  }
  for (int i = 1; i < regionCount; ++i) {
    const HtpOpsRasterRegion* r = regions + i;
    if (r->srcIndex < 0 || r->srcIndex >= srcNumber || src[r->srcIndex] == NULL ||
        r->size[0] != r0->size[0] || r->size[1] != r0->size[1] || r->size[2] != 64 ||
        r->srcStride[0] != r0->srcStride[0] || r->dstStride[0] != r0->dstStride[0] ||
        r->srcStride[1] != r0->srcStride[1] || r->dstStride[1] != r0->dstStride[1] ||
        r->srcStride[2] != 1 || r->dstStride[2] != 1) {
      return false;
    }
  }

  HtpOpsRasterPackArea1GroupTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.regions = regions;
  state.regionCount = regionCount;
  state.zCount = r0->size[0];
  state.batch = r0->size[1];
  state.task_id = 0;
  const int total = regionCount * state.zCount;
  const size_t workBytes = (size_t)total * state.batch * 64 * sizeof(__fp16);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, total);
  state.grain = (total + nTasks - 1) / nTasks;
  if (nTasks <= 1) {
    htp_ops_pack_area1_group_range(&state, 0, total);
    return true;
  }
  worker_pool_job_t job;
  job.fptr = htp_ops_pack_area1_group_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int rowCount;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterPackToPackTaskState;

static inline void htp_ops_pack_to_pack_copy_range(HtpOpsRasterPackToPackTaskState* state,
                                                   int rowStart, int rowEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const int area = r->size[2];
  const size_t rowBytes = (size_t)r->size[1] * sizeof(uint16_t);
  const bool fullVector = rowBytes == (size_t)__HVX_LENGTH__;
  for (int index = rowStart; index < rowEnd; ++index) {
    const int z = index / area;
    const int x = index - z * area;
    const uint8_t* srcRow = state->srcBase + ((ptrdiff_t)z * r->srcStride[0] +
                                              (ptrdiff_t)x * r->srcStride[2]) * (ptrdiff_t)sizeof(uint16_t);
    uint8_t* dstRow = state->dstBase + ((ptrdiff_t)z * r->dstStride[0] +
                                        (ptrdiff_t)x * r->dstStride[2]) * (ptrdiff_t)sizeof(uint16_t);
    if (fullVector) {
      vmemu((HVX_Vector*)dstRow) = vmemu((const HVX_Vector*)srcRow);
    } else {
      htp_ops_copy_bytes_scalar(dstRow, srcRow, rowBytes);
    }
  }
}

static void htp_ops_pack_to_pack_copy_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterPackToPackTaskState* state = (HtpOpsRasterPackToPackTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    if (begin >= state->rowCount) {
      break;
    }
    int end = begin + state->grain;
    if (end > state->rowCount) {
      end = state->rowCount;
    }
    htp_ops_pack_to_pack_copy_range(state, begin, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_pack_to_pack_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                                 const HtpOpsRasterRegion* r, int32_t bytes) {
  if (bytes != 2 || r->size[0] <= 0 || r->size[1] <= 0 || r->size[1] > 64 || r->size[2] <= 0 ||
      r->srcStride[1] != 1 || r->dstStride[1] != 1 ||
      r->srcStride[2] != 64 || r->dstStride[2] != 64 ||
      (r->size[0] > 1 && (r->srcStride[0] < 0 || r->dstStride[0] < 0))) {
    return false;
  }
  const int rowCount = r->size[0] * r->size[2];
  const size_t workBytes = (size_t)rowCount * (size_t)r->size[1] * sizeof(uint16_t);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, rowCount);
  HtpOpsRasterPackToPackTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.rowCount = rowCount;
  state.grain = (rowCount + nTasks - 1) / nTasks;
  if (nTasks <= 1) {
    htp_ops_pack_to_pack_copy_range(&state, 0, rowCount);
    return true;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_pack_to_pack_copy_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  int size;
  ptrdiff_t srcStrideBytes;
  ptrdiff_t dstStrideBytes;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterStridedLineTaskState;

static inline void htp_ops_strided_line_copy_range(HtpOpsRasterStridedLineTaskState* state, int xStart, int xEnd) {
  const uint8_t* src = state->srcBase + (ptrdiff_t)xStart * state->srcStrideBytes;
  uint8_t* dst = state->dstBase + (ptrdiff_t)xStart * state->dstStrideBytes;
  for (int x = xStart; x < xEnd; ++x) {
    *(uint16_t*)dst = *(const uint16_t*)src;
    src += state->srcStrideBytes;
    dst += state->dstStrideBytes;
  }
}

static void htp_ops_strided_line_copy_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterStridedLineTaskState* state = (HtpOpsRasterStridedLineTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int xStart = taskId * state->grain;
    if (xStart >= state->size) {
      break;
    }
    int xEnd = xStart + state->grain;
    if (xEnd > state->size) {
      xEnd = state->size;
    }
    htp_ops_strided_line_copy_range(state, xStart, xEnd);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_strided_line_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                                 const HtpOpsRasterRegion* r, int32_t bytes) {
  if (bytes != 2 || r->size[0] != 1 || r->size[1] != 1 || r->size[2] < 4096 ||
      r->srcStride[0] != 0 || r->dstStride[0] != 0 ||
      r->srcStride[1] == 0 || r->dstStride[1] == 0 ||
      r->srcStride[2] <= 0 || r->dstStride[2] <= 0 ||
      g_max_num_workers <= 1) {
    return false;
  }
  const bool oneSideContiguous = (r->srcStride[2] == 1 || r->dstStride[2] == 1);
  if (!oneSideContiguous) {
    return false;
  }
  const int nTasks = htp_ops_raster_pick_task_count((size_t)r->size[2] * bytes, (int)g_max_num_workers);
  if (nTasks <= 1) {
    return false;
  }

  HtpOpsRasterStridedLineTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.size = r->size[2];
  state.srcStrideBytes = (ptrdiff_t)r->srcStride[2] * bytes;
  state.dstStrideBytes = (ptrdiff_t)r->dstStride[2] * bytes;
  state.grain = (state.size + nTasks - 1) / nTasks;
  state.task_id = 0;

  worker_pool_job_t job;
  job.fptr = htp_ops_strided_line_copy_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int bytes;
  int rowCount;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterRowCopyTaskState;

typedef struct {
  HtpOpsRasterRowCopyTaskState* state;
  int begin;
  int end;
} HtpOpsRasterRowCopyFixedTask;

static inline void htp_ops_row_copy_bytes(uint8_t* dst, const uint8_t* src, size_t rowBytes, bool allowOverread) {
  htp_ops_copy_bytes_hvx(dst, src, rowBytes, allowOverread);
}

static inline void htp_ops_row_copy_range(HtpOpsRasterRowCopyTaskState* state, int rowStart, int rowEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const ptrdiff_t unitBytes = state->bytes;
  const size_t rowBytes = (size_t)r->size[2] * state->bytes;
  const bool singleHvxRow = rowBytes >= 64 && rowBytes <= (size_t)__HVX_LENGTH__;
  int index = rowStart;
  while (index < rowEnd) {
    const int z = index / r->size[1];
    int y = index - z * r->size[1];
    int yEnd = y + (rowEnd - index);
    if (yEnd > r->size[1]) {
      yEnd = r->size[1];
    }
    const uint8_t* srcZ = state->srcBase + (ptrdiff_t)z * r->srcStride[0] * unitBytes;
    uint8_t* dstZ = state->dstBase + (ptrdiff_t)z * r->dstStride[0] * unitBytes;
    if (singleHvxRow) {
      const ptrdiff_t srcStrideBytes = (ptrdiff_t)r->srcStride[1] * unitBytes;
      const ptrdiff_t dstStrideBytes = (ptrdiff_t)r->dstStride[1] * unitBytes;
      const bool fullVector = rowBytes == (size_t)__HVX_LENGTH__;
      const uint32_t storeBytes = (uint32_t)rowBytes;
      for (; y + 3 < yEnd && (fullVector || index + 4 < state->rowCount); y += 4, index += 4) {
        const uint8_t* srcRow = srcZ + (ptrdiff_t)y * srcStrideBytes;
        uint8_t* dstRow = dstZ + (ptrdiff_t)y * dstStrideBytes;
        HVX_Vector v0 = vmemu((const HVX_Vector*)srcRow);
        HVX_Vector v1 = vmemu((const HVX_Vector*)(srcRow + srcStrideBytes));
        HVX_Vector v2 = vmemu((const HVX_Vector*)(srcRow + 2 * srcStrideBytes));
        HVX_Vector v3 = vmemu((const HVX_Vector*)(srcRow + 3 * srcStrideBytes));
        if (fullVector) {
          vmemu((HVX_Vector*)dstRow) = v0;
          vmemu((HVX_Vector*)(dstRow + dstStrideBytes)) = v1;
          vmemu((HVX_Vector*)(dstRow + 2 * dstStrideBytes)) = v2;
          vmemu((HVX_Vector*)(dstRow + 3 * dstStrideBytes)) = v3;
        } else {
          vstu_variable(dstRow, storeBytes, v0);
          vstu_variable(dstRow + dstStrideBytes, storeBytes, v1);
          vstu_variable(dstRow + 2 * dstStrideBytes, storeBytes, v2);
          vstu_variable(dstRow + 3 * dstStrideBytes, storeBytes, v3);
        }
      }
      for (; y < yEnd; ++y, ++index) {
        const uint8_t* srcRow = srcZ + (ptrdiff_t)y * srcStrideBytes;
        uint8_t* dstRow = dstZ + (ptrdiff_t)y * dstStrideBytes;
        if (fullVector || index + 1 < state->rowCount) {
          HVX_Vector v = vmemu((const HVX_Vector*)srcRow);
          if (fullVector) {
            vmemu((HVX_Vector*)dstRow) = v;
          } else {
            vstu_variable(dstRow, storeBytes, v);
          }
        } else {
          htp_ops_copy_bytes_scalar(dstRow, srcRow, rowBytes);
        }
      }
    } else {
      for (; y < yEnd; ++y, ++index) {
        const uint8_t* srcRow = srcZ + (ptrdiff_t)y * r->srcStride[1] * unitBytes;
        uint8_t* dstRow = dstZ + (ptrdiff_t)y * r->dstStride[1] * unitBytes;
        htp_ops_row_copy_bytes(dstRow, srcRow, rowBytes, index + 1 < state->rowCount);
      }
    }
  }
}

static void htp_ops_row_copy_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterRowCopyFixedTask* task = (HtpOpsRasterRowCopyFixedTask*)data;
  htp_ops_row_copy_range(task->state, task->begin, task->end);
  worker_pool_synctoken_jobdone(&(task->state->sync_ctx));
}

static inline bool htp_ops_try_row_copy_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                             const HtpOpsRasterRegion* r, int32_t bytes) {
  if ((bytes != 2 && bytes != 4 && bytes != 1 && bytes != __HVX_LENGTH__) ||
      r->size[0] <= 0 || r->size[1] <= 0 || r->size[2] <= 0 ||
      r->srcStride[2] != 1 || r->dstStride[2] != 1 ||
      r->srcStride[0] < 0 || r->srcStride[1] < 0 ||
      r->dstStride[0] < 0 || r->dstStride[1] < 0 ||
      (r->srcStride[1] == r->size[2] && r->dstStride[1] == r->size[2]) ||
      g_max_num_workers <= 1) {
    return false;
  }
  const int rowCount = r->size[0] * r->size[1];
  const size_t workBytes = (size_t)rowCount * r->size[2] * bytes;
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, rowCount);
  if (nTasks <= 1) {
    return false;
  }

  HtpOpsRasterRowCopyTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.bytes = bytes;
  state.rowCount = rowCount;
  state.grain = (rowCount + nTasks - 1) / nTasks;
  state.task_id = 0;

  worker_pool_job_t job;
  job.fptr = htp_ops_row_copy_worker;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  HtpOpsRasterRowCopyFixedTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsRasterRowCopyFixedTask, nTasks);
  for (int i = 0; i < nTasks; ++i) {
    const int begin = i * state.grain;
    int end = begin + state.grain;
    if (end > rowCount) {
      end = rowCount;
    }
    tasks[i].state = &state;
    tasks[i].begin = begin;
    tasks[i].end = end;
    job.dptr = tasks + i;
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int bytes;
  size_t planeBytes;
  int zCount;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterContiguousTaskState;

static inline void htp_ops_contiguous_blit_range(HtpOpsRasterContiguousTaskState* state, int zStart, int zEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const ptrdiff_t unitBytes = state->bytes;
  for (int z = zStart; z < zEnd; ++z) {
    const ptrdiff_t srcZOffset = (ptrdiff_t)z * r->srcStride[0] * unitBytes;
    const ptrdiff_t dstZOffset = (ptrdiff_t)z * r->dstStride[0] * unitBytes;
    htp_ops_copy_bytes_hvx(state->dstBase + dstZOffset, state->srcBase + srcZOffset, state->planeBytes, false);
  }
}

static void htp_ops_contiguous_blit_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterContiguousTaskState* state = (HtpOpsRasterContiguousTaskState*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int zStart = taskId * state->grain;
    if (zStart >= state->zCount) {
      break;
    }
    int zEnd = zStart + state->grain;
    if (zEnd > state->zCount) {
      zEnd = state->zCount;
    }
    htp_ops_contiguous_blit_range(state, zStart, zEnd);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline void htp_ops_run_contiguous_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                               const HtpOpsRasterRegion* r, int32_t bytes) {
  HtpOpsRasterContiguousTaskState state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.bytes = bytes;
  state.planeBytes = (size_t)r->size[1] * r->srcStride[1] * bytes;
  state.zCount = r->size[0];
  state.task_id = 0;
  const size_t workBytes = (size_t)state.zCount * state.planeBytes;
  if (state.zCount == 1 && htp_ops_try_parallel_linear_copy(dstBase, srcBase, workBytes)) {
    return;
  }
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, state.zCount);
  state.grain = (state.zCount + nTasks - 1) / nTasks;
  if (nTasks <= 1 || r->dstStride[0] == 0) {
    htp_ops_contiguous_blit_range(&state, 0, state.zCount);
    return;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_contiguous_blit_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

typedef struct {
  uint8_t* dst;
  uint8_t** src;
  int srcNumber;
  const HtpOpsRasterRegion* regions;
  int regionCount;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterFlattenCWTaskState;

static inline void htp_ops_flatten_cw_region(uint8_t* dst, uint8_t** src, const HtpOpsRasterRegion* r) {
  const uint16_t* srcBase = (const uint16_t*)src[r->srcIndex] + r->srcOffset;
  uint16_t* dstBase = (uint16_t*)dst + r->dstOffset;
  for (int y = 0; y < r->size[1]; ++y) {
    const uint16_t* srcY = srcBase + (ptrdiff_t)y * r->srcStride[1];
    uint16_t* dstY = dstBase + (ptrdiff_t)y * 64;
    int x = 0;
    for (; x + 3 < r->size[2]; x += 4) {
      dstY[x + 0] = srcY[(x + 0) * 64];
      dstY[x + 1] = srcY[(x + 1) * 64];
      dstY[x + 2] = srcY[(x + 2) * 64];
      dstY[x + 3] = srcY[(x + 3) * 64];
    }
    for (; x < r->size[2]; ++x) {
      dstY[x] = srcY[x * 64];
    }
  }
}

static void htp_ops_flatten_cw_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterFlattenCWTaskState* state = (HtpOpsRasterFlattenCWTaskState*)data;
  while (true) {
    const int index = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (index >= state->regionCount) {
      break;
    }
    htp_ops_flatten_cw_region(state->dst, state->src, state->regions + index);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_flatten_cw_blit(uint8_t* dst, uint8_t** src, int srcNumber,
                                               const HtpOpsRasterRegion* regions, int regionCount, int32_t bytes) {
  if (bytes != 2 || regionCount < 4 || g_max_num_workers <= 1) {
    return false;
  }
  size_t workBytes = 0;
  for (int i = 0; i < regionCount; ++i) {
    const HtpOpsRasterRegion* r = regions + i;
    if (r->srcIndex < 0 || r->srcIndex >= srcNumber || src[r->srcIndex] == NULL ||
        r->size[0] != 1 || r->size[1] <= 0 || r->size[2] <= 0 ||
        r->srcStride[0] != 0 || r->dstStride[0] != 0 ||
        r->srcStride[2] != 64 || r->dstStride[2] != 1 ||
        r->dstStride[1] != 64) {
      return false;
    }
    workBytes += (size_t)r->size[1] * r->size[2] * sizeof(__fp16);
  }
  if (workBytes < 8192) {
    return false;
  }

  int nTasks = (int)g_max_num_workers;
  if (nTasks > regionCount) {
    nTasks = regionCount;
  }
  HtpOpsRasterFlattenCWTaskState state = {};
  state.dst = dst;
  state.src = src;
  state.srcNumber = srcNumber;
  state.regions = regions;
  state.regionCount = regionCount;
  state.task_id = 0;

  worker_pool_job_t job;
  job.fptr = htp_ops_flatten_cw_worker;
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
  uint8_t** src;
  const HtpOpsRasterRegion* regions;
  int rows;
  int ySize;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterDeinterleaveC64State;

static inline void htp_ops_deinterleave_c64_range(HtpOpsRasterDeinterleaveC64State* state,
                                                  int rowStart, int rowEnd) {
  const HtpOpsRasterRegion* r0 = state->regions + 0;
  const HtpOpsRasterRegion* r1 = state->regions + 1;
  const HtpOpsRasterRegion* r2 = state->regions + 2;
  const HtpOpsRasterRegion* r3 = state->regions + 3;
  const uint16_t* srcBase = (const uint16_t*)state->src[r0->srcIndex];
  uint16_t* dstBase = (uint16_t*)state->dst;
  for (int row = rowStart; row < rowEnd; ++row) {
    const int z = row / state->ySize;
    const int y = row - z * state->ySize;
    const uint16_t* s0 = srcBase + r0->srcOffset + (ptrdiff_t)z * r0->srcStride[0] + (ptrdiff_t)y * r0->srcStride[1];
    const uint16_t* s1 = srcBase + r1->srcOffset + (ptrdiff_t)z * r1->srcStride[0] + (ptrdiff_t)y * r1->srcStride[1];
    const uint16_t* s2 = srcBase + r2->srcOffset + (ptrdiff_t)z * r2->srcStride[0] + (ptrdiff_t)y * r2->srcStride[1];
    const uint16_t* s3 = srcBase + r3->srcOffset + (ptrdiff_t)z * r3->srcStride[0] + (ptrdiff_t)y * r3->srcStride[1];
    uint64_t* dst64 = (uint64_t*)(dstBase + r0->dstOffset + (ptrdiff_t)z * r0->dstStride[0] + (ptrdiff_t)y * r0->dstStride[1]);
    if (s1 == s0 + 16 && s2 == s0 + 32 && s3 == s0 + 48) {
      HVX_Vector v = vmemu((const HVX_Vector*)s0);
      HVX_Vector v1 = Q6_V_vror_VR(v, 32);
      HVX_Vector v2 = Q6_V_vror_VR(v, 64);
      HVX_Vector v3 = Q6_V_vror_VR(v, 96);
      HVX_VectorPair ab = Q6_W_vshuff_VVR(v1, v, -2);
      HVX_VectorPair cd = Q6_W_vshuff_VVR(v3, v2, -2);
      HVX_VectorPair out = Q6_W_vshuff_VVR(Q6_V_lo_W(cd), Q6_V_lo_W(ab), -4);
      vmemu((HVX_Vector*)dst64) = Q6_V_lo_W(out);
      continue;
    }
    for (int x = 0; x < 16; x += 4) {
      const uint64_t v0 = *(const uint64_t*)(s0 + x);
      const uint64_t v1 = *(const uint64_t*)(s1 + x);
      const uint64_t v2 = *(const uint64_t*)(s2 + x);
      const uint64_t v3 = *(const uint64_t*)(s3 + x);
      dst64[x + 0] = (v0 & 0xffffULL) | ((v1 & 0xffffULL) << 16) |
                     ((v2 & 0xffffULL) << 32) | ((v3 & 0xffffULL) << 48);
      dst64[x + 1] = ((v0 >> 16) & 0xffffULL) | (v1 & 0xffff0000ULL) |
                     ((v2 & 0xffff0000ULL) << 16) | ((v3 & 0xffff0000ULL) << 32);
      dst64[x + 2] = ((v0 >> 32) & 0xffffULL) | ((v1 >> 16) & 0xffff0000ULL) |
                     (v2 & 0xffff00000000ULL) | ((v3 & 0xffff00000000ULL) << 16);
      dst64[x + 3] = ((v0 >> 48) & 0xffffULL) | ((v1 >> 32) & 0xffff0000ULL) |
                     ((v2 >> 16) & 0xffff00000000ULL) | (v3 & 0xffff000000000000ULL);
    }
  }
}

static void htp_ops_deinterleave_c64_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterDeinterleaveC64State* state = (HtpOpsRasterDeinterleaveC64State*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    if (begin >= state->rows) {
      break;
    }
    int end = begin + state->grain;
    if (end > state->rows) {
      end = state->rows;
    }
    htp_ops_deinterleave_c64_range(state, begin, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_deinterleave_c64_blit(uint8_t* dst, uint8_t** src, int srcNumber,
                                                     const HtpOpsRasterRegion* regions, int regionCount,
                                                     int32_t bytes) {
  if (bytes != 2 || regionCount != 4 || srcNumber <= 0 || g_max_num_workers <= 1) {
    return false;
  }
  const HtpOpsRasterRegion* r0 = regions + 0;
  if (r0->srcIndex < 0 || r0->srcIndex >= srcNumber || src[r0->srcIndex] == NULL ||
      r0->size[0] <= 0 || r0->size[1] <= 0 || r0->size[2] != 16 ||
      r0->srcStride[2] != 1 || r0->dstStride[2] != 4 ||
      r0->dstStride[1] != 64 || r0->dstStride[0] != r0->size[1] * 64 ||
      r0->srcStride[0] != 16 ||
      (((uintptr_t)src[r0->srcIndex] + (uintptr_t)r0->srcOffset * sizeof(uint16_t)) & 7) != 0 ||
      (((uintptr_t)dst + (uintptr_t)r0->dstOffset * sizeof(uint16_t)) & 7) != 0) {
    return false;
  }
  for (int i = 1; i < 4; ++i) {
    const HtpOpsRasterRegion* r = regions + i;
    if (r->srcIndex != r0->srcIndex ||
        r->size[0] != r0->size[0] || r->size[1] != r0->size[1] || r->size[2] != 16 ||
        r->srcStride[0] != r0->srcStride[0] || r->srcStride[1] != r0->srcStride[1] ||
        r->srcStride[2] != 1 ||
        r->dstStride[0] != r0->dstStride[0] || r->dstStride[1] != r0->dstStride[1] ||
        r->dstStride[2] != 4 ||
        r->srcOffset != r0->srcOffset + i * 16 ||
        r->dstOffset != r0->dstOffset + i) {
      return false;
    }
  }

  HtpOpsRasterDeinterleaveC64State state = {};
  state.dst = dst;
  state.src = src;
  state.regions = regions;
  state.rows = r0->size[0] * r0->size[1];
  state.ySize = r0->size[1];
  state.task_id = 0;
  const size_t workBytes = (size_t)state.rows * 64 * sizeof(uint16_t);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, state.rows);
  state.grain = (state.rows + nTasks - 1) / nTasks;
  if (nTasks <= 1) {
    htp_ops_deinterleave_c64_range(&state, 0, state.rows);
    return true;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_deinterleave_c64_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

typedef struct {
  uint8_t* dstBase;
  const uint8_t* srcBase;
  const HtpOpsRasterRegion* region;
  int rows;
  int ySize;
  int grain;
  unsigned int task_id;
  worker_synctoken_t sync_ctx;
} HtpOpsRasterBroadcastInnerFp16State;

static inline void htp_ops_fill_fp16(uint16_t* dst, int count, uint16_t value) {
  HVX_Vector v = Q6_Vh_vsplat_R(value);
  int x = 0;
  for (; x + 64 <= count; x += 64) {
    vmemu((HVX_Vector*)(dst + x)) = v;
  }
  if (x < count) {
    vstu_variable(dst + x, (uint32_t)((count - x) * (int)sizeof(uint16_t)), v);
  }
}

static inline void htp_ops_broadcast_inner_fp16_range(HtpOpsRasterBroadcastInnerFp16State* state,
                                                      int rowStart, int rowEnd) {
  const HtpOpsRasterRegion* r = state->region;
  const uint16_t* srcBase = (const uint16_t*)state->srcBase;
  uint16_t* dstBase = (uint16_t*)state->dstBase;
  for (int row = rowStart; row < rowEnd; ++row) {
    const int z = row / state->ySize;
    const int y = row - z * state->ySize;
    const uint16_t value = srcBase[(ptrdiff_t)z * r->srcStride[0] + (ptrdiff_t)y * r->srcStride[1]];
    uint16_t* dst = dstBase + (ptrdiff_t)z * r->dstStride[0] + (ptrdiff_t)y * r->dstStride[1];
    htp_ops_fill_fp16(dst, r->size[2], value);
  }
}

static void htp_ops_broadcast_inner_fp16_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsRasterBroadcastInnerFp16State* state = (HtpOpsRasterBroadcastInnerFp16State*)data;
  while (true) {
    const int taskId = (int)worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    const int begin = taskId * state->grain;
    if (begin >= state->rows) {
      break;
    }
    int end = begin + state->grain;
    if (end > state->rows) {
      end = state->rows;
    }
    htp_ops_broadcast_inner_fp16_range(state, begin, end);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static inline bool htp_ops_try_broadcast_inner_fp16_blit(uint8_t* dstBase, const uint8_t* srcBase,
                                                         const HtpOpsRasterRegion* r, int32_t bytes) {
  if (bytes != 2 || r->size[0] <= 0 || r->size[1] <= 0 || r->size[2] < 64 ||
      r->srcStride[2] != 0 || r->dstStride[2] != 1 ||
      r->dstStride[0] < 0 || r->dstStride[1] < 0 ||
      r->srcStride[0] < 0 || r->srcStride[1] < 0) {
    return false;
  }
  HtpOpsRasterBroadcastInnerFp16State state = {};
  state.dstBase = dstBase;
  state.srcBase = srcBase;
  state.region = r;
  state.rows = r->size[0] * r->size[1];
  state.ySize = r->size[1];
  state.task_id = 0;
  const size_t workBytes = (size_t)state.rows * r->size[2] * sizeof(uint16_t);
  const int nTasks = htp_ops_raster_pick_task_count(workBytes, state.rows);
  state.grain = (state.rows + nTasks - 1) / nTasks;
  if (nTasks <= 1) {
    htp_ops_broadcast_inner_fp16_range(&state, 0, state.rows);
    return true;
  }

  worker_pool_job_t job;
  job.fptr = htp_ops_broadcast_inner_fp16_worker;
  job.dptr = &state;
  worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
  for (int i = 0; i < nTasks; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return true;
}

AEEResult htp_ops_zero(uint8_t* dst, int32_t size) {
  if (dst == NULL || size < 0) {
    return -1;
  }
  if (size == 0) {
    return 0;
  }
  memset(dst, 0, (size_t)size);
  return 0;
}

AEEResult htp_ops_raster_blit(uint8_t* dst, uint8_t** src, int src_number, uint8_t* region, int32_t regionCount, int32_t bytes) {
  if (regionCount <= 0) {
    return -1;
  }
  if (bytes <= 0) {
    return -1;
  }
  if (src_number <= 0 || src_number > 10) {
    return -1;
  }
  uint8_t *pDst = dst;
  uint8_t *pReg = region;
  const ptrdiff_t unitBytes = bytes;
  HtpOpsRasterBlitProc blitProc = htp_ops_select_blit_proc(bytes);

  const HtpOpsRasterRegion *regions = (const HtpOpsRasterRegion *) pReg;

  if (htp_ops_try_deinterleave_c64_blit(pDst, src, src_number, regions, regionCount, bytes)) {
    return 0;
  }
  if (htp_ops_try_flatten_cw_blit(pDst, src, src_number, regions, regionCount, bytes)) {
    return 0;
  }
  if (htp_ops_try_pack_area1_group_blit(pDst, src, src_number, regions, regionCount, bytes)) {
    return 0;
  }

  for (int i = 0; i < regionCount; ++i) {
    const HtpOpsRasterRegion *r = regions + i;

    if (r->srcIndex < 0 || r->srcIndex >= src_number) {
      FARF(ALWAYS, "Invalid srcIndex: %d, src_number: %d", r->srcIndex, src_number);
      continue;
    }

    uint8_t *pSrc = src[r->srcIndex];
    if (pSrc == NULL) {
      FARF(ALWAYS, "src[%d] is NULL", r->srcIndex);
      continue;
    }

    const uint8_t *srcBase = pSrc + (ptrdiff_t)r->srcOffset * unitBytes;
    uint8_t *dstBase = pDst + (ptrdiff_t)r->dstOffset * unitBytes;
    if (srcBase == dstBase &&
        r->srcStride[0] == r->dstStride[0] &&
        r->srcStride[1] == r->dstStride[1] &&
        r->srcStride[2] == r->dstStride[2]) {
      continue;
    }
    if (htp_ops_try_broadcast_inner_fp16_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (htp_ops_try_pack_area1_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (htp_ops_try_pack_to_pack_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (htp_ops_try_pack_area_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (htp_ops_try_strided_line_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (htp_ops_try_row_copy_blit(dstBase, srcBase, r, bytes)) {
      continue;
    }
    if (r->srcStride[1] == r->size[2] && r->dstStride[1] == r->size[2] && r->srcStride[2] == 1) {
      htp_ops_run_contiguous_blit(dstBase, srcBase, r, bytes);
      continue;
    }
    if (bytes == 2) {
      int dims[4];
      int keepDim = -1;
      if (htp_ops_prepare_transpose(r, dims, &keepDim)) {
        htp_ops_transpose_16_hvx(dstBase, srcBase, r);
        continue;
      }
    }
    for (int z = 0; z < r->size[0]; ++z) {
      const ptrdiff_t srcZOffset = (ptrdiff_t)z * r->srcStride[0] * unitBytes;
      const ptrdiff_t dstZOffset = (ptrdiff_t)z * r->dstStride[0] * unitBytes;
      const uint8_t *srcZ = srcBase + srcZOffset;
      uint8_t *dstZ = dstBase + dstZOffset;
      for (int y = 0; y < r->size[1]; ++y) {
        const ptrdiff_t srcYOffset = (ptrdiff_t)y * r->srcStride[1] * unitBytes;
        const ptrdiff_t dstYOffset = (ptrdiff_t)y * r->dstStride[1] * unitBytes;
        const uint8_t *srcY = srcZ + srcYOffset;
        uint8_t *dstY = dstZ + dstYOffset;

        if (r->srcStride[2] == 1 && r->dstStride[2] == 1) {
          const size_t rowBytes = (size_t) r->size[2] * bytes;
          htp_ops_copy_bytes_hvx(dstY, srcY, rowBytes, false);
        } else {
          blitProc(dstY, srcY, r->size[2], (ptrdiff_t)r->srcStride[2] * unitBytes,
                   (ptrdiff_t)r->dstStride[2] * unitBytes, bytes);
        }
      }
    }
  }

  return 0;
}

AEEResult htp_ops_weight_reorder(uint8_t* dst_ptr, uint8_t* src_ptr, const WeightReorderParam* params) {
    const int32_t ic = params->ic;
    const int32_t oc = params->oc;
    const int32_t kernelX = params->kernelX;
    const int32_t kernelY = params->kernelY;
    int16_t* weightHalf = (int16_t*)src_ptr;
    int16_t* weightPtr = (int16_t*)dst_ptr;

    int icPack = 32;
    int ocPack = 32;
    int icP = (ic + icPack - 1) / icPack;
    int ocP = (oc + ocPack - 1) / ocPack;
    int kp = kernelY * kernelX * icP;
    int packs = icPack * ocPack;
    const size_t reorderedSize = (size_t)ocP * icP * kernelY * kernelX * packs;
    memset(weightPtr, 0, reorderedSize * sizeof(int16_t));
    for (int oz = 0; oz < ocP; ++oz) {
        for (int kk = 0; kk < kp; ++kk) {
            const int kernelIndex = kk / icP;
            const int iz = kk % icP;
            const int ky = kernelIndex / kernelX;
            const int kx = kernelIndex % kernelX;
            const size_t blockBase = ((size_t)oz * kp + kk) * packs;
            for (int oy = 0; oy < ocPack; ++oy) {
                const int o = oz * ocPack + oy;
                if (o >= oc) {
                    continue;
                }
                for (int ix = 0; ix < icPack; ++ix) {
                    const int i = iz * icPack + ix;
                    if (i >= ic) {
                        continue;
                    }
                    const size_t srcIndex = (((size_t)o * ic + i) * kernelY + ky) * kernelX + kx;
                    const int ix_pair = ix / 2;
                    const int ix_rem = ix & 1;
                    const size_t dstIndex = blockBase + (size_t)ix_pair * 64 + oy * 2 + ix_rem;
                    weightPtr[dstIndex] = weightHalf[srcIndex];
                }
            }
        }
    }

    return AEE_SUCCESS;
}

}  // extern "C"
