// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "dsp/hvx_utils.h"
#include "dsp/hmx_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include <remote.h>
#include <HAP_compute_res.h>
#include "qurt.h"
#include "htp_command.h"
static int htp_ops_hmx_matmul_fp16_benchmark_core(__fp16 *restrict c, const __fp16 *restrict a,
                                                  const __fp16 *restrict b, __fp16 *restrict scales,
                                                  int M, int K, int N) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const __fp16 *a_tiles = a + (size_t)i * kt * HMX_FP16_TILE_N_ELMS;
      const __fp16 *b_tiles = b + (size_t)j * kt * HMX_FP16_TILE_N_ELMS;
      __fp16       *c_tile  = c + ((size_t)i * nt + j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        int tiles = kt - k;
        if (tiles > 32) {
          tiles = 32;
        }
        hmx_load_tiles_fp16(a_tiles + (size_t)k * HMX_FP16_TILE_N_ELMS,
                            b_tiles + (size_t)k * HMX_FP16_TILE_N_ELMS,
                            tiles);
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}

static void htp_ops_fill_matmul_flops(int32_t *dst) {
  float *flops = (float *)(dst + 6);
  uint8_t *vtcm = (uint8_t *)vtcm_manager_get_vtcm_base();
  if (vtcm == nullptr) {
    return;
  }

  __fp16 *a = (__fp16 *)vtcm;
  __fp16 *b = (__fp16 *)(vtcm + 2 * 1024 * 1024);
  __fp16 *c = (__fp16 *)(vtcm + 4 * 1024 * 1024);
  __fp16 *s = (__fp16 *)(vtcm + 6 * 1024 * 1024);

  const int n_repeat = 1000;
  const int sizes[6] = {32, 64, 128, 256, 512, 1024};

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int i = 0; i < 6; ++i) {
    const int64_t n = sizes[i];
    const int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      htp_ops_hmx_matmul_fp16_benchmark_core(c, a, b, s, (int)n, (int)n, (int)n);
    }
    const int64_t t1 = HAP_perf_get_qtimer_count();
    const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);
    if (elapsed_us > 0) {
      flops[i] = 1e-3f * n_repeat * (2.0f * n * n * n) / elapsed_us;
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
}

extern "C" {

struct HtpOpsSessionContext {
  bool initialized;
  dspqueue_t queue;
};

enum BackendLifecycleState {
  BACKEND_UNINITIALIZED = 0,
  BACKEND_INITIALIZING = -1,
  BACKEND_DEINITIALIZING = -2,
};

static volatile int g_backend_ref_count = BACKEND_UNINITIALIZED;

static void htp_ops_global_backend_setup() {
  power_setup();
  vtcm_manager_setup();
  hmx_manager_setup();
  worker_pool_global_init();
}

static void htp_ops_global_backend_reset() {
  worker_pool_global_deinit();
  mmap_manager_release_all();

  hmx_manager_reset();
  vtcm_manager_reset();
  power_reset();
}

static bool htp_ops_acquire_global_backend() {
  while (true) {
    int state = __atomic_load_n(&g_backend_ref_count, __ATOMIC_ACQUIRE);
    if (state > 0) {
      if (__sync_bool_compare_and_swap(&g_backend_ref_count, state, state + 1)) {
        return true;
      }
      continue;
    }
    if (state == BACKEND_UNINITIALIZED) {
      if (__sync_bool_compare_and_swap(&g_backend_ref_count, BACKEND_UNINITIALIZED, BACKEND_INITIALIZING)) {
        htp_ops_global_backend_setup();
        __atomic_store_n(&g_backend_ref_count, 1, __ATOMIC_RELEASE);
        return true;
      }
      continue;
    }
    qurt_sleep(100);
  }
}

static void htp_ops_release_global_backend() {
  while (true) {
    int state = __atomic_load_n(&g_backend_ref_count, __ATOMIC_ACQUIRE);
    if (state <= 0) {
      return;
    }
    if (state > 1) {
      if (__sync_bool_compare_and_swap(&g_backend_ref_count, state, state - 1)) {
        return;
      }
      continue;
    }
    if (__sync_bool_compare_and_swap(&g_backend_ref_count, 1, BACKEND_DEINITIALIZING)) {
      htp_ops_global_backend_reset();
      __atomic_store_n(&g_backend_ref_count, BACKEND_UNINITIALIZED, __ATOMIC_RELEASE);
      return;
    }
  }
}

static void htp_ops_queue_packet_callback(dspqueue_t queue, int error, void *context);
static void htp_ops_queue_error_callback(dspqueue_t queue, int error, void *context);

AEEResult htp_ops_execute_command_group(remote_handle64 handle, int32 groupFd, int32 groupOffset, int32 count,
                                        int32 syncGroupFd, int32 syncGroupOffset, int32 syncGroupSize);
AEEResult htp_ops_execute_command_group_profile(remote_handle64 handle, int32 groupFd, int32 groupOffset, int32 count,
                                                int32 syncGroupFd, int32 syncGroupOffset, int32 syncGroupSize,
                                                int32 profileFd, int32 profileOffset, int32 profileSize);

// FastRPC interface
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle) {
  if (handle == nullptr) {
    return AEE_EBADPARM;
  }

  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)calloc(1, sizeof(HtpOpsSessionContext));
  if (ctx == nullptr) {
    return AEE_ENOMEMORY;
  }

  *handle = (remote_handle64)(uintptr_t)ctx;
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_close(remote_handle64 handle) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)handle;
  if (ctx == nullptr) {
    return AEE_EBADPARM;
  }

  if (ctx->queue) {
    dspqueue_close(ctx->queue);
    ctx->queue = NULL;
  }

  if (ctx->initialized) {
    htp_ops_release_global_backend();
    ctx->initialized = false;
  }

  free(ctx);

  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_start_queue(remote_handle64 handle, uint64 queueId) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)handle;
  if (ctx == nullptr || !ctx->initialized) {
    return AEE_EBADSTATE;
  }
  if (ctx->queue) {
    return AEE_EITEMBUSY;
  }

  int err = dspqueue_import(queueId,
                            htp_ops_queue_packet_callback,
                            htp_ops_queue_error_callback,
                            (void*)ctx,
                            &ctx->queue);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "htp_ops_start_queue: dspqueue_import failed: 0x%08x", (unsigned)err);
    ctx->queue = NULL;
    return err;
  }
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_stop_queue(remote_handle64 handle) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)handle;
  if (ctx == nullptr) {
    return AEE_EBADPARM;
  }
  if (!ctx->queue) {
    return AEE_SUCCESS;
  }

  int err = dspqueue_close(ctx->queue);
  ctx->queue = NULL;
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "htp_ops_stop_queue: dspqueue_close failed: 0x%08x", (unsigned)err);
    return err;
  }
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_init_backend(remote_handle64 handle) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)handle;
  if (ctx == nullptr) {
    return AEE_EBADPARM;
  }
  if (ctx->initialized) {
    return AEE_SUCCESS;
  }

  FARF(ALWAYS, "init_backend called");

  if (!htp_ops_acquire_global_backend()) {
    return AEE_EFAILED;
  }

  ctx->initialized = true;

  return AEE_SUCCESS;
}

#define MNN_DSPQUEUE_POLL_TIMEOUT_USEC 100
#define MNN_DSPQUEUE_POLL_COUNT 100

static void htp_ops_queue_packet_callback(dspqueue_t queue, int error, void *context) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)context;
  if (ctx == nullptr) {
    return;
  }

  uint32_t poll_count = MNN_DSPQUEUE_POLL_COUNT;
  while (true) {
    struct DSPQueueCommandGroupReq req;
    memset(&req, 0, sizeof(req));
    uint32_t req_size = sizeof(req);
    uint32_t n_dbufs = 0;
    uint32_t flags = 0;

    int err = dspqueue_read_noblock(queue,
                                    &flags,
                                    0,
                                    &n_dbufs,
                                    NULL,
                                    sizeof(req),
                                    &req_size,
                                    (uint8_t *)&req);
    if (err == AEE_EWOULDBLOCK) {
      if (vtcm_manager_needs_release()) {
        break;
      }
      if (--poll_count) {
        qurt_sleep(MNN_DSPQUEUE_POLL_TIMEOUT_USEC);
        continue;
      }
      break;
    }
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "htp_ops_queue_packet_callback: dspqueue_read_noblock failed: 0x%08x", (unsigned)err);
      break;
    }
    if (req_size != sizeof(req) || n_dbufs != 0) {
      FARF(ERROR, "htp_ops_queue_packet_callback: invalid request size=%u n_dbufs=%u",
           (unsigned)req_size, (unsigned)n_dbufs);
      continue;
    }

    poll_count = MNN_DSPQUEUE_POLL_COUNT;
    int status = AEE_SUCCESS;
    if (req.count > 0 && !vtcm_manager_is_acquired()) {
      status = vtcm_manager_acquire();
      if (status != AEE_SUCCESS) {
        FARF(ERROR, "htp_ops_queue_packet_callback: failed to acquire VTCM/HMX resource: 0x%08x",
             (unsigned)status);
      }
    }
    if (status == AEE_SUCCESS) {
      status = req.profile ?
        htp_ops_execute_command_group_profile((remote_handle64)(uintptr_t)ctx,
                                              req.groupFd, req.groupOffset, req.count,
                                              req.syncGroupFd, req.syncGroupOffset, req.syncGroupSize,
                                              req.profileFd, req.profileOffset, req.profileSize) :
        htp_ops_execute_command_group((remote_handle64)(uintptr_t)ctx,
                                      req.groupFd, req.groupOffset, req.count,
                                      req.syncGroupFd, req.syncGroupOffset, req.syncGroupSize);
    }

    struct DSPQueueCommandGroupRsp rsp;
    rsp.id = req.id;
    rsp.status = status;
    err = dspqueue_write(queue,
                         0,
                         0,
                         NULL,
                         sizeof(rsp),
                         (const uint8_t *)&rsp,
                         DSPQUEUE_TIMEOUT_NONE);
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "htp_ops_queue_packet_callback: dspqueue_write failed: 0x%08x", (unsigned)err);
      break;
    }
    if (vtcm_manager_needs_release()) {
      break;
    }
  }

  vtcm_manager_release();
}

static void htp_ops_queue_error_callback(dspqueue_t queue, int error, void *context) {
  FARF(ERROR, "htp_ops_queue_error_callback: 0x%08x", (unsigned)error);
}

// FastRPC interface
AEEResult htp_ops_test_ops(remote_handle64 handle) {
  FARF(ALWAYS, "Op Tests!");

  return 0;
}

static AEEResult htp_ops_getInfo_impl_internal(uint8_t *p0, bool benchmark_flops) {
  int32_t* dst;
  if (!p0) {
    FARF(ALWAYS, "htp_ops_getInfo: invalid pointer");
    return -1;
  }
  dst = (int32_t*)p0;
  // Use fp16 lanes as vectorSize for host-side packing
  *dst = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
  dst[1] = TEST_M_PACK;
  dst[2] = TEST_K_PACK;
  dst[3] = TEST_N_PACK;

  unsigned int total_vtcm_size = 0, avail_vtcm_size = 0;
  compute_res_vtcm_page_t total_pages, avail_pages;
  int vtcm_err = HAP_compute_res_query_VTCM(0, &total_vtcm_size, &total_pages, &avail_vtcm_size, &avail_pages);
  if (vtcm_err == 0) {
      dst[4] = (int32_t)total_vtcm_size;
  } else {
      dst[4] = 0;
  }
  dst[5] = (int32_t)(qurt_hvx_get_units() >> 8);
  float *flops = (float *)(dst + 6);
  for (int i = 0; i < 6; ++i) {
    flops[i] = 0.0f;
  }
  if (benchmark_flops) {
    if (vtcm_manager_acquire() == 0) {
      htp_ops_fill_matmul_flops(dst);
      vtcm_manager_release();
    }
  }
#ifdef __HVX_ARCH__
  dst[12] = __HVX_ARCH__;
#else
  dst[12] = 0;
#endif

  qurt_mem_cache_clean((qurt_addr_t) dst, 256, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  return 0;
}

AEEResult htp_ops_getInfo_impl(uint8_t *p0) {
  return htp_ops_getInfo_impl_internal(p0, false);
}

AEEResult htp_ops_getInfo(remote_handle64 _h, int32 fd0, int32 offset0) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)_h;
  if (ctx == nullptr || !ctx->initialized) {
    return AEE_EBADSTATE;
  }
  uint8_t* p0 = NULL;
  if ((p0 = (uint8_t*)mmap_manager_get_map(fd0)) == NULL) return -1;
  AEEResult res = htp_ops_getInfo_impl_internal(p0 + offset0, false);
  HAP_mmap_put(fd0);
  return res;
}

AEEResult htp_ops_getInfoProfile(remote_handle64 _h, int32 fd0, int32 offset0) {
  HtpOpsSessionContext *ctx = (HtpOpsSessionContext *)(uintptr_t)_h;
  if (ctx == nullptr || !ctx->initialized) {
    return AEE_EBADSTATE;
  }
  uint8_t* p0 = NULL;
  if ((p0 = (uint8_t*)mmap_manager_get_map(fd0)) == NULL) return -1;
  AEEResult res = htp_ops_getInfo_impl_internal(p0 + offset0, true);
  HAP_mmap_put(fd0);
  return res;
}

}  // extern "C"
