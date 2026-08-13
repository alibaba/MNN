#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
#include "dsp/utils.h"
#include "dsp/vrmpy_to_hmx.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "HAP_farf.h"
#include "HAP_perf.h"

// Phase profiling (diagnostic): run-total microsecond accumulators for the HMX
// prefill matmul, surfaced by execute_command.cc into profile[] spare slots
// 200..204 (host prints as "DSPOpType (200..204)"). Cumulative over the whole run.
// [0]=HMX compute, [1]=wait weight dequant, [2]=wait DMA (weight+act), [3]=output
// store, [4]=activation shuffle(non-reuse), [5]=total wall (accum), [6]=last np_chunk,
// [7]=last oy-chunk count, [8]=setup(entry->loop), [9]=activation deshuffle(reuse
// path, copy_shuffle), [10]=store worker time (summed), [11]=dequant work (summed),
// [12]=worker DMA-poll wait (summed). Slots 200..212.
// Set HTP_MM_PHASE_PROFILE 1 to re-enable (diagnostic; off by default).
#ifndef HTP_MM_PHASE_PROFILE
#  define HTP_MM_PHASE_PROFILE 0
#endif

// Output store: hand each oy-chunk store to a worker (1) or run it on the dispatch
// thread (0), and how deep the per-oy-parity buffer ring is when async.
#ifndef MM_STORE_ASYNC
#  define MM_STORE_ASYNC 0
#endif
#ifndef MM_STORE_INFLIGHT
#  define MM_STORE_INFLIGHT 4
#endif
#define MM_MAX_OUTPUT_BUFFERS 16
#if HTP_MM_PHASE_PROFILE
unsigned long long g_mm_phase_us[13];
#  define MM_PHASE_T0() (HAP_perf_get_time_us())
#  define MM_PHASE_ADD(i, t0)                              \
    do {                                                   \
      g_mm_phase_us[(i)] += HAP_perf_get_time_us() - (t0); \
    } while (0)
#else
#  define MM_PHASE_T0() (0ULL)
#  define MM_PHASE_ADD(i, t0) \
    do {                      \
      (void) (t0);            \
    } while (0)
#endif

typedef uint64_t unaligned_uint64_t __attribute__((aligned(1)));

typedef struct {
    const uint8_t* rawInt4Data;
    uint8_t* dstWeight;
    int icP;
    int ocP;
    int ic_bytes;
    int ic;
    int oc;
    bool align;
    worker_synctoken_t sync_ctx;
} HtpOpsWeightReorderInt4State;

typedef struct {
    HtpOpsWeightReorderInt4State* state;
    uint8_t* local32x32;
    int block_start;
    int block_end;
} HtpOpsWeightReorderInt4Task;

static inline void htp_ops_weight_reorder_int4_block(const HtpOpsWeightReorderInt4State* state,
                                                      uint8_t* local32x32, int block) {
    const int y = block / state->icP;
    const int x = block - y * state->icP;
    if (!state->align) {
        memset(local32x32, 8, 32 * 32);
    }
    int ySta = y * 32;
    int yEnd = ySta + 32;
    if (yEnd > state->oc) {
        yEnd = state->oc;
    }
    int yCount = yEnd - ySta;
    int xSta = x * 16;
    int xEnd = xSta + 16;
    if (xEnd > state->ic_bytes) {
        xEnd = state->ic_bytes;
    }
    int xCount = xEnd - xSta;
    for (int yi = 0; yi < yCount; ++yi) {
        int sy = 32 * y + yi;
        for (int xi = 0; xi < xCount; ++xi) {
            int sx = x * 16 + xi;
            uint8_t val = state->rawInt4Data[(size_t)sy * state->ic_bytes + sx];
            local32x32[2 * xi * 32 + 2 * yi] = val >> 4;
            local32x32[2 * xi * 32 + 2 * yi + 1] = val & 0x0f;
        }
    }

    for (int q = 0; q < 8; ++q) {
        vmemu(local32x32 + q * 128) = Q6_Vb_vshuff_Vb(vmemu(local32x32 + q * 128));
    }

    uint8_t* dst_chunk = state->dstWeight + (size_t)block * 32 * 16;
    for (int q = 0; q < 4; ++q) {
        HVX_Vector v_low = vmemu(local32x32 + q * 256);
        HVX_Vector v_high = vmemu(local32x32 + q * 256 + 128);
        HVX_Vector v_shifted = Q6_Vh_vasl_VhR(v_high, 4);
        HVX_Vector v_packed = Q6_V_vor_VV(v_low, v_shifted);
        vmemu(dst_chunk + q * 128) = v_packed;
    }
}

static void htp_ops_weight_reorder_int4_worker(void* data, int worker_index) {
    (void)worker_index;
    HtpOpsWeightReorderInt4Task* task = (HtpOpsWeightReorderInt4Task*)data;
    for (int block = task->block_start; block < task->block_end; ++block) {
        htp_ops_weight_reorder_int4_block(task->state, task->local32x32, block);
    }
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

int htp_ops_weight_reorder_int4(uint8_t* dst_ptr, uint8_t* src_ptr, int ic, int oc, int alphaSize) {
    const uint8_t* rawInt4Data = src_ptr;
    size_t rawInt4Size = ((size_t)(ic + 1) / 2) * oc;
    const float* rawAlphaData = (const float*)(src_ptr + rawInt4Size);

    int icP = (ic + 31) / 32;
    int ocP = (oc + 31) / 32;
    size_t ic_bytes = (ic + 1) / 2;
    bool align = ic % 32 == 0 && oc % 32 == 0;

    uint8_t* dstWeight = dst_ptr;
    __fp16* dstScale = (__fp16*)(dst_ptr + (size_t)icP * ocP * 32 * 16);

    if (!align) {
        memset(dst_ptr, 0, (size_t)icP * ocP * 32 * 16 + (size_t)ocP * 32 * sizeof(__fp16));
    }
    uint8_t *vtcm_ptr = (uint8_t *)vtcm_manager_get_vtcm_base();
    const int total_blocks = icP * ocP;
    int task_count = (int)g_max_num_workers;
    if (task_count < 1) {
        task_count = 1;
    }
    if (task_count > total_blocks) {
        task_count = total_blocks;
    }
    uint8_t *local32x32 = (uint8_t *)vtcm_seq_alloc(&vtcm_ptr, (size_t)task_count * 32 * 32);

    HtpOpsWeightReorderInt4State state = {};
    state.rawInt4Data = rawInt4Data;
    state.dstWeight = dstWeight;
    state.icP = icP;
    state.ocP = ocP;
    state.ic_bytes = (int)ic_bytes;
    state.ic = ic;
    state.oc = oc;
    state.align = align;

    if (task_count <= 1) {
        for (int block = 0; block < total_blocks; ++block) {
            htp_ops_weight_reorder_int4_block(&state, local32x32, block);
        }
    } else {
        HtpOpsWeightReorderInt4Task tasks[g_max_num_workers];
        worker_pool_job_t job;
        job.fptr = htp_ops_weight_reorder_int4_worker;
        worker_pool_synctoken_init(&state.sync_ctx, task_count);
        const int blocks_per_task = (total_blocks + task_count - 1) / task_count;
        for (int i = 0; i < task_count; ++i) {
            const int block_start = i * blocks_per_task;
            int block_end = block_start + blocks_per_task;
            if (block_end > total_blocks) {
                block_end = total_blocks;
            }
            tasks[i].state = &state;
            tasks[i].local32x32 = local32x32 + (size_t)i * 32 * 32;
            tasks[i].block_start = block_start;
            tasks[i].block_end = block_end;
            job.dptr = tasks + i;
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&state.sync_ctx);
    }

    for (int y = 0; y < oc; ++y) {
        dstScale[y] = (__fp16)rawAlphaData[y];
    }

    return 0;
}


#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (2 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (4 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)

#define PREPARE_ACT_DMA(ox_val, buf_idx, out_idx) do { \
    out_idx = 0; \
    int base_offset = (buf_idx) * mp_chunk * kp * 32 * 32; \
    int cur_ox_end = (ox_val) + mp_chunk; \
    if (cur_ox_end > mp) cur_ox_end = mp; \
    for (int ox_iter = (ox_val); ox_iter < cur_ox_end; ++ox_iter) { \
      int ox_offset_inner = (ox_iter - (ox_val)) * kp * 32 * 32; \
      for (int k = 0; k < kp; ++k) { \
        int pack_idx = (k * 32) / pack; \
        int pack_inner = (k * 32) % pack; \
        int valid_xi = M - ox_iter * 32; \
        if (valid_xi > 32) valid_xi = 32; \
        if (valid_xi < 0) valid_xi = 0; \
        uint8_t* dst_addr = (uint8_t*)(vtcm_activation + base_offset + ox_offset_inner + k * 32 * 32); \
        if (valid_xi > 0) { \
          size_t src_offset = (size_t)(pack_idx * M + ox_iter * 32) * pack * sizeof(int16_t) + pack_inner * sizeof(int16_t); \
          const uint8_t* src_addr = a + src_offset; \
          memset(&act_descs[buf_idx][out_idx], 0, sizeof(dma_desc_2d_t)); \
          act_descs[buf_idx][out_idx].type       = DMA_DESC_TYPE_2D; \
          act_descs[buf_idx][out_idx].src_bypass = 0; \
          act_descs[buf_idx][out_idx].dst_bypass = 1; \
          act_descs[buf_idx][out_idx].ordered    = 1; \
          act_descs[buf_idx][out_idx].src              = (uint32_t) src_addr; \
          act_descs[buf_idx][out_idx].dst              = (uint32_t) dst_addr; \
          act_descs[buf_idx][out_idx].roi_width        = 32 * sizeof(int16_t); \
          act_descs[buf_idx][out_idx].roi_height       = valid_xi; \
          act_descs[buf_idx][out_idx].src_stride       = pack * sizeof(int16_t); \
          act_descs[buf_idx][out_idx].dst_stride       = 32 * sizeof(int16_t); \
          if (out_idx > 0) { \
            act_descs[buf_idx][out_idx - 1].next = (uint32_t)&act_descs[buf_idx][out_idx]; \
          } \
          out_idx++; \
        } \
      } \
    } \
} while(0)

#define ISSUE_ACT_DMA(ox_val, buf_idx) do { \
    int act_idx = 0; \
    PREPARE_ACT_DMA(ox_val, buf_idx, act_idx); \
    if (act_idx > 0) { \
      dmstart(&act_descs[buf_idx][0]); \
    } \
} while(0)

static const __fp16 q4_to_fp16_lut[64] __attribute__((aligned(128))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static inline void dequant_q4_tile_scaled(const uint8_t* src, __fp16* dst,
                                          HVX_Vector vlut_cvt, HVX_Vector vBlockScale) {
  const HVX_Vector* src_vec = (const HVX_Vector*)src;
  HVX_Vector* dst_vec = (HVX_Vector*)dst;
  HVX_Vector vq0 = src_vec[0];
  HVX_Vector vq1 = src_vec[1];
  HVX_Vector vq2 = src_vec[2];
  HVX_Vector vq3 = src_vec[3];
  HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
  HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
  HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
  HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);
  HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR_nomatch(vq0, vlut_cvt, 0);
  HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi0, vlut_cvt, 0);
  HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR_nomatch(vq1, vlut_cvt, 0);
  HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi1, vlut_cvt, 0);
  HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR_nomatch(vq2, vlut_cvt, 0);
  HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi2, vlut_cvt, 0);
  HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR_nomatch(vq3, vlut_cvt, 0);
  HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi3, vlut_cvt, 0);
  dst_vec[0] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), vBlockScale);
  dst_vec[1] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), vBlockScale);
  dst_vec[2] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), vBlockScale);
  dst_vec[3] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), vBlockScale);
  dst_vec[4] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), vBlockScale);
  dst_vec[5] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), vBlockScale);
  dst_vec[6] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), vBlockScale);
  dst_vec[7] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), vBlockScale);
  dst_vec[8] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), vBlockScale);
  dst_vec[9] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), vBlockScale);
  dst_vec[10] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), vBlockScale);
  dst_vec[11] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), vBlockScale);
  dst_vec[12] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), vBlockScale);
  dst_vec[13] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), vBlockScale);
  dst_vec[14] = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), vBlockScale);
  dst_vec[15] = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), vBlockScale);
}

static inline void process_q4_weight_lut(int oy_start, int start_idx, int count, int kp,
                                         const uint8_t *__restrict vtcm_weight_int4, __fp16 *__restrict vtcm_weight,
                                         const uint8_t *b_scale, int scale_block_num, int weight_is_vrmpy) {
  HVX_Vector vlut_cvt = vmemu(q4_to_fp16_lut);
  const int weight_int4_stride = 512 * kp;
  const int weight_fp16_stride = 1024 * kp;
  const uint8_t* src_oy = vtcm_weight_int4 + start_idx * weight_int4_stride;
  __fp16* dst_oy = vtcm_weight + start_idx * weight_fp16_stride;
  // Path A: the DMA'd tiles are in vrmpy layout and b_scale points at fp32 vrmpy
  // block scales (sw[(oy*nblk+b)*32+ocIn]). Convert each 512B tile to HMX and build
  // the per-block dup fp16 scale on the fly, then run the unchanged dequant. Only
  // block-quant reaches here in Path A (scale_block_num > 1).
  if (weight_is_vrmpy && scale_block_num > 1) {
    const float                          *sw = (const float *) b_scale;
    __attribute__((aligned(128))) uint8_t tmp[512];
    for (int local_oy = 0; local_oy < count; ++local_oy) {
      const int      oy    = oy_start + start_idx + local_oy;
      const float   *sw_oy = sw + (size_t) oy * scale_block_num * 32;
      const uint8_t *src   = src_oy;
      __fp16        *dst   = dst_oy;
      for (int k = 0; k < kp; ++k) {
        const int  scale_idx   = (int) (((size_t) k * scale_block_num) / kp);
        HVX_Vector vBlockScale = vrmpy_scale_block_dup_hf(sw_oy, scale_idx);
        vrmpy_tile_to_hmx_int4_512(tmp, src);
        dequant_q4_tile_scaled(tmp, dst, vlut_cvt, vBlockScale);
        src += 512;
        dst += 1024;
      }
      src_oy += weight_int4_stride;
      dst_oy += weight_fp16_stride;
    }
    return;
  }
  if (scale_block_num > 1) {
    const int scale_block_bytes = 128;
    for (int local_oy = 0; local_oy < count; ++local_oy) {
      const int oy = oy_start + start_idx + local_oy;
      const uint8_t* src = src_oy;
      __fp16* dst = dst_oy;
      const uint8_t* block_scale_ptr = b_scale + (size_t)oy * scale_block_num * scale_block_bytes;
      if (scale_block_num == kp) {
        for (int k = 0; k < kp; ++k) {
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vmemu(block_scale_ptr));
          block_scale_ptr += scale_block_bytes;
          src += 512;
          dst += 1024;
        }
      } else if (kp % scale_block_num == 0) {
        const int k_per_scale = kp / scale_block_num;
        int scale_idx = 0;
        int next_scale_k = k_per_scale;
        HVX_Vector vBlockScale = vmemu(block_scale_ptr);
        for (int k = 0; k < kp; ++k) {
          if (k == next_scale_k) {
            ++scale_idx;
            next_scale_k += k_per_scale;
            vBlockScale = vmemu(block_scale_ptr + scale_idx * scale_block_bytes);
          }
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vBlockScale);
          src += 512;
          dst += 1024;
        }
      } else {
        for (int k = 0; k < kp; ++k) {
          const int scale_idx = (k * scale_block_num) / kp;
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vmemu(block_scale_ptr + scale_idx * scale_block_bytes));
          src += 512;
          dst += 1024;
        }
      }
      src_oy += weight_int4_stride;
      dst_oy += weight_fp16_stride;
    }
    return;
  }

  for (int local_oy = 0; local_oy < count; ++local_oy) {
    const uint8_t* src = src_oy;
    __fp16* dst = dst_oy;
    for (int k = 0; k < kp; ++k) {
      const HVX_Vector* src_vec = (const HVX_Vector*)src;
      HVX_Vector* dst_vec = (HVX_Vector*)dst;
      HVX_Vector vq0 = src_vec[0];
      HVX_Vector vq1 = src_vec[1];
      HVX_Vector vq2 = src_vec[2];
      HVX_Vector vq3 = src_vec[3];
      HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
      HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
      HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
      HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);
      HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR_nomatch(vq0, vlut_cvt, 0);
      HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi0, vlut_cvt, 0);
      HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR_nomatch(vq1, vlut_cvt, 0);
      HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi1, vlut_cvt, 0);
      HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR_nomatch(vq2, vlut_cvt, 0);
      HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi2, vlut_cvt, 0);
      HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR_nomatch(vq3, vlut_cvt, 0);
      HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR_nomatch(v_hi3, vlut_cvt, 0);
      dst_vec[0] = Q6_V_lo_W(vp_lo0);
      dst_vec[1] = Q6_V_hi_W(vp_lo0);
      dst_vec[2] = Q6_V_lo_W(vp_hi0);
      dst_vec[3] = Q6_V_hi_W(vp_hi0);
      dst_vec[4] = Q6_V_lo_W(vp_lo1);
      dst_vec[5] = Q6_V_hi_W(vp_lo1);
      dst_vec[6] = Q6_V_lo_W(vp_hi1);
      dst_vec[7] = Q6_V_hi_W(vp_hi1);
      dst_vec[8] = Q6_V_lo_W(vp_lo2);
      dst_vec[9] = Q6_V_hi_W(vp_lo2);
      dst_vec[10] = Q6_V_lo_W(vp_hi2);
      dst_vec[11] = Q6_V_hi_W(vp_hi2);
      dst_vec[12] = Q6_V_lo_W(vp_lo3);
      dst_vec[13] = Q6_V_hi_W(vp_lo3);
      dst_vec[14] = Q6_V_lo_W(vp_hi3);
      dst_vec[15] = Q6_V_hi_W(vp_hi3);
      src += 512;
      dst += 1024;
    }
    src_oy += weight_int4_stride;
    dst_oy += weight_fp16_stride;
  }
}

typedef struct {
  int oy_start;
  int start_idx;
  int count;
  int kp;
  int scale_block_num;
  int                  weight_is_vrmpy;
  const uint8_t* b_scale;
  const uint8_t* vtcm_weight_int4;
  __fp16* vtcm_weight;
  const dma_desc_1d_t* depend;
  worker_synctoken_t* shared_sync;
} process_q4_weight_task_state_t;

static inline void wait_q4_weight_dma_done(const dma_desc_1d_t* depend) {
  volatile uint32_t *ctrl_word = (volatile uint32_t*)&depend->dstate_order_bypass_type_length;
  while (((*ctrl_word >> 31) & 0x1) == 0) {
    asm volatile("nop");
  }
}

static void process_q4_weight_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  process_q4_weight_task_state_t* state = (process_q4_weight_task_state_t*)data;
  // [12] spinning on this chunk's DMA, [11] the dequant itself, both summed across
  // workers: [11] against the dispatch thread's [1] separates real work from handshake.
  unsigned long long              _td   = MM_PHASE_T0();
  wait_q4_weight_dma_done(state->depend);
  MM_PHASE_ADD(12, _td);
  unsigned long long _tw = MM_PHASE_T0();
  process_q4_weight_lut(state->oy_start, state->start_idx, state->count, state->kp, state->vtcm_weight_int4,
                        state->vtcm_weight, state->b_scale, state->scale_block_num, state->weight_is_vrmpy);
  MM_PHASE_ADD(11, _tw);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

static inline void hmx_load_q4_tiles(const __fp16* activation, const __fp16* weight, int kp) {
  if (kp == 32) {
    hmx_load_tiles_fp16(activation, weight, 32);
    return;
  }
  if (kp == 64) {
    hmx_load_tiles_fp16(activation, weight, 32);
    hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
    return;
  }
  if (kp == 88) {
    hmx_load_tiles_fp16(activation, weight, 32);
    hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
    hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 24);
    return;
  }
  if (kp == 96) {
    hmx_load_tiles_fp16(activation, weight, 32);
    hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
    hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 32);
    return;
  }
  if (kp == 128) {
    hmx_load_tiles_fp16(activation, weight, 32);
    hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
    hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 32);
    hmx_load_tiles_fp16(activation + 96 * 1024, weight + 96 * 1024, 32);
    return;
  }
  if ((kp & 31) == 0) {
    for (int k = 0; k < kp; k += 32) {
      hmx_load_tiles_fp16(activation + k * 1024, weight + k * 1024, 32);
    }
  } else {
    for (int k = 0; k < kp; k += 32) {
      int kend = k + 32;
      if (kend > kp) {
        kend = kp;
      }
      hmx_load_tiles_fp16(activation + k * 1024, weight + k * 1024, kend - k);
    }
  }
}

static inline void shuffle_q4_activation_tiles(__fp16* vtcm_activation, int base_offset,
                                               int ox_start, int ox_end, int kp, int M) {
  for (int ox = ox_start; ox < ox_end; ++ox) {
    int ox_offset = base_offset + (ox - ox_start) * kp * 32 * 32;
    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi < 0) valid_xi = 0;
    int valid_vecs = (valid_xi + 1) / 2;
    for (int k = 0; k < kp; ++k) {
      __fp16* act_tile = vtcm_activation + ox_offset + k * 32 * 32;
      for (int xi = 0; xi < valid_vecs; ++xi) {
        HVX_Vector* va = (HVX_Vector*)(act_tile + xi * 64);
        va[0] = Q6_Vh_vshuff_Vh(va[0]);
      }
    }
  }
}

static inline void copy_shuffle_q4_activation_tiles_pack64(__fp16* vtcm_activation, const uint8_t* a,
                                                           int base_offset, int ox_start,
                                                           int ox_end, int kp, int M) {
  for (int ox = ox_start; ox < ox_end; ++ox) {
    const int ox_offset = base_offset + (ox - ox_start) * kp * 32 * 32;
    int valid_rows = M - ox * 32;
    if (valid_rows > 32) valid_rows = 32;
    if (valid_rows <= 0) continue;
    // Not latency-bound: an explicit one-block-ahead l2fetch of these 32x128B DDR
    // blocks (k-pairs are M*128B apart, so the hardware prefetcher cannot see the
    // pattern) measured 1.7ms *worse* on a 1053-token prefill. Left out deliberately.
    for (int k = 0; k < kp; k += 2) {
      const int has_high_tile = (k + 1) < kp;
      const uint8_t* src_base = a + (size_t)(((k >> 1) * M + ox * 32) * 64) * sizeof(int16_t);
      __fp16* tile0 = vtcm_activation + ox_offset + k * 32 * 32;
      __fp16* tile1 = tile0 + 32 * 32;
      int r = 0;
      for (; r <= valid_rows - 2; r += 2) {
        HVX_Vector v0 = vmemu((const HVX_Vector*)(src_base + (size_t)r * 64 * sizeof(int16_t)));
        HVX_Vector v1 = vmemu((const HVX_Vector*)(src_base + (size_t)(r + 1) * 64 * sizeof(int16_t)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        if (has_high_tile) {
          vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
        }
      }
      if (r < valid_rows) {
        HVX_Vector v0 = vmemu((const HVX_Vector*)(src_base + (size_t)r * 64 * sizeof(int16_t)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        if (has_high_tile) {
          vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
        }
      }
    }
  }
}

typedef struct {
  __fp16* vtcm_activation;
  const uint8_t* a;
  int base_offset;
  int ox_start;
  int ox_end;
  int kp;
  int M;
  worker_synctoken_t* shared_sync;
} copy_shuffle_q4_activation_task_state_t;

static void copy_shuffle_q4_activation_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  copy_shuffle_q4_activation_task_state_t* state = (copy_shuffle_q4_activation_task_state_t*)data;
  copy_shuffle_q4_activation_tiles_pack64(state->vtcm_activation, state->a, state->base_offset,
                                          state->ox_start, state->ox_end, state->kp, state->M);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

static inline void copy_shuffle_q4_activation_tiles_pack64_parallel(__fp16* vtcm_activation,
                                                                    const uint8_t* a,
                                                                    int base_offset, int ox_start,
                                                                    int ox_end, int kp, int M) {
  const int ox_count = ox_end - ox_start;
  int task_count = (int)g_max_num_workers;
  if (task_count > ox_count) {
    task_count = ox_count;
  }
  if (task_count <= 1) {
    copy_shuffle_q4_activation_tiles_pack64(vtcm_activation, a, base_offset, ox_start, ox_end, kp, M);
    return;
  }

  copy_shuffle_q4_activation_task_state_t tasks[g_max_num_workers];
  worker_synctoken_t sync_token;
  worker_pool_synctoken_init(&sync_token, task_count);
  const int ox_per_task = (ox_count + task_count - 1) / task_count;
  for (int i = 0; i < task_count; ++i) {
    const int task_ox_start = ox_start + i * ox_per_task;
    int task_ox_end = task_ox_start + ox_per_task;
    if (task_ox_end > ox_end) {
      task_ox_end = ox_end;
    }
    tasks[i].vtcm_activation = vtcm_activation;
    tasks[i].a = a;
    tasks[i].base_offset = base_offset + (task_ox_start - ox_start) * kp * 32 * 32;
    tasks[i].ox_start = task_ox_start;
    tasks[i].ox_end = task_ox_end;
    tasks[i].kp = kp;
    tasks[i].M = M;
    tasks[i].shared_sync = &sync_token;

    worker_pool_job_t job;
    job.fptr = copy_shuffle_q4_activation_worker_loop;
    job.dptr = &tasks[i];
    if (worker_pool_submit(NULL, job) != 0) {
      copy_shuffle_q4_activation_worker_loop(&tasks[i], 0);
    }
  }
  worker_pool_synctoken_wait(&sync_token);
}

static inline HVX_Vector load_q4_output_scale(const uint8_t* vtcm_scales, int oy_start, int oy) {
  const __fp16* scale_ptr = (const __fp16*)(vtcm_scales + (oy - oy_start) * 64);
  HVX_Vector vScale_raw = vmemu(scale_ptr);
  HVX_Vector vScale_rot = Q6_V_vror_VR(vScale_raw, 64);
  return Q6_V_valign_VVR(vScale_raw, vScale_rot, 64);
}

static inline void init_identity_q4_output_scales(uint8_t* vtcm_scales, int count) {
  __fp16* scale_ptr = (__fp16*)vtcm_scales;
  for (int i = 0; i < count * 32; ++i) {
    scale_ptr[i] = (__fp16)1.0f;
  }
}

static inline HVX_Vector post_q4_output_vec_scale(HVX_Vector v, HVX_Vector vScale) {
  return Q6_Vhf_vmpy_VhfVhf(Q6_Vh_vdeal_Vh(v), vScale);
}

static inline HVX_Vector post_q4_output_vec_identity(HVX_Vector v) {
  return Q6_Vh_vdeal_Vh(v);
}

static inline HVX_Vector post_q4_output_vec_bias(HVX_Vector v, HVX_Vector vScale, HVX_Vector vBias) {
  v = Q6_Vhf_vmpy_VhfVhf(Q6_Vh_vdeal_Vh(v), vScale);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vBias));
}

static inline HVX_Vector post_q4_output_vec_identity_bias(HVX_Vector v, HVX_Vector vBias) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_Vh_vdeal_Vh(v), vBias));
}

static inline void store_q4_output_tile(uint8_t* c, __fp16* vtcm_output, const uint8_t* bias,
                                        const uint8_t* vtcm_scales, int M, int ox, int oy_start, int oy) {
  HVX_Vector vBias = Q6_V_vzero();
  if (bias) {
    const __fp16* bias_ptr = (const __fp16*)bias + oy * 32;
    HVX_Vector vBias_raw = vmemu(bias_ptr);
    HVX_Vector vBias_rot = Q6_V_vror_VR(vBias_raw, 64);
    vBias = Q6_V_valign_VVR(vBias_raw, vBias_rot, 64);
  }
  const int useScale = vtcm_scales != NULL;
  HVX_Vector vScale = useScale ? load_q4_output_scale(vtcm_scales, oy_start, oy) : Q6_V_vzero();
  const int hasBias = bias != NULL;
  const int pack_idx = (oy * 32) / 64;
  const int pack_inner = (oy * 32) % 64;
  int valid_xi = M - ox * 32;
  if (valid_xi > 32) valid_xi = 32;
  if (valid_xi < 0) valid_xi = 0;
  const int xi_limit = valid_xi & ~1;
  HVX_VectorPred q = pack_inner == 0 ? Q6_Q_vsetq_R(64) : Q6_Q_not_Q(Q6_Q_vsetq_R(64));

  HVX_Vector* src_ptr = (HVX_Vector*)vtcm_output;
  uint8_t* dst_ptr = c + (size_t)(pack_idx * M + ox * 32) * 128;
  int xi = 0;
  if (hasBias) {
    if (useScale) {
      for (; xi < xi_limit; xi += 2) {
        HVX_Vector v = post_q4_output_vec_bias(*src_ptr++, vScale, vBias);
        HVX_Vector v_rot = Q6_V_valign_VVR(v, v, 64);
        HVX_Vector v_first = pack_inner == 0 ? v : v_rot;
        HVX_Vector v_second = pack_inner == 0 ? v_rot : v;
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
        dst_ptr += 256;
      }
      if (xi < valid_xi) {
        HVX_Vector v = post_q4_output_vec_bias(*src_ptr++, vScale, vBias);
        if (pack_inner != 0) {
          v = Q6_V_valign_VVR(v, v, 64);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
      }
    } else {
      for (; xi < xi_limit; xi += 2) {
        HVX_Vector v = post_q4_output_vec_identity_bias(*src_ptr++, vBias);
        HVX_Vector v_rot = Q6_V_valign_VVR(v, v, 64);
        HVX_Vector v_first = pack_inner == 0 ? v : v_rot;
        HVX_Vector v_second = pack_inner == 0 ? v_rot : v;
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
        dst_ptr += 256;
      }
      if (xi < valid_xi) {
        HVX_Vector v = post_q4_output_vec_identity_bias(*src_ptr++, vBias);
        if (pack_inner != 0) {
          v = Q6_V_valign_VVR(v, v, 64);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
      }
    }
  } else {
    if (useScale) {
      for (; xi < xi_limit; xi += 2) {
        HVX_Vector v = post_q4_output_vec_scale(*src_ptr++, vScale);
        HVX_Vector v_rot = Q6_V_valign_VVR(v, v, 64);
        HVX_Vector v_first = pack_inner == 0 ? v : v_rot;
        HVX_Vector v_second = pack_inner == 0 ? v_rot : v;
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
        dst_ptr += 256;
      }
      if (xi < valid_xi) {
        HVX_Vector v = post_q4_output_vec_scale(*src_ptr++, vScale);
        if (pack_inner != 0) {
          v = Q6_V_valign_VVR(v, v, 64);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
      }
    } else {
      for (; xi < xi_limit; xi += 2) {
        HVX_Vector v = post_q4_output_vec_identity(*src_ptr++);
        HVX_Vector v_rot = Q6_V_valign_VVR(v, v, 64);
        HVX_Vector v_first = pack_inner == 0 ? v : v_rot;
        HVX_Vector v_second = pack_inner == 0 ? v_rot : v;
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
        dst_ptr += 256;
      }
      if (xi < valid_xi) {
        HVX_Vector v = post_q4_output_vec_identity(*src_ptr++);
        if (pack_inner != 0) {
          v = Q6_V_valign_VVR(v, v, 64);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
      }
    }
  }
}

static inline int pack_q4_output_tile_pair(uint8_t* packed, __fp16* vtcm_output0, __fp16* vtcm_output1,
                                           const uint8_t* bias, const uint8_t* vtcm_scales,
                                           int M, int ox, int oy_start, int oy) {
  HVX_Vector vBias0 = Q6_V_vzero();
  HVX_Vector vBias1 = Q6_V_vzero();
  if (bias) {
    const __fp16* bias_ptr0 = (const __fp16*)bias + oy * 32;
    HVX_Vector vBias_raw0 = vmemu(bias_ptr0);
    HVX_Vector vBias_rot0 = Q6_V_vror_VR(vBias_raw0, 64);
    vBias0 = Q6_V_valign_VVR(vBias_raw0, vBias_rot0, 64);
    const __fp16* bias_ptr1 = (const __fp16*)bias + (oy + 1) * 32;
    HVX_Vector vBias_raw1 = vmemu(bias_ptr1);
    HVX_Vector vBias_rot1 = Q6_V_vror_VR(vBias_raw1, 64);
    vBias1 = Q6_V_valign_VVR(vBias_raw1, vBias_rot1, 64);
  }
  const int useScale = vtcm_scales != NULL;
  HVX_Vector vScale0 = useScale ? load_q4_output_scale(vtcm_scales, oy_start, oy) : Q6_V_vzero();
  HVX_Vector vScale1 = useScale ? load_q4_output_scale(vtcm_scales, oy_start, oy + 1) : Q6_V_vzero();
  const int hasBias = bias != NULL;
  int valid_xi = M - ox * 32;
  if (valid_xi > 32) valid_xi = 32;
  if (valid_xi < 0) valid_xi = 0;
  const int xi_limit = valid_xi & ~1;
  HVX_VectorPred q_low = Q6_Q_vsetq_R(64);

  HVX_Vector* src0_ptr = (HVX_Vector*)vtcm_output0;
  HVX_Vector* src1_ptr = (HVX_Vector*)vtcm_output1;
  uint8_t* dst_ptr = packed;
  if (!useScale) {
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector v0 = hasBias ? post_q4_output_vec_identity_bias(*src0_ptr++, vBias0) :
                                post_q4_output_vec_identity(*src0_ptr++);
      HVX_Vector v1 = hasBias ? post_q4_output_vec_identity_bias(*src1_ptr++, vBias1) :
                                post_q4_output_vec_identity(*src1_ptr++);
      HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
      dst_ptr += 256;
    }
    if (xi < valid_xi) {
      HVX_Vector v0 = hasBias ? post_q4_output_vec_identity_bias(*src0_ptr++, vBias0) :
                                post_q4_output_vec_identity(*src0_ptr++);
      HVX_Vector v1 = hasBias ? post_q4_output_vec_identity_bias(*src1_ptr++, vBias1) :
                                post_q4_output_vec_identity(*src1_ptr++);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    }
    return valid_xi * 128;
  }
  if (valid_xi == 32) {
#define PACK_Q4_OUTPUT_PAIR_FULL_STEP(POST0, POST1) do { \
      HVX_Vector v0 = (POST0); \
      HVX_Vector v1 = (POST1); \
      HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64); \
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64); \
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot); \
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1); \
      dst_ptr += 256; \
    } while (0)
    if (hasBias) {
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0),
                                    post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1));
    } else {
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
      PACK_Q4_OUTPUT_PAIR_FULL_STEP(post_q4_output_vec_scale(*src0_ptr++, vScale0),
                                    post_q4_output_vec_scale(*src1_ptr++, vScale1));
    }
#undef PACK_Q4_OUTPUT_PAIR_FULL_STEP
    return 32 * 128;
  }
  int xi = 0;
  if (hasBias) {
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector v0 = post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0);
      HVX_Vector v1 = post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1);
      HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
      dst_ptr += 256;
    }
    if (xi < valid_xi) {
      HVX_Vector v0 = post_q4_output_vec_bias(*src0_ptr++, vScale0, vBias0);
      HVX_Vector v1 = post_q4_output_vec_bias(*src1_ptr++, vScale1, vBias1);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    }
  } else {
    for (; xi < xi_limit; xi += 2) {
      HVX_Vector v0 = post_q4_output_vec_scale(*src0_ptr++, vScale0);
      HVX_Vector v1 = post_q4_output_vec_scale(*src1_ptr++, vScale1);
      HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
      vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
      dst_ptr += 256;
    }
    if (xi < valid_xi) {
      HVX_Vector v0 = post_q4_output_vec_scale(*src0_ptr++, vScale0);
      HVX_Vector v1 = post_q4_output_vec_scale(*src1_ptr++, vScale1);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    }
  }
  return valid_xi * 128;
}

static inline void store_q4_output_range(uint8_t* c, __fp16* vtcm_output, const uint8_t* bias,
                                         const uint8_t* vtcm_scales, int M, int ox,
                                         int oy_start, int oy_begin, int oy_end, int pack) {
  for (int oy = oy_begin; oy < oy_end; ++oy) {
    __fp16* output_tile = vtcm_output + (oy - oy_start) * 1024;
    if (pack == 64 && oy + 1 < oy_end && ((oy * 32) & 63) == 0) {
      __fp16* output_tile_next = vtcm_output + (oy + 1 - oy_start) * 1024;
      uint8_t* dst_ptr = c + (size_t)(((oy * 32) / 64) * M + ox * 32) * 128;
      pack_q4_output_tile_pair(dst_ptr, output_tile, output_tile_next, bias, vtcm_scales,
                               M, ox, oy_start, oy);
      ++oy;
    } else {
      store_q4_output_tile(c, output_tile, bias, vtcm_scales, M, ox, oy_start, oy);
    }
  }
}

typedef struct {
  uint8_t* c;
  __fp16* vtcm_output;
  const uint8_t* bias;
  const uint8_t* vtcm_scales;
  int M;
  int ox;
  int oy_start;
  int oy_begin;
  int oy_end;
  int pack;
  worker_synctoken_t* shared_sync;
} store_q4_output_task_state_t;

static void store_q4_output_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  store_q4_output_task_state_t* state = (store_q4_output_task_state_t*)data;
  // [10] sums the per-task store time across workers, so [10] vs the main thread's
  // [3] stall tells how much store concurrency is actually realised. Diagnostic only
  // (the accumulate races between workers; close enough at 2-4 in flight).
  unsigned long long            _ts   = MM_PHASE_T0();
  store_q4_output_range(state->c, state->vtcm_output, state->bias, state->vtcm_scales, state->M,
                        state->ox, state->oy_start, state->oy_begin, state->oy_end, state->pack);
  MM_PHASE_ADD(10, _ts);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

// Temporary Path A Step 2.3.2 validation. Validated on device 2026-08-06
// (HVX == scalar; scalar == reorderInt4WeightForHmx on host). Left in, gated OFF,

int hmx_matmulq4fp16(uint8_t *c, const uint8_t *a, const uint8_t *b, const uint8_t *b_scale, const uint8_t *bias, int M,
                     int K, int N, int mp_max, int np_max, int kp_max, int scale_block_num, int scale_asymmetric,
                     int weight_is_vrmpy) {
  unsigned long long _mm_tot = MM_PHASE_T0();
  if (scale_block_num <= 0) scale_block_num = 1;
  (void)scale_asymmetric;
  const int dequant_in_weight = scale_block_num > 1;
  int np_chunk = np_max;
  if (np_chunk == 0) np_chunk = 1;
  int mp_chunk = mp_max;
  if (mp_chunk == 0) mp_chunk = 1;

  int weight_bytes_per_np = 32 * K * sizeof(int16_t);
  int weight_int4_bytes_per_np = 32 * K / 2;
  int act_bytes_per_mp = 32 * K * sizeof(int16_t);
  int mp = (M + 31) / 32;
  int np = N / 32;
  int kp = K / 32;
  const bool reuse_activation = (mp > 0 && mp <= mp_chunk);
  const int activation_buffers = reuse_activation ? 1 : 2;
  // Handing the store to workers (MM_STORE_ASYNC=1) is measurably *worse* than storing on
  // the dispatch thread now that each store covers a whole oy chunk: on a 1053-token prefill
  // the store costs 26.1ms inline, but 34.5ms of main-thread stall when submitted per ox,
  // because ~700MB of paired full-vector writes already sit at the DDR write ceiling
  // (~27 GB/s) and only the per-task handshake is added. Deepening the ring to hide the
  // worker wake-up latency does not recover it either (36.4 -> 34.5ms at depth 2 -> 4).
  // Kept gated off, with the ring depth sized from what VTCM can spare, for future shapes
  // where a store task is large enough to pay for the handshake.
  const int    async_output_candidate          = (MM_STORE_ASYNC && mp > 1 && g_max_num_workers > 1);
  const int candidate_cross_oy_output_store = async_output_candidate && ((np_chunk & 1) == 0);
  const int    store_groups                    = candidate_cross_oy_output_store ? 2 : 1;
  const size_t output_buffer_bytes             = (size_t) np_chunk * 1024 * sizeof(__fp16);
  const int candidate_scale_buffers = async_output_candidate ? 2 : 1;
  const size_t q4_vtcm_safe_size = 8 * 1024 * 1024 - 16 * 1024;
  const size_t async_base_vtcm_bytes =
    (size_t) np_chunk * weight_bytes_per_np + (size_t) np_chunk * weight_int4_bytes_per_np +
    (size_t) activation_buffers * mp_chunk * act_bytes_per_mp + (size_t) np_chunk * 256 +
    (size_t) candidate_scale_buffers * (np_chunk * 64 + 64);
  int store_inflight = MM_STORE_INFLIGHT;
  if (store_inflight > (int) g_max_num_workers) {
    store_inflight = (int) g_max_num_workers;
  }
  if (store_inflight * store_groups > MM_MAX_OUTPUT_BUFFERS) {
    store_inflight = MM_MAX_OUTPUT_BUFFERS / store_groups;
  }
  if (async_base_vtcm_bytes < q4_vtcm_safe_size && output_buffer_bytes > 0) {
    const int fits = (int) ((q4_vtcm_safe_size - async_base_vtcm_bytes) / output_buffer_bytes) / store_groups;
    if (store_inflight > fits) {
      store_inflight = fits;
    }
  } else {
    store_inflight = 0;
  }
  const int async_output_store    = async_output_candidate && store_inflight >= 2;
  const int cross_oy_output_store = async_output_store && candidate_cross_oy_output_store;
  const int output_buffers        = async_output_store ? store_inflight * store_groups : 1;
  const int scale_buffer_bytes = np_chunk * 64 + 64;
  const int scale_buffers = async_output_store ? 2 : 1;

  const int weight_buffers = 1;

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) weight_buffers * np_chunk * weight_bytes_per_np);
  uint8_t *vtcm_weight_int4 =
    (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, (size_t) weight_buffers * np_chunk * weight_int4_bytes_per_np);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_buffers * mp_chunk * act_bytes_per_mp);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_buffers * np_chunk * 1024 * sizeof(__fp16));
  HVX_Vector *vtcm_hmx_scales = (HVX_Vector *) vtcm_seq_alloc(&vtcm_ptr, np_chunk * 256);
  uint8_t  *vtcm_scales     = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, scale_buffers * scale_buffer_bytes);
  dma_desc_2d_t (*act_descs)[mp_chunk * kp] =
      (dma_desc_2d_t (*)[mp_chunk * kp]) memalign(64, (size_t)2 * mp_chunk * kp * sizeof(dma_desc_2d_t));
  if (act_descs == NULL) {
    return 1;
  }
  memset(vtcm_scales, 0, scale_buffers * scale_buffer_bytes);
  if (dequant_in_weight) {
    init_identity_q4_output_scales(vtcm_scales, np_chunk);
  }

  hmx_manager_enable_execution();
  hmx_init_column_scales(vtcm_hmx_scales, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(vtcm_hmx_scales);

  int pack = 64;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif
  bool activation_prepared = false;
  worker_synctoken_t           output_store_sync[MM_MAX_OUTPUT_BUFFERS];
  store_q4_output_task_state_t output_store_tasks[MM_MAX_OUTPUT_BUFFERS];
  int                          output_store_active[MM_MAX_OUTPUT_BUFFERS] = { 0 };

  MM_PHASE_ADD(8, _mm_tot);  // setup: entry -> loop start
  for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
    int oy_end = oy_start + np_chunk;
    if (oy_end > np) oy_end = np;
    const int oy_block = oy_start / np_chunk;
    const int scale_buf_idx = async_output_store ? (oy_block & 1) : 0;
    if (cross_oy_output_store) {
      const int output_group = scale_buf_idx * store_inflight;
      for (int i = 0; i < store_inflight; ++i) {
        const int output_buf_idx = output_group + i;
        if (output_store_active[output_buf_idx]) {
          worker_pool_synctoken_wait(&output_store_sync[output_buf_idx]);
          output_store_active[output_buf_idx] = 0;
        }
      }
    }
    uint8_t* current_vtcm_scales = vtcm_scales + scale_buf_idx * scale_buffer_bytes;

    // Weight staging buffer for this oy_block. Every weight DMA / dequant / HMX read in this
    // iteration goes through w_cur / w4_cur, so a future double-buffering step only has to point
    // these two at the parity half.
    __fp16  *w_cur  = vtcm_weight;
    uint8_t *w4_cur = vtcm_weight_int4;

    int weight_dma_count = oy_end - oy_start;

    int num_chunks = (int)g_max_num_workers;
    if (num_chunks > weight_dma_count) {
      num_chunks = weight_dma_count;
    }
    if (num_chunks < 1) {
      num_chunks = 1;
    }
    int chunk_size = (weight_dma_count + num_chunks - 1) / num_chunks;
    if (chunk_size > 1) {
      chunk_size = (chunk_size + 1) & ~1;
    }
    int chunk_counts[num_chunks];
    int chunk_starts[num_chunks];
    int current_start = 0;
    int valid_chunks = 0;
    for (int i = 0; i < num_chunks; ++i) {
      chunk_starts[i] = current_start;
      int end = current_start + chunk_size;
      if (end >= weight_dma_count) {
        end = weight_dma_count;
      }
      chunk_counts[i] = end - current_start;
      ++valid_chunks;
      current_start = end;
      if (current_start >= weight_dma_count) {
        break;
      }
    }

    _Alignas(64) dma_desc_1d_t weight_desc[num_chunks];
    _Alignas(64) dma_desc_2d_t scale_desc[1];
#define SET_WEIGHT_DMA(start_idx, count, desc_idx, next_ptr, pre_index)                                     \
  do {                                                                                                      \
    {                                                                                                       \
      if ((pre_index) >= 0)                                                                                 \
        weight_desc[pre_index].next = (uint32_t) &weight_desc[desc_idx];                                    \
      memset(&weight_desc[desc_idx], 0, sizeof(dma_desc_1d_t));                                             \
      weight_desc[desc_idx].next       = (uint32_t) (next_ptr);                                             \
      weight_desc[desc_idx].length     = 16 * 32 * kp * (count);                                            \
      weight_desc[desc_idx].type       = DMA_DESC_TYPE_1D;                                                  \
      weight_desc[desc_idx].src_bypass = 1;                                                                 \
      weight_desc[desc_idx].dst_bypass = 1;                                                                 \
      weight_desc[desc_idx].ordered    = 1;                                                                 \
      weight_desc[desc_idx].dstate     = DMA_DESC_DSTATE_PENDING;                                           \
      weight_desc[desc_idx].src        = (uint32_t) (b + (size_t) (oy_start + (start_idx)) * kp * 32 * 16); \
      weight_desc[desc_idx].dst        = (uint32_t) (w4_cur + (start_idx) * 512 * kp);                      \
    }                                                                                                       \
  } while (0)

    int act_buf_idx = 0;
    int act_idx = 0;
    if (!reuse_activation && mp > 0) {
        PREPARE_ACT_DMA(0, 0, act_idx);
    }
    if (!dequant_in_weight) {
      memset(&scale_desc[0], 0, sizeof(dma_desc_2d_t));
      scale_desc[0].type       = DMA_DESC_TYPE_2D;
      scale_desc[0].src_bypass = 1;
      scale_desc[0].dst_bypass = 1;
      scale_desc[0].ordered    = 1;
      scale_desc[0].src        = (uint32_t) (b_scale + oy_start * 32 * sizeof(__fp16));
      scale_desc[0].dst        = (uint32_t) current_vtcm_scales;
      scale_desc[0].roi_width  = 32 * sizeof(__fp16);
      scale_desc[0].roi_height = weight_dma_count;
      scale_desc[0].src_stride = 32 * sizeof(__fp16);
      scale_desc[0].dst_stride = 64;
      scale_desc[0].next       = 0;
    }
    uint32_t scale_dma_ptr = !dequant_in_weight ? (uint32_t)&scale_desc[0] : 0;
    uint32_t tail_dma_ptr = 0;
    if (act_idx > 0) {
      if (scale_dma_ptr != 0) {
        act_descs[0][act_idx - 1].next = scale_dma_ptr;
      }
      tail_dma_ptr = (uint32_t)&act_descs[0][0];
    } else {
      tail_dma_ptr = scale_dma_ptr;
    }

    SET_WEIGHT_DMA(chunk_starts[0], chunk_counts[0], 0, 0, -1);
    for (int i = 1; i < valid_chunks; ++i) {
      SET_WEIGHT_DMA(chunk_starts[i], chunk_counts[i], i, 0, i - 1);
    }
    weight_desc[valid_chunks - 1].next = tail_dma_ptr;

    dmstart(&weight_desc[0]);

    if (reuse_activation && !activation_prepared) {
      unsigned long long _ta = MM_PHASE_T0();
      copy_shuffle_q4_activation_tiles_pack64_parallel(vtcm_activation, a, 0, 0, mp, kp, M);
      MM_PHASE_ADD(9, _ta);
      activation_prepared = true;
    }

    worker_synctoken_t sync_token[num_chunks];
    process_q4_weight_task_state_t chunk_states[num_chunks];
    int chunk_weight_ready[num_chunks];
    for (int i = 0; i < valid_chunks; ++i) {
      chunk_weight_ready[i] = 0;
      worker_pool_synctoken_init(&sync_token[i], 1);
      chunk_states[i].oy_start = oy_start;
      chunk_states[i].start_idx = chunk_starts[i];
      chunk_states[i].count = chunk_counts[i];
      chunk_states[i].kp = kp;
      chunk_states[i].scale_block_num = scale_block_num;
      chunk_states[i].weight_is_vrmpy  = weight_is_vrmpy;
      chunk_states[i].b_scale = b_scale;
      chunk_states[i].vtcm_weight_int4 = w4_cur;
      chunk_states[i].vtcm_weight      = w_cur;
      chunk_states[i].depend = &weight_desc[i];
      chunk_states[i].shared_sync = &sync_token[i];

      worker_pool_job_t job;
      job.fptr = process_q4_weight_worker_loop;
      job.dptr = &chunk_states[i];
      if (worker_pool_submit(NULL, job) != 0) {
        process_q4_weight_worker_loop(&chunk_states[i], 0);
      }
    }

    {
      unsigned long long _t = MM_PHASE_T0();
      dma_wait_for_idle();
      MM_PHASE_ADD(2, _t);
    }
#undef SET_WEIGHT_DMA

    for (int ox_start = 0; ox_start < mp; ox_start += mp_chunk) {
      int next_ox_start = ox_start + mp_chunk;
      int next_buf_idx = 1 - act_buf_idx;

      if (!reuse_activation) {
        unsigned long long _t = MM_PHASE_T0();
        dma_wait_for_idle();
        MM_PHASE_ADD(2, _t);
      }

      if (!reuse_activation && next_ox_start < mp) {
          ISSUE_ACT_DMA(next_ox_start, next_buf_idx);
      }

      int ox_end = ox_start + mp_chunk;
      if (ox_end > mp) ox_end = mp;

      int current_base_offset = act_buf_idx * mp_chunk * kp * 32 * 32;

      if (!reuse_activation) {
        unsigned long long _t = MM_PHASE_T0();
        shuffle_q4_activation_tiles(vtcm_activation, current_base_offset, ox_start, ox_end, kp, M);
        MM_PHASE_ADD(4, _t);
      }

      {
        for (int ox = ox_start; ox < ox_end; ++ox) {
          // Same ox always maps to the same buffer, so waiting on it also drains this
          // ox's store from the previous oy-chunk -- that is what keeps the pack-64
          // 128B c-slot shared by oy pairs straddling a chunk boundary race-free.
          const int ring_slot      = async_output_store ? ox % store_inflight : 0;
          const int output_buf_idx = cross_oy_output_store ? scale_buf_idx * store_inflight + ring_slot : ring_slot;
          if (output_store_active[output_buf_idx]) {
            unsigned long long _tw = MM_PHASE_T0();
            worker_pool_synctoken_wait(&output_store_sync[output_buf_idx]);
            MM_PHASE_ADD(3, _tw);
            output_store_active[output_buf_idx] = 0;
          }
          __fp16* current_vtcm_output = vtcm_output + (size_t)output_buf_idx * np_chunk * 1024;
          int     ox_offset           = current_base_offset + (ox - ox_start) * kp * 32 * 32;
          for (int chunk_idx = 0; chunk_idx < valid_chunks; ++chunk_idx) {
            if (!chunk_weight_ready[chunk_idx]) {
              unsigned long long _t = MM_PHASE_T0();
              worker_pool_synctoken_wait(&sync_token[chunk_idx]);
              MM_PHASE_ADD(1, _t);
              chunk_weight_ready[chunk_idx] = 1;
            }
            int oy_chunk_start = oy_start + chunk_starts[chunk_idx];
            int oy_chunk_end = oy_chunk_start + chunk_counts[chunk_idx];
            unsigned long long _th            = MM_PHASE_T0();
            for (int oy = oy_chunk_start; oy < oy_chunk_end; ++oy) {
              int oy_offset = (oy - oy_start) * 16 * kp;
              __fp16* output_tile = current_vtcm_output + (oy - oy_start) * 1024;
              hmx_load_q4_tiles(vtcm_activation + ox_offset, w_cur + oy_offset * 64, kp);
              hmx_consume_accumulator_fp16(output_tile);
            }
            MM_PHASE_ADD(0, _th);
          }
          // One store for the whole oy chunk, after all its HMX tiles are in VTCM. Storing
          // per weight-chunk instead (chunks are often a single oy) never hits the paired
          // path below, and a lone tile owns only half of a pack-64 128B c-slot, so it has
          // to read-modify-write DDR: 60.4ms vs 11.9ms for the same output on a 1053-token
          // prefill.
          const uint8_t *store_scales = dequant_in_weight ? NULL : current_vtcm_scales;
          if (async_output_store) {
            worker_pool_synctoken_init(&output_store_sync[output_buf_idx], 1);
            output_store_tasks[output_buf_idx].c           = c;
            output_store_tasks[output_buf_idx].vtcm_output = current_vtcm_output;
            output_store_tasks[output_buf_idx].bias        = bias;
            output_store_tasks[output_buf_idx].vtcm_scales = store_scales;
            output_store_tasks[output_buf_idx].M           = M;
            output_store_tasks[output_buf_idx].ox          = ox;
            output_store_tasks[output_buf_idx].oy_start    = oy_start;
            output_store_tasks[output_buf_idx].oy_begin    = oy_start;
            output_store_tasks[output_buf_idx].oy_end      = oy_end;
            output_store_tasks[output_buf_idx].pack        = pack;
            output_store_tasks[output_buf_idx].shared_sync = &output_store_sync[output_buf_idx];
            worker_pool_job_t job;
            job.fptr = store_q4_output_worker_loop;
            job.dptr = &output_store_tasks[output_buf_idx];
            if (worker_pool_submit(NULL, job) != 0) {
              store_q4_output_worker_loop(&output_store_tasks[output_buf_idx], 0);
            }
            output_store_active[output_buf_idx] = 1;
          } else {
            unsigned long long _ts = MM_PHASE_T0();
            store_q4_output_range(c, current_vtcm_output, bias, store_scales, M, ox, oy_start, oy_start, oy_end, pack);
            MM_PHASE_ADD(3, _ts);
          }
        }
      }
      act_buf_idx = next_buf_idx;
    }
    for (int i = 0; i < valid_chunks; ++i) {
      if (!chunk_weight_ready[i]) {
        worker_pool_synctoken_wait(&sync_token[i]);
      }
    }
#undef ISSUE_ACT_DMA
  }
  for (int i = 0; i < output_buffers; ++i) {
    if (output_store_active[i]) {
      worker_pool_synctoken_wait(&output_store_sync[i]);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  free(act_descs);
#if HTP_MM_PHASE_PROFILE
  g_mm_phase_us[6] = (unsigned long long) np_chunk;
  g_mm_phase_us[7] = (unsigned long long) ((np + np_chunk - 1) / np_chunk);
  MM_PHASE_ADD(5, _mm_tot);
#endif
  return 0;
}
