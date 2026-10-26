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
#include "htp_command.h"

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

#ifndef HTP_MM_PHASE_PROFILE
#  define HTP_MM_PHASE_PROFILE 0
#endif
#if HTP_MM_PHASE_PROFILE
#  include "HAP_perf.h"
// Slots 0..23 are the original phase/traffic set; 24..28 split the activation staging into its DMA and
// HVX halves and count descriptors, which is what decides whether a stage is bandwidth- or
// latency-bound; 29..31 are the per-call fixed costs (pre-loop submits, next-panel submits, allocation).
// Keep in lockstep with the loop in execute_command.cc and the names in dsp_op_name.h.
unsigned long long g_mm_phase_us[32] = { 0 };
#  define MM_T0()              HAP_perf_get_time_us()
#  define MM_ADD(index, start) (g_mm_phase_us[(index)] += HAP_perf_get_time_us() - (start))
// Slots 9..12 are summed across workers, so they need an atomic add.
#  define MM_WORKER_ADD(index, start) \
    __atomic_fetch_add(&g_mm_phase_us[(index)], HAP_perf_get_time_us() - (start), __ATOMIC_RELAXED)
#  define MM_BUMP(index, v) (g_mm_phase_us[(index)] += (unsigned long long) (v))
#else
#  define MM_BUMP(index, v)           ((void) 0)
#  define MM_T0()                     0ULL
#  define MM_ADD(index, start)        ((void) (start))
#  define MM_WORKER_ADD(index, start) ((void) (start))
#endif

// What the last command staged into the activation area at the base of VTCM. Plain globals: commands are
// executed one at a time by the queue thread, so there is no concurrent writer.
static struct {
  const uint8_t *base;
  int            m, k;
  unsigned long  cmd_seq;
} g_act_stage = { NULL, 0, 0, 0 };

static const __fp16 q4_to_fp16_lut[64] __attribute__((aligned(128))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

// Applies the block scale (and the offset, for asymmetric weights) to one LUT result and stores it.
#define DEQUANT_Q4_SCALE_STORE(idx, vsrc) dst_vec[(idx)] = Q6_Vhf_vmpy_VhfVhf((vsrc), vBlockScale)
// Folding this into the single Q6_Vhf_vmpyacc_VhfVhfVhf(vBlockOffset, vsrc, vBlockScale) was measured
// slower (2B pp512 1763 -> 1737), even though it is three instructions down to one: the accumulator
// operand is in-out, so vBlockOffset -- live across all 16 outputs of the tile -- has to be copied
// before every call, trading two ALU ops for a vmov and more register pressure.
#define DEQUANT_Q4_SCALE_OFFSET_STORE(idx, vsrc) \
  dst_vec[(idx)] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf((vsrc), vBlockScale), vBlockOffset))
// Expands the low- and high-nibble vectors of one packed source vector into four scaled halfword vectors.
#define DEQUANT_Q4_NIBBLE_PAIR(STORE, dst_idx, vnib_lo, vnib_hi)           \
  do {                                                                     \
    HVX_VectorPair _pl = Q6_Wh_vlut16_VbVhR_nomatch(vnib_lo, vlut_cvt, 0); \
    HVX_VectorPair _ph = Q6_Wh_vlut16_VbVhR_nomatch(vnib_hi, vlut_cvt, 0); \
    STORE((dst_idx) + 0, Q6_V_lo_W(_pl));                                  \
    STORE((dst_idx) + 1, Q6_V_hi_W(_pl));                                  \
    STORE((dst_idx) + 2, Q6_V_lo_W(_ph));                                  \
    STORE((dst_idx) + 3, Q6_V_hi_W(_ph));                                  \
  } while (0)

// Same, for two source vectors at once: the four LUT lookups are issued before any of the eight
// scale-and-store chains, so each chain's latency is covered by the other group's work.
#define DEQUANT_Q4_NIBBLE_PAIR2(STORE, dst_idx, nlo0, nhi0, nlo1, nhi1) \
  do {                                                                  \
    HVX_VectorPair _a0 = Q6_Wh_vlut16_VbVhR_nomatch(nlo0, vlut_cvt, 0); \
    HVX_VectorPair _b0 = Q6_Wh_vlut16_VbVhR_nomatch(nhi0, vlut_cvt, 0); \
    HVX_VectorPair _a1 = Q6_Wh_vlut16_VbVhR_nomatch(nlo1, vlut_cvt, 0); \
    HVX_VectorPair _b1 = Q6_Wh_vlut16_VbVhR_nomatch(nhi1, vlut_cvt, 0); \
    STORE((dst_idx) + 0, Q6_V_lo_W(_a0));                               \
    STORE((dst_idx) + 4, Q6_V_lo_W(_a1));                               \
    STORE((dst_idx) + 1, Q6_V_hi_W(_a0));                               \
    STORE((dst_idx) + 5, Q6_V_hi_W(_a1));                               \
    STORE((dst_idx) + 2, Q6_V_lo_W(_b0));                               \
    STORE((dst_idx) + 6, Q6_V_lo_W(_b1));                               \
    STORE((dst_idx) + 3, Q6_V_hi_W(_b0));                               \
    STORE((dst_idx) + 7, Q6_V_hi_W(_b1));                               \
  } while (0)

// Path A dequantization: one vrmpy-layout int4 tile (512B) straight to scaled fp16 in HMX layout. In the vrmpy
// layout byte
// g*64 + 2*oc + p holds w(oc, 4g+2p) in its low nibble and w(oc, 4g+2p+1) in its high one, so the two k-pairs
// (4g, 4g+1) and (4g+2, 4g+3) of one oc sit in one halfword. Interleaving the source with its >> 4 copy by
// halfwords puts, for every oc, [pair0 lo, pair1 lo, pair0 hi, pair1 hi] in four consecutive bytes, and
// vlut16 splits even and odd bytes into its two outputs -- which are then exactly the HMX k-pair vectors
// (lane 2*oc = even k, 2*oc + 1 = odd k). That is four instructions per source vector; reordering into the
// HMX int4 layout first (vdeal, vshuff, two vror, two vmux, two more vshuff) took twelve, nearly all permutes
// competing with vlut16 for the one permute slot. v81: 62 -> 38 ns a tile symmetric, 76 -> 56 asymmetric.
static inline void dequant_q4_tile_scaled_from_vrmpy(const uint8_t *src, __fp16 *dst, HVX_Vector vlut_cvt,
                                                     HVX_Vector vBlockScale, HVX_Vector vBlockOffset,
                                                     int scale_asymmetric) {
  const HVX_Vector *src_vec = (const HVX_Vector *) src;
  HVX_Vector       *dst_vec = (HVX_Vector *) dst;
  for (int i = 0; i < 4; i += 2) {
    const HVX_Vector     v0 = src_vec[i];
    const HVX_Vector     v1 = src_vec[i + 1];
    const HVX_VectorPair w0 = Q6_W_vshuff_VVR(Q6_Vub_vlsr_VubR(v0, 4), v0, -2);
    const HVX_VectorPair w1 = Q6_W_vshuff_VVR(Q6_Vub_vlsr_VubR(v1, 4), v1, -2);
    if (scale_asymmetric) {
      DEQUANT_Q4_NIBBLE_PAIR2(DEQUANT_Q4_SCALE_OFFSET_STORE, i * 4, Q6_V_lo_W(w0), Q6_V_hi_W(w0), Q6_V_lo_W(w1),
                              Q6_V_hi_W(w1));
    } else {
      DEQUANT_Q4_NIBBLE_PAIR2(DEQUANT_Q4_SCALE_STORE, i * 4, Q6_V_lo_W(w0), Q6_V_hi_W(w0), Q6_V_lo_W(w1),
                              Q6_V_hi_W(w1));
    }
  }
}

// Tiles [k0, k1) of one oy in the vrmpy block layout, block by block: the scale repack runs once per block and
// the tiles of a block go back to back. Finding the block per tile was a 64-bit division -- a library call on
// Hexagon -- for every 38 ns tile. kp is a multiple of scale_block_num (the host checks).
static inline void dequant_q4_vrmpy_oy(const float *sw_oy, const uint8_t *src_oy, __fp16 *dst_oy, int kp, int k0,
                                       int k1, int scale_block_num, int scale_asymmetric, HVX_Vector vlut_cvt) {
  const int      k_per = kp / scale_block_num;
  const uint8_t *src   = src_oy + (size_t) k0 * 512;
  __fp16        *dst   = dst_oy + (size_t) k0 * 1024;
  for (int blk = k0 / k_per, k = k0; k < k1; ++blk) {
    const HVX_Vector vScale = vrmpy_scale_block_dup_hf(sw_oy, blk, scale_asymmetric);
    const HVX_Vector vQBias = scale_asymmetric ? vrmpy_qbias_block_dup_hf(sw_oy, blk, scale_asymmetric) : Q6_V_vzero();
    const int        k_end  = (blk + 1) * k_per < k1 ? (blk + 1) * k_per : k1;
    for (; k < k_end; ++k) {
      dequant_q4_tile_scaled_from_vrmpy(src, dst, vlut_cvt, vScale, vQBias, scale_asymmetric);
      src += 512;
      dst += 1024;
    }
  }
}

static inline void dequant_q4_tile_scaled(const uint8_t *src, __fp16 *dst, HVX_Vector vlut_cvt, HVX_Vector vBlockScale,
                                          HVX_Vector vBlockOffset, int scale_asymmetric) {
  const HVX_Vector* src_vec = (const HVX_Vector*)src;
  HVX_Vector* dst_vec = (HVX_Vector*)dst;
  // Scale and offset are applied in one pass: storing the scaled value first and adding the offset in a
  // second loop costs 16 extra VTCM stores plus 16 reloads per tile, which triples the VTCM traffic of a
  // stage that is already the op's critical path (HQQ weights are always asymmetric, so that was the
  // common case).
  //
  // The tile is expanded one input vector at a time rather than all four up front. All four means eight
  // live HVX_VectorPairs -- 16 of the 32 vector registers -- which leaves the compiler no room to software
  // pipeline, so the vmpy -> vadd -> cvt dependency chain (about 4 cycles per level) stays exposed. Going
  // from four to one cut this stage from 258 to 239 ms on a 512-token prefill.
  //
  // Computing nibble - 8 arithmetically (vzxt + Q6_Vhf_vcvt_Vuh, with the -8 folded into the offset) to
  // avoid the LUT was tried and is 18% slower -- the convert chain costs more than vlut16 does.
  // The LUT indexes on the low 4 bits, so the packed vector itself is the low-nibble vector.
#define DEQUANT_Q4_GROUP(STORE, src_idx)                  \
  do {                                                    \
    HVX_Vector _q = src_vec[(src_idx)];                   \
    HVX_Vector _h = Q6_Vub_vlsr_VubR(_q, 4);              \
    DEQUANT_Q4_NIBBLE_PAIR(STORE, (src_idx) * 4, _q, _h); \
  } while (0)
  if (scale_asymmetric) {
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_OFFSET_STORE, 0);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_OFFSET_STORE, 1);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_OFFSET_STORE, 2);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_OFFSET_STORE, 3);
  } else {
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_STORE, 0);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_STORE, 1);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_STORE, 2);
    DEQUANT_Q4_GROUP(DEQUANT_Q4_SCALE_STORE, 3);
  }
#undef DEQUANT_Q4_GROUP
}

// These three expand to expressions referring to the caller's dst_vec / vBlockScale / vBlockOffset /
// vlut_cvt, so they are only meaningful inside the two functions above.
#undef DEQUANT_Q4_NIBBLE_PAIR
#undef DEQUANT_Q4_NIBBLE_PAIR2
#undef DEQUANT_Q4_SCALE_OFFSET_STORE
#undef DEQUANT_Q4_SCALE_STORE

static inline void process_q4_weight_lut(int oy_start, int start_idx, int count, int kp,
                                         const uint8_t *__restrict vtcm_weight_int4, __fp16 *__restrict vtcm_weight,
                                         const uint8_t *b_scale, int scale_block_num, int scale_asymmetric,
                                         int weight_is_vrmpy) {
  HVX_Vector vlut_cvt = vmemu(q4_to_fp16_lut);
  const int weight_int4_stride = 512 * kp;
  const int weight_fp16_stride = 1024 * kp;
  const uint8_t* src_oy = vtcm_weight_int4 + start_idx * weight_int4_stride;
  __fp16* dst_oy = vtcm_weight + start_idx * weight_fp16_stride;
  // Path A: the DMA'd tiles are in vrmpy layout and b_scale points at fp32 vrmpy block scales
  // (sw[(oy*nblk+b)*entry_floats+ocIn]). Asymmetric entries are twice as wide, carrying 32 fp32
  // qbias after their 32 fp32 scale, so the same single scale DMA per oc-tile covers both.
  if (weight_is_vrmpy && scale_block_num > 1) {
    const float *sw           = (const float *) b_scale;
    const int    entry_floats = VRMPY_SCALE_ENTRY_FLOATS(scale_asymmetric);
    for (int local_oy = 0; local_oy < count; ++local_oy) {
      const int      oy    = oy_start + start_idx + local_oy;
      dequant_q4_vrmpy_oy(sw + (size_t) oy * scale_block_num * entry_floats, src_oy, dst_oy, kp, 0, kp, scale_block_num,
                          scale_asymmetric, vlut_cvt);
      src_oy += weight_int4_stride;
      dst_oy += weight_fp16_stride;
    }
    return;
  }
  if (scale_block_num > 1) {
    const int scale_block_bytes = scale_asymmetric ? 256 : 128;
    for (int local_oy = 0; local_oy < count; ++local_oy) {
      const int oy = oy_start + start_idx + local_oy;
      const uint8_t* src = src_oy;
      __fp16* dst = dst_oy;
      const uint8_t* block_scale_ptr = b_scale + (size_t)oy * scale_block_num * scale_block_bytes;
      if (scale_block_num == kp) {
        for (int k = 0; k < kp; ++k) {
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vmemu(block_scale_ptr),
                                 scale_asymmetric ? vmemu(block_scale_ptr + 128) : Q6_V_vzero(), scale_asymmetric);
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
          const uint8_t *current_scale_ptr = block_scale_ptr + scale_idx * scale_block_bytes;
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vBlockScale,
                                 scale_asymmetric ? vmemu(current_scale_ptr + 128) : Q6_V_vzero(), scale_asymmetric);
          src += 512;
          dst += 1024;
        }
      } else {
        for (int k = 0; k < kp; ++k) {
          const int scale_idx = (k * scale_block_num) / kp;
          const uint8_t *current_scale_ptr = block_scale_ptr + scale_idx * scale_block_bytes;
          dequant_q4_tile_scaled(src, dst, vlut_cvt, vmemu(current_scale_ptr),
                                 scale_asymmetric ? vmemu(current_scale_ptr + 128) : Q6_V_vzero(), scale_asymmetric);
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
  int k_begin, k_end;  // k-tile range of the chunk's oy (the whole [0, kp) unless the panel was split along k)
  int scale_block_num;
  int                  scale_asymmetric;
  int                  weight_is_vrmpy;
  const uint8_t* b_scale;
  const uint8_t* vtcm_weight_int4;
  __fp16* vtcm_weight;
  dma_desc_1d_t       *depend;
  int own_dma;  // the job starts its own chunk's DMA on this worker's engine instead of waiting on a chain
  worker_synctoken_t* shared_sync;
} process_q4_weight_task_state_t;

#define Q4_MAX_WEIGHT_CHUNKS 16
// Upper bound on the asynchronous output slots, which the kernel sizes down to mp and to whatever VTCM
// is left over. 16 covers M = 512 at one slot per 32-row tile.
#define Q4_MAX_OUTPUT_SLOTS  16
// Row-tile groups of the asynchronously staged activation (see hmx_matmulq4fp16).
#define Q4_ACT_GROUPS        16
// Waits for the activation group holding row tile ox, the first time any panel needs it.
#define Q4_WAIT_ACT_GROUP(ox)                                     \
  do {                                                            \
    const int _g = (ox) == 0 ? 0 : 1 + ((ox) - 1) / act_group_ox; \
    if (act_pending[_g]) {                                        \
      const unsigned long long _t = MM_T0();                      \
      worker_pool_synctoken_wait(&act_sync[_g]);                  \
      MM_ADD(2, _t);                                              \
      act_pending[_g] = 0;                                        \
    }                                                             \
  } while (0)

static void process_q4_weight_worker_loop(void *data, int worker_index);

// One panel's worth of weight staging: its DMA chain, its dequantization jobs and which of them this thread has
// already waited for. Two of these let the next panel's DMA + dequantization run while HMX consumes the current
// one, which is otherwise impossible because they share the single fp16 weight buffer.
typedef struct {
  int oy_start, oy_end, count, valid_chunks;
  int chunk_starts[Q4_MAX_WEIGHT_CHUNKS];
  int chunk_counts[Q4_MAX_WEIGHT_CHUNKS];
  int chunk_ready[Q4_MAX_WEIGHT_CHUNKS];
  // k-tile range of each chunk. A chunk that ends before kp is one part of an oy split along k, and the oy's
  // HMX waits for its last part (the parts of one oy are consecutive chunks).
  int chunk_k0[Q4_MAX_WEIGHT_CHUNKS];
  int chunk_k1[Q4_MAX_WEIGHT_CHUNKS];
  _Alignas(64) dma_desc_1d_t weight_desc[Q4_MAX_WEIGHT_CHUNKS];
  worker_synctoken_t             sync[Q4_MAX_WEIGHT_CHUNKS];
  process_q4_weight_task_state_t states[Q4_MAX_WEIGHT_CHUNKS];
  int                            issued;
  const struct q4_weight_ctx    *drv_ctx;  // what the driver job (q4_panel_issue_job) fills the panel from
  int                            drv_buf;
} q4_panel_state_t;

typedef struct q4_weight_ctx {
  const uint8_t *b;
  const uint8_t *b_scale;
  uint8_t       *vtcm_weight_int4;  // buffer 0; buffer i is + i * int4_stride
  __fp16        *vtcm_weight;
  size_t         int4_stride, fp16_stride;
  int            kp, scale_block_num, scale_asymmetric, weight_is_vrmpy, max_chunks;
  int            mp;  // row tiles each panel's weights are multiplied with
} q4_weight_ctx_t;

// Weight descriptors and job states of the panel's chunks (layout already chosen). own_dma: each job starts its
// own descriptor on its worker's engine; otherwise the descriptors are chained for this thread's engine.
static void q4_panel_fill(const q4_weight_ctx_t *ctx, q4_panel_state_t *panel, int buf, int own_dma) {
  uint8_t *int4_base = ctx->vtcm_weight_int4 + (size_t) buf * ctx->int4_stride;
  __fp16  *fp16_base = (__fp16 *) ((uint8_t *) ctx->vtcm_weight + (size_t) buf * ctx->fp16_stride);
  for (int i = 0; i < panel->valid_chunks; ++i) {
    dma_desc_1d_t *d = &panel->weight_desc[i];
    memset(d, 0, sizeof(*d));
    const size_t k_off = (size_t) panel->chunk_k0[i] * 512;
    d->length          = 512 * (panel->chunk_k1[i] - panel->chunk_k0[i]) * panel->chunk_counts[i];
    d->type            = DMA_DESC_TYPE_1D;
    d->src_bypass      = 1;
    d->dst_bypass      = 1;
    d->ordered         = 1;
    d->dstate          = DMA_DESC_DSTATE_PENDING;
    d->src  = (uint32_t) (ctx->b + (size_t) (panel->oy_start + panel->chunk_starts[i]) * ctx->kp * 32 * 16 + k_off);
    d->dst  = (uint32_t) (int4_base + (size_t) panel->chunk_starts[i] * 512 * ctx->kp + k_off);
    d->next = (!own_dma && i + 1 < panel->valid_chunks) ? (uint32_t) &panel->weight_desc[i + 1] : 0;
    process_q4_weight_task_state_t *st = &panel->states[i];
    st->oy_start                       = panel->oy_start;
    st->start_idx                      = panel->chunk_starts[i];
    st->count                          = panel->chunk_counts[i];
    st->kp                             = ctx->kp;
    st->k_begin                        = panel->chunk_k0[i];
    st->k_end                          = panel->chunk_k1[i];
    st->scale_block_num                = ctx->scale_block_num;
    st->scale_asymmetric               = ctx->scale_asymmetric;
    st->weight_is_vrmpy                = ctx->weight_is_vrmpy;
    st->b_scale                        = ctx->b_scale;
    st->vtcm_weight_int4               = int4_base;
    st->vtcm_weight                    = fp16_base;
    st->depend                         = d;
    st->own_dma                        = own_dma;
    st->shared_sync                    = &panel->sync[i];
  }
}

// Driver job of an own-DMA panel: fills it, queues chunks 1.. for the other workers and dequantizes chunk 0
// itself.
static void q4_panel_issue_job(void *data, int worker_index) {
  q4_panel_state_t *panel = (q4_panel_state_t *) data;
  q4_panel_fill(panel->drv_ctx, panel, panel->drv_buf, 1);
  worker_pool_job_t jobs[Q4_MAX_WEIGHT_CHUNKS];
  for (int i = 1; i < panel->valid_chunks; ++i) {
    jobs[i].fptr = process_q4_weight_worker_loop;
    jobs[i].dptr = &panel->states[i];
  }
  if (panel->valid_chunks > 1 && worker_pool_enqueue_batch(NULL, jobs + 1, (unsigned) panel->valid_chunks - 1) != 0) {
    for (int i = 1; i < panel->valid_chunks; ++i) {
      process_q4_weight_worker_loop(&panel->states[i], worker_index);
    }
  }
  process_q4_weight_worker_loop(&panel->states[0], worker_index);
}

// Chooses the panel's chunk layout, then has its weights DMA'd and dequantized by the workers.
// Each job spins on its own descriptor, so the chain does not have to be complete before they are submitted.
static void q4_panel_submit(const q4_weight_ctx_t *ctx, q4_panel_state_t *panel, int oy_start, int oy_end, int buf,
                            uint32_t tail_next) {
  panel->oy_start = oy_start;
  panel->oy_end   = oy_end;
  panel->count    = oy_end - oy_start;
  panel->issued   = 1;

  int num_chunks = ctx->max_chunks;
  if (num_chunks > panel->count) {
    num_chunks = panel->count;
  }
  if (num_chunks < 1) {
    num_chunks = 1;
  }
  // Rounding the chunk size up and aligning it to even deliberately produces fewer, larger chunks: with
  // count=36 over 8 workers this is 6 chunks of 6 rather than 5,5,5,5,4,4,4,4. Spreading the remainder to
  // get all 8 workers busy was measured and is worse (101.4 -> 104.4 ms): the chunk count also splits the
  // weight DMA chain, so 8 chunks means more per-transfer latency and two more workers spinning on
  // descriptors, which steals memory ports from the main thread (MMW_DMA_SPIN 22 -> 35 ms,
  // MM_STORE_INLINE 19.7 -> 24.4 ms).
  int chunk_size = (panel->count + num_chunks - 1) / num_chunks;
  if (chunk_size > 1) {
    chunk_size = (chunk_size + 1) & ~1;
  }
  // A panel with fewer oy than workers leaves the rest idle while HMX waits for it: on the windowed K = 9216
  // GEMMs a panel is two oy of 288 tiles, 28 us each on one worker against 11 us of HMX. Split each oy along
  // k so every worker gets a part. Only where each job moves its own chunk (no chain to keep in order) and on
  // the vrmpy block layout, whose dequantization can start at any k.
  // Split only where one oy's dequantization (~97 ns a tile) clearly outlasts the panel's HMX (~2.4 ns a tile
  // per row tile and oy): mp * count < 20. Splitting every small panel made Qwen3.5-2B 3.7 ms slower (16% more
  // jobs, no dequantization wait to win back); at < 40 it also catches 2B's ffn_down, for no gain. Below K = 4096
  // an oy dequantizes in under 12 us and the extra jobs are not worth it either.
  int k_parts = 1;
  if (tail_next == 0 && ctx->scale_block_num > 1 && ctx->weight_is_vrmpy && panel->count < ctx->max_chunks &&
      ctx->mp * panel->count < 20 && ctx->kp >= 128) {
    k_parts = ctx->max_chunks / panel->count;
    if (k_parts > ctx->kp / 16) {
      k_parts = ctx->kp / 16;  // keep parts of at least 16 tiles
    }
    if (k_parts < 1) {
      k_parts = 1;
    }
  }
  panel->valid_chunks = 0;
  if (k_parts > 1) {
    for (int oy = 0; oy < panel->count; ++oy) {
      for (int part = 0; part < k_parts; ++part) {
        const int i            = panel->valid_chunks++;
        panel->chunk_starts[i] = oy;
        panel->chunk_counts[i] = 1;
        panel->chunk_ready[i]  = 0;
        panel->chunk_k0[i]     = ctx->kp * part / k_parts;
        panel->chunk_k1[i]     = ctx->kp * (part + 1) / k_parts;
      }
    }
  } else {
    for (int i = 0, start = 0; i < num_chunks && start < panel->count; ++i) {
      int end = start + chunk_size;
      if (end > panel->count) {
        end = panel->count;
      }
      panel->chunk_starts[i] = start;
      panel->chunk_counts[i] = end - start;
      panel->chunk_ready[i]  = 0;
      panel->chunk_k0[i]     = 0;
      panel->chunk_k1[i]     = ctx->kp;
      ++panel->valid_chunks;
      start = end;
    }
  }

  MM_BUMP(27, panel->valid_chunks);
  for (int i = 0; i < panel->valid_chunks; ++i) {
    worker_pool_synctoken_init(&panel->sync[i], 1);
  }
  // With nothing chained behind the weights (the block path), every job moves its own chunk on its own
  // worker's DMA engine. One chain on this thread's engine made chunk i wait behind chunks 0..i-1 with its
  // worker spinning the whole time; separate engines move them concurrently and the worker sleeps in dmwait
  // instead of spinning.
  // Filling the descriptors and job states and queueing the jobs is then left to one driver job: done here it
  // cost this thread ~0.5 us a job, and this thread is the one feeding HMX, so it was HMX time -- 21 ms per
  // Qwen3.5-4B prefill, which hands out ~39K jobs. The chunk layout and the sync tokens stay on this thread,
  // since it reads them while the driver may still be running.
  // Only once this panel's HMX (about 2.4 ns a tile per row tile) outlasts the handoff: the driver job is
  // picked up and fills the panel before the first chunk's dequantization starts, which at mp = 4 (pp128,
  // ~11 us of HMX a panel) cost more than the submits it saves -- 2B and 4B pp128 both -1.3%.
  if (tail_next == 0 && ctx->scale_block_num > 1 && ctx->mp >= 8) {
    panel->drv_ctx = ctx;
    panel->drv_buf = buf;
    worker_pool_job_t job;
    job.fptr = q4_panel_issue_job;
    job.dptr = panel;
    if (worker_pool_submit(NULL, job) != 0) {
      q4_panel_issue_job(panel, 0);
    }
    return;
  }
  // Same choice of DMA owner as the driver path above; only the submitting thread differs. Chaining a
  // block-path panel onto this thread's engine instead deadlocks: the engine is usually busy with the
  // activation staging, so the chain never starts and every worker spins on its descriptor's done bit
  // (v79 Qwen3.5-2B pp128 hung every time; v81 happened to find the engine idle).
  const int own_dma = tail_next == 0 && ctx->scale_block_num > 1;
  q4_panel_fill(ctx, panel, buf, own_dma);
  if (!own_dma) {
    panel->weight_desc[panel->valid_chunks - 1].next = tail_next;
    dmstart(&panel->weight_desc[0]);
  }
  for (int i = 0; i < panel->valid_chunks; ++i) {
    worker_pool_job_t job;
    job.fptr = process_q4_weight_worker_loop;
    job.dptr = &panel->states[i];
    if (worker_pool_submit(NULL, job) != 0) {
      process_q4_weight_worker_loop(&panel->states[i], 0);
    }
  }
}

static void q4_panel_drain(q4_panel_state_t *panel) {
  if (!panel->issued) {
    return;
  }
  for (int i = 0; i < panel->valid_chunks; ++i) {
    if (!panel->chunk_ready[i]) {
      worker_pool_synctoken_wait(&panel->sync[i]);
      panel->chunk_ready[i] = 1;
    }
  }
  panel->issued = 0;
}

static inline void wait_q4_weight_dma_done(const dma_desc_1d_t* depend) {
  volatile uint32_t *ctrl_word = (volatile uint32_t*)&depend->dstate_order_bypass_type_length;
  while (((*ctrl_word >> 31) & 0x1) == 0) {
    asm volatile("nop");
  }
}

static void process_q4_weight_worker_loop(void *data, int _worker_index) {
  (void)_worker_index;
  process_q4_weight_task_state_t* state = (process_q4_weight_task_state_t*)data;
  // The block scales are read straight from DDR, a 128 B line every two tiles, and each of those loads is
  // on the dequantization's dependency chain. Cold, they made it 320 ns/tile against 97 ns/tile with the
  // range already in L2 (v81 microbenchmark, HTP_MM_MICROBENCH): the stage was waiting on DDR latency, not
  // computing. The chunk's scales are one contiguous range, so one l2fetch covers it, and issuing it before
  // the weight DMA wait overlaps the two.
  if (state->scale_block_num > 1) {
    const size_t per_oy =
      (size_t) state->scale_block_num * (state->weight_is_vrmpy ? 128 : (state->scale_asymmetric ? 256 : 128));
    const size_t bytes = per_oy * state->count;
    l2fetch(state->b_scale + (size_t) (state->oy_start + state->start_idx) * per_oy, 4096, 4096,
            (uint32_t) ((bytes + 4095) / 4096), 0);
  }
  const unsigned long long t_dma = MM_T0();
  if (state->own_dma) {
    dma_wait_for_idle();
    dmstart(state->depend);
    dma_wait_for_idle();
  } else {
    wait_q4_weight_dma_done(state->depend);
  }
  MM_WORKER_ADD(12, t_dma);
  const unsigned long long t = MM_T0();
  if (state->k_end - state->k_begin != state->kp) {
    // One k part of an oy that q4_panel_submit split along k; always the vrmpy block layout.
    const int oy = state->oy_start + state->start_idx;
    dequant_q4_vrmpy_oy((const float *) state->b_scale +
                          (size_t) oy * state->scale_block_num * VRMPY_SCALE_ENTRY_FLOATS(state->scale_asymmetric),
                        state->vtcm_weight_int4 + (size_t) state->start_idx * 512 * state->kp,
                        state->vtcm_weight + (size_t) state->start_idx * 1024 * state->kp, state->kp, state->k_begin,
                        state->k_end, state->scale_block_num, state->scale_asymmetric, vmemu(q4_to_fp16_lut));
  } else {
    process_q4_weight_lut(state->oy_start, state->start_idx, state->count, state->kp, state->vtcm_weight_int4,
                          state->vtcm_weight, state->b_scale, state->scale_block_num, state->scale_asymmetric,
                          state->weight_is_vrmpy);
  }
  MM_WORKER_ADD(9, t);
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

// Every source address here is a + ((k>>1)*M + ox*32)*128 + r*128, so all of them are 128-byte aligned as
// long as the activation itself is -- which it is, since the host hands over DMA-able tensor memory. vmemu
// spends an extra slot assembling from two aligned reads, and this stage moves 392 MiB per 512-token
// prefill, so it is worth branching once per row tile.
#define COPY_SHUFFLE_Q4_LOAD_ALIGNED(p)   (*(const HVX_Vector *) (p))
#define COPY_SHUFFLE_Q4_LOAD_UNALIGNED(p) vmemu((const HVX_Vector *) (p))
#define COPY_SHUFFLE_Q4_K_LOOP(LOAD)                                                                                \
  for (int k = 0; k < kp; k += 2) {                                                                                 \
    const int      has_high_tile = (k + 1) < kp;                                                                    \
    const uint8_t *src_base      = a + (size_t) (((k >> 1) * ms + ox * 32) * 64) * sizeof(int16_t);                 \
    __fp16        *tile0         = vtcm_activation + ox_offset + k * 32 * 32;                                       \
    __fp16        *tile1         = tile0 + 32 * 32;                                                                 \
    int            r             = 0;                                                                               \
    for (; r <= valid_rows - 2; r += 2) {                                                                           \
      HVX_Vector     v0                                 = LOAD(src_base + (size_t) r * 64 * sizeof(int16_t));       \
      HVX_Vector     v1                                 = LOAD(src_base + (size_t) (r + 1) * 64 * sizeof(int16_t)); \
      HVX_VectorPair vp                                 = Q6_W_vdeal_VVR(v1, v0, 64);                               \
      vmem((HVX_Vector *) ((uint8_t *) tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));                           \
      if (has_high_tile) {                                                                                          \
        vmem((HVX_Vector *) ((uint8_t *) tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));                         \
      }                                                                                                             \
    }                                                                                                               \
    if (r < valid_rows) {                                                                                           \
      HVX_Vector     v0                                 = LOAD(src_base + (size_t) r * 64 * sizeof(int16_t));       \
      HVX_VectorPair vp                                 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);                     \
      vmem((HVX_Vector *) ((uint8_t *) tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));                           \
      if (has_high_tile) {                                                                                          \
        vmem((HVX_Vector *) ((uint8_t *) tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));                         \
      }                                                                                                             \
    }                                                                                                               \
  }

static inline void copy_shuffle_q4_activation_tiles_pack64(__fp16 *vtcm_activation, const uint8_t *a, int base_offset,
                                                           int ox_start, int ox_end, int kp, int M, int ms) {
  const int src_aligned = (((uintptr_t) a) & 127) == 0;
  for (int ox = ox_start; ox < ox_end; ++ox) {
    const int ox_offset = base_offset + (ox - ox_start) * kp * 32 * 32;
    int valid_rows = M - ox * 32;
    if (valid_rows > 32) valid_rows = 32;
    if (valid_rows <= 0) continue;
    if (src_aligned) {
      COPY_SHUFFLE_Q4_K_LOOP(COPY_SHUFFLE_Q4_LOAD_ALIGNED)
    } else {
      COPY_SHUFFLE_Q4_K_LOOP(COPY_SHUFFLE_Q4_LOAD_UNALIGNED)
    }
  }
#undef COPY_SHUFFLE_Q4_K_LOOP
#undef COPY_SHUFFLE_Q4_LOAD_UNALIGNED
#undef COPY_SHUFFLE_Q4_LOAD_ALIGNED
}

typedef struct {
  __fp16* vtcm_activation;
  const uint8_t* a;
  int base_offset;
  int ox_start;
  int ox_end;
  int kp;
  int M;
  int                 ms;  // row stride of the pack-64 activation: M of the whole tensor when this call is a row window
  // DMA-fed staging: when raw is set, each worker owns raw + worker_index * 2 * raw_packs * 4 KB in VTCM.
  uint8_t            *raw;
  int                 raw_packs;
  int                 p_begin, p_end;  // pack range of the DMA-fed staging (raw != NULL)
  worker_synctoken_t* shared_sync;
} copy_shuffle_q4_activation_task_state_t;

// Per-worker descriptors for the DMA-fed activation staging, two blocks of up to Q4_ACT_DMA_PACKS each.
#define Q4_ACT_DMA_PACKS 8
static _Alignas(64) dma_desc_1d_t g_q4_act_desc[Q4_MAX_WEIGHT_CHUNKS][2][Q4_ACT_DMA_PACKS];

// Stages row tiles [ox_start, ox_end) by first moving blocks of (row tiles x packs) DDR -> VTCM on this worker's
// own DMA engine and shuffling them from there. Shuffling straight from DDR made every 128 B load a DRAM round
// trip: 5.8 GB/s per thread, and the stage the first panel waits on. The next block's DMA runs under the
// current block's shuffle; the engine is idle-checked before every start because a start on a busy
// engine faults the DSP.
// A block takes as many row tiles as the landing buffer holds (up to raw_packs of 4 KB) and as many packs as
// fit beside them: the row tiles of one pack are contiguous in DDR, so each pack is one descriptor of
// rows * 128 B however many tiles it spans. One descriptor per (tile, pack) instead cost ~0.75 us per 4 KB
// on every engine, which is what bounded the staging.
static void copy_shuffle_q4_activation_dma(const copy_shuffle_q4_activation_task_state_t *st, int worker_index) {
  const int pb                           = st->p_begin;
  const int packs                        = st->p_end;  // one past the last pack of this task
  const int ox_count                     = st->ox_end - st->ox_start;
  const int nox                          = ox_count < st->raw_packs ? ox_count : st->raw_packs;
  const int ppb                          = st->raw_packs / nox;
  const int npb                          = (packs - pb + ppb - 1) / ppb;
  const int steps                        = (ox_count + nox - 1) / nox * npb;
  uint8_t  *raw                          = st->raw + (size_t) worker_index * 2 * st->raw_packs * 4096;
  dma_desc_1d_t(*desc)[Q4_ACT_DMA_PACKS] = g_q4_act_desc[worker_index];
#define Q4_ACT_STEP_GEOM(step, ox0, oxn, rows, p0, np)                        \
  const int ox0  = st->ox_start + ((step) / npb) * nox;                       \
  const int oxn  = st->ox_end - ox0 < nox ? st->ox_end - ox0 : nox;           \
  const int rows = st->M - ox0 * 32 < oxn * 32 ? st->M - ox0 * 32 : oxn * 32; \
  const int p0   = pb + ((step) % npb) * ppb;                                 \
  const int np   = packs - p0 < ppb ? packs - p0 : ppb
#define Q4_ACT_STEP_START(step)                                                                         \
  do {                                                                                                  \
    Q4_ACT_STEP_GEOM(step, ox0_, oxn_, rows_, p0_, np_);                                                \
    (void) oxn_;                                                                                        \
    dma_desc_1d_t *d_   = desc[(step) & 1];                                                             \
    uint8_t       *buf_ = raw + (size_t) ((step) & 1) * st->raw_packs * 4096;                           \
    for (int q = 0; q < np_; ++q) {                                                                     \
      memset(&d_[q], 0, sizeof(d_[q]));                                                                 \
      d_[q].length     = (unsigned) rows_ * 128;                                                        \
      d_[q].type       = DMA_DESC_TYPE_1D;                                                              \
      d_[q].src_bypass = 0;                                                                             \
      d_[q].dst_bypass = 1;                                                                             \
      d_[q].ordered    = 0;                                                                             \
      d_[q].dstate     = DMA_DESC_DSTATE_PENDING;                                                       \
      d_[q].src        = (uint32_t) (st->a + ((size_t) (p0_ + q) * st->ms + (size_t) ox0_ * 32) * 128); \
      d_[q].dst        = (uint32_t) (buf_ + (size_t) q * rows_ * 128);                                  \
      d_[q].next       = q + 1 < np_ ? (uint32_t) &d_[q + 1] : 0;                                       \
    }                                                                                                   \
    dma_wait_for_idle();                                                                                \
    dmstart(&d_[0]);                                                                                    \
  } while (0)
  Q4_ACT_STEP_START(0);
  for (int step = 0; step < steps; ++step) {
    dma_wait_for_idle();
    if (step + 1 < steps) {
      Q4_ACT_STEP_START(step + 1);
    }
    Q4_ACT_STEP_GEOM(step, ox0, oxn, rows, p0, np);
    const uint8_t *buf = raw + (size_t) (step & 1) * st->raw_packs * 4096;
    // The block is laid out like a rows-row activation: pack q at q * rows * 128, tile j at row 32 * j.
    for (int j = 0; j < oxn; ++j) {
      const int vr = rows - j * 32 < 32 ? rows - j * 32 : 32;
      copy_shuffle_q4_activation_tiles_pack64(st->vtcm_activation + (size_t) p0 * 2 * 1024, buf + (size_t) j * 32 * 128,
                                              st->base_offset + (ox0 + j - st->ox_start) * st->kp * 32 * 32, 0, 1,
                                              2 * np, vr, rows);
    }
  }
#undef Q4_ACT_STEP_START
#undef Q4_ACT_STEP_GEOM
}

static void copy_shuffle_q4_activation_worker_loop(void *data, int _worker_index) {
  copy_shuffle_q4_activation_task_state_t* state = (copy_shuffle_q4_activation_task_state_t*)data;
  const unsigned long long                 t     = MM_T0();
  if (state->raw != NULL) {
    copy_shuffle_q4_activation_dma(state, _worker_index);
  } else {
    copy_shuffle_q4_activation_tiles_pack64(state->vtcm_activation, state->a, state->base_offset, state->ox_start,
                                            state->ox_end, state->kp, state->M, state->ms);
  }
  MM_WORKER_ADD(11, t);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

// Submits the DMA-fed staging of row tiles [ox_start, ox_end) split by packs over n workers, all signalling
// sync. Returns without waiting.
static void q4_act_stage_submit(copy_shuffle_q4_activation_task_state_t *tasks, worker_synctoken_t *sync, int n,
                                __fp16 *vtcm_activation, const uint8_t *a, int base_offset, int ox_start, int ox_end,
                                int kp, int M, int ms, uint8_t *raw, int raw_packs) {
  const int packs = kp / 2;
  if (n > packs) {
    n = packs;
  }
  worker_pool_synctoken_init(sync, n);
  for (int w = 0; w < n; ++w) {
    tasks[w].vtcm_activation = vtcm_activation;
    tasks[w].a               = a;
    tasks[w].base_offset     = base_offset;
    tasks[w].ox_start        = ox_start;
    tasks[w].ox_end          = ox_end;
    tasks[w].kp              = kp;
    tasks[w].M               = M;
    tasks[w].ms              = ms;
    tasks[w].raw             = raw;
    tasks[w].raw_packs       = raw_packs;
    tasks[w].p_begin         = packs * w / n;
    tasks[w].p_end           = packs * (w + 1) / n;
    tasks[w].shared_sync     = sync;
    worker_pool_job_t job;
    job.fptr = copy_shuffle_q4_activation_worker_loop;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      copy_shuffle_q4_activation_worker_loop(&tasks[w], 0);
    }
  }
}

static inline void copy_shuffle_q4_activation_tiles_pack64_parallel(__fp16 *vtcm_activation, const uint8_t *a,
                                                                    int base_offset, int ox_start, int ox_end, int kp,
                                                                    int M, int ms) {
  const int ox_count = ox_end - ox_start;
  int task_count = (int)g_max_num_workers;
  if (task_count > ox_count) {
    task_count = ox_count;
  }
  if (task_count <= 1) {
    copy_shuffle_q4_activation_tiles_pack64(vtcm_activation, a, base_offset, ox_start, ox_end, kp, M, ms);
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
    tasks[i].ms              = ms;
    tasks[i].raw             = NULL;
    tasks[i].raw_packs       = 0;
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
  // count * 64 bytes of fp16 1.0; the buffer is padded by one vector, so whole vectors are safe.
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
  for (int i = 0; i < count * 64; i += 128) {
    vmemu(vtcm_scales + i) = one;
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

static inline void store_q4_output_tile(uint8_t *c, __fp16 *vtcm_output, const uint8_t *bias,
                                        const uint8_t *vtcm_scales, int M, int ms, int ox, int oy_start, int oy) {
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
  uint8_t    *dst_ptr = c + (size_t) (pack_idx * ms + ox * 32) * 128;
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

static inline void store_q4_output_range(uint8_t *c, __fp16 *vtcm_output, const uint8_t *bias,
                                         const uint8_t *vtcm_scales, int M, int ms, int ox, int oy_start, int oy_begin,
                                         int oy_end, int pack) {
  for (int oy = oy_begin; oy < oy_end; ++oy) {
    __fp16* output_tile = vtcm_output + (oy - oy_start) * 1024;
    if (pack == 64 && oy + 1 < oy_end && ((oy * 32) & 63) == 0) {
      __fp16* output_tile_next = vtcm_output + (oy + 1 - oy_start) * 1024;
      uint8_t *dst_ptr          = c + (size_t) (((oy * 32) / 64) * ms + ox * 32) * 128;
      pack_q4_output_tile_pair(dst_ptr, output_tile, output_tile_next, bias, vtcm_scales,
                               M, ox, oy_start, oy);
      ++oy;
    } else {
      store_q4_output_tile(c, output_tile, bias, vtcm_scales, M, ms, ox, oy_start, oy);
    }
  }
}

typedef struct {
  uint8_t* c;
  __fp16* vtcm_output;
  const uint8_t* bias;
  const uint8_t* vtcm_scales;
  int M;
  int                 ms;
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
  const unsigned long long      t     = MM_T0();
  store_q4_output_range(state->c, state->vtcm_output, state->bias, state->vtcm_scales, state->M, state->ms, state->ox,
                        state->oy_start, state->oy_begin, state->oy_end, state->pack);
  MM_WORKER_ADD(10, t);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

// Unit costs of the prefill GEMM's resources, measured once on the first large call (diagnostic, off). What a
// pipeline can reach is set by these, so they are measured in isolation and under the contention the
// pipeline would create. Written to profile[104..118] (the host prints value / 1000), us unless noted:
//   [104] production dequantization (process_q4_weight_lut), scales cold in DDR, per 4096 tiles
//   [105] the same with the scale range l2fetched first          [106] the same with the scales in VTCM
//   [107] dequant core only, symmetric, constant scale            [108] the same, asymmetric (scale + offset)
//   [109] 64 x (16 HMX products, then store their 8 tile pairs), sequential
//   [110] the same software-pipelined: issue the next 16 products before storing the previous 8 pairs
//   [111] the 64 x 16 HMX products alone                          [112] the 64 x 8 pair stores alone
//   [113] 1024 kp=32 products alone                               [114] the same while 8 workers dequantize
//   [115] 512 kp=64 products   [116] 256 kp=128 products   [118] 4096 kp=8 products (same MACs as [113])
//   [117] 256 rounds of dispatching one empty job per worker
// Measured earlier with this harness on v81 and dropped: one DMA engine moves 58 GB/s DDR -> VTCM, 2-8
// engines together 78-82 GB/s, so the weight DMA is not a bandwidth limit.
#ifndef HTP_MM_MICROBENCH
#  define HTP_MM_MICROBENCH 0
#endif
#if HTP_MM_MICROBENCH
#  include "HAP_perf.h"
unsigned long long g_mm_mb[16];
static int         g_mm_mb_done;

typedef struct {
  const uint8_t      *int4;
  __fp16             *dst;
  int                 tiles, reps, asym;
  worker_synctoken_t *sync;
} mm_mb_task_t;

static void mm_mb_dequant(const mm_mb_task_t *t) {
  const HVX_Vector vlut  = vmemu(q4_to_fp16_lut);
  // 1/64, so the HMX products below stay well inside fp16: inf or denormal operands slow HMX down.
  const HVX_Vector scale = Q6_Vh_vsplat_R(0x2400);
  const HVX_Vector off   = Q6_Vh_vsplat_R(0x2000);
  for (int r = 0; r < t->reps; ++r) {
    for (int i = 0; i < t->tiles; ++i) {
      dequant_q4_tile_scaled_from_vrmpy(t->int4 + (size_t) i * 512, t->dst + (size_t) i * 1024, vlut, scale,
                                        t->asym ? off : Q6_V_vzero(), t->asym);
    }
  }
}

static void mm_mb_worker(void *data, int worker_index) {
  (void) worker_index;
  const mm_mb_task_t *t = (const mm_mb_task_t *) data;
  mm_mb_dequant(t);
  worker_pool_synctoken_jobdone(t->sync);
}

static void mm_mb_noop(void *data, int worker_index) {
  (void) worker_index;
  worker_pool_synctoken_jobdone((worker_synctoken_t *) data);
}

static void mm_mb_launch(mm_mb_task_t *tasks, int n, worker_synctoken_t *sync) {
  worker_pool_synctoken_init(sync, n);
  for (int w = 0; w < n; ++w) {
    tasks[w].sync = sync;
    worker_pool_job_t job;
    job.fptr = mm_mb_worker;
    job.dptr = &tasks[w];
    if (worker_pool_submit(NULL, job) != 0) {
      mm_mb_worker(&tasks[w], 0);
    }
  }
}

static void mm_mb_hmx(const __fp16 *act, const __fp16 *wgt, __fp16 *out, int kt, int iters) {
  for (int it = 0; it < iters; ++it) {
    hmx_load_tiles_fp16(act, wgt, kt);
    hmx_consume_accumulator_fp16(out + (it & 15) * 1024);
  }
}

static void mm_microbench(uint8_t *c, const uint8_t *b, const uint8_t *b_scale, int M, int N, int K,
                          int scale_block_num, int scale_asymmetric, int weight_is_vrmpy) {
  uint8_t           *base = (uint8_t *) vtcm_manager_get_vtcm_base();
  uint8_t           *int4 = base;                            // 8 x 128 KB
  __fp16            *fp16 = (__fp16 *) (base + (1u << 20));  // 8 x 256 KB
  __fp16            *hact = (__fp16 *) (base + (3u << 20));  // 128 tiles
  __fp16            *hwgt = hact + 128 * 1024;               // 128 tiles
  __fp16            *hout = hwgt + 128 * 1024;               // 2 x 16 tiles
  uint8_t           *hsc  = (uint8_t *) (hout + 32 * 1024);
  uint8_t           *vsc  = base + (5u << 20);               // scales copied to VTCM
  const int          nw   = (int) g_max_num_workers < 8 ? (int) g_max_num_workers : 8;
  const int          kp = K / 32, np = N / 32, mp = M / 32;
  mm_mb_task_t       tasks[8];
  worker_synctoken_t sync;
  memcpy(int4, b, 8 * 128 * 1024);

  // Production dequantization: cold scales, prefetched scales, scales in VTCM.
  {
    const int    count     = (1 << 20) / (kp * 512);
    const size_t sc_per_oy = (size_t) scale_block_num * (weight_is_vrmpy ? 128 : (scale_asymmetric ? 256 : 128));
    int          reps      = count > 0 ? np / count : 0;
    if (reps > 3) {
      reps = 3;
    }
    for (int v = 0; v < 3 && reps > 0; ++v) {
      unsigned long long us = 0;
      for (int r = 0; r < reps; ++r) {
        // A fresh oy range per rep and per variant keeps the DDR scales cold for variant 0.
        const int      o0  = ((v * reps + r) % (np / count)) * count;
        const uint8_t *scp = b_scale;
        if (v == 1) {
          l2fetch(b_scale + o0 * sc_per_oy, 8192, 8192, (uint32_t) ((count * sc_per_oy + 8191) / 8192), 0);
        } else if (v == 2) {
          memcpy(vsc, b_scale + o0 * sc_per_oy, count * sc_per_oy);
          scp = vsc - o0 * sc_per_oy;
        }
        const unsigned long long t0 = HAP_perf_get_time_us();
        process_q4_weight_lut(o0, 0, count, kp, int4, fp16, scp, scale_block_num, scale_asymmetric, weight_is_vrmpy);
        us += HAP_perf_get_time_us() - t0;
      }
      g_mm_mb[v] = us * 4096 / ((unsigned long long) reps * count * kp);
    }
  }
  for (int asym = 0; asym <= 1; ++asym) {
    mm_mb_task_t             t  = { int4, fp16, 256, 16, asym, NULL };
    const unsigned long long t0 = HAP_perf_get_time_us();
    mm_mb_dequant(&t);
    g_mm_mb[3 + asym] = HAP_perf_get_time_us() - t0;
  }

  // HMX, alone, with stores, and with different chain lengths.
  memcpy(hact, fp16, 256 * 2048);
  hmx_manager_enable_execution();
  hmx_init_column_scales(hsc, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(hsc);
  if (mp >= 1 && np >= 16) {
    for (int v = 0; v < 4; ++v) {
      const unsigned long long t0 = HAP_perf_get_time_us();
      for (int it = 0; it < 64; ++it) {
        __fp16   *o  = hout + (it & 1) * 16 * 1024;
        const int ox = it % mp;
        if (v == 0 || v == 2) {
          for (int oy = 0; oy < 16; ++oy) {
            hmx_load_tiles_fp16(hact, hwgt + oy * 32 * 1024 % (96 * 1024), 32);
            hmx_consume_accumulator_fp16(o + oy * 1024);
          }
        }
        if (v == 1) {
          if (it == 0) {
            for (int oy = 0; oy < 16; ++oy) {
              hmx_load_tiles_fp16(hact, hwgt + oy * 32 * 1024 % (96 * 1024), 32);
              hmx_consume_accumulator_fp16(o + oy * 1024);
            }
          }
          if (it + 1 < 64) {
            __fp16 *on = hout + ((it + 1) & 1) * 16 * 1024;
            for (int oy = 0; oy < 16; ++oy) {
              hmx_load_tiles_fp16(hact, hwgt + oy * 32 * 1024 % (96 * 1024), 32);
              hmx_consume_accumulator_fp16(on + oy * 1024);
            }
          }
        }
        if (v != 2) {
          store_q4_output_range(c, o, NULL, NULL, M, M, ox, 0, 0, 16, 64);
        }
      }
      if (v == 2) {
        // Make sure the last products are drained before the clock stops.
        volatile HVX_Vector sink = vmem(hout);
        (void) sink;
      }
      g_mm_mb[5 + v] = HAP_perf_get_time_us() - t0;
    }
  }
  for (int v = 0; v < 2; ++v) {
    if (v == 1) {
      for (int w = 0; w < nw; ++w) {
        tasks[w] = (mm_mb_task_t) { int4 + w * 131072, fp16 + w * 131072, 256, 48, 1, NULL };
      }
      mm_mb_launch(tasks, nw, &sync);
    }
    const unsigned long long t0 = HAP_perf_get_time_us();
    mm_mb_hmx(hact, hwgt, hout, 32, 1024);
    g_mm_mb[9 + v] = HAP_perf_get_time_us() - t0;
    if (v == 1) {
      worker_pool_synctoken_wait(&sync);
    }
  }
  {
    unsigned long long t0 = HAP_perf_get_time_us();
    mm_mb_hmx(hact, hwgt, hout, 64, 512);
    g_mm_mb[11] = HAP_perf_get_time_us() - t0;
    t0          = HAP_perf_get_time_us();
    mm_mb_hmx(hact, hwgt, hout, 128, 256);
    g_mm_mb[12] = HAP_perf_get_time_us() - t0;
    t0          = HAP_perf_get_time_us();
    mm_mb_hmx(hact, hwgt, hout, 8, 4096);
    g_mm_mb[14] = HAP_perf_get_time_us() - t0;
  }
  hmx_unit_release();
  hmx_manager_disable_execution();
  {
    const unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < 256; ++it) {
      worker_pool_synctoken_init(&sync, nw);
      for (int w = 0; w < nw; ++w) {
        worker_pool_job_t job;
        job.fptr = mm_mb_noop;
        job.dptr = &sync;
        if (worker_pool_submit(NULL, job) != 0) {
          mm_mb_noop(&sync, 0);
        }
      }
      worker_pool_synctoken_wait(&sync);
    }
    g_mm_mb[13] = HAP_perf_get_time_us() - t0;
  }
}
#endif

int hmx_matmulq4fp16(uint8_t *c, const uint8_t *a, const uint8_t *b, const uint8_t *b_scale, const uint8_t *bias, int M,
                     int K, int N, int mp_max, int np_max, int kp_max, int scale_block_num, int scale_asymmetric,
                     int weight_is_vrmpy, int ms) {
  // ms is the row stride of a and c (both pack-64): the host runs a GEMM whose activation does not fit VTCM as
  // several row windows of one tensor, each with a and c pointing at its first row.
  if (ms < M) {
    ms = M;
  }
  const unsigned long long mm_total_start = MM_T0();
#if HTP_MM_MICROBENCH
  if (!g_mm_mb_done && M >= 256 && (size_t) N * K / 2 >= (2u << 20) && scale_block_num > 1) {
    g_mm_mb_done = 1;
    mm_microbench(c, b, b_scale, M, N, K, scale_block_num, scale_asymmetric, weight_is_vrmpy);
  }
#endif
  if (scale_block_num <= 0) scale_block_num = 1;
  const int dequant_in_weight = scale_block_num > 1;
  // Asymmetric qbias is folded into the dequantized weight, which only happens in the block path.
  // Channel-wise scales are applied at the HMX output stage instead and have no qbias slot, so the
  // host rejects that combination; fail loudly rather than silently dropping the offset.
  if (scale_asymmetric && !dequant_in_weight) {
    return -4;
  }
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
  // Traffic accounting, all in KB, so the bandwidth of each stage can be divided out of its phase time.
  MM_BUMP(14, 1);
  MM_BUMP(19, (unsigned long long) M * N * K / 1000000);         // MMAC
  MM_BUMP(15, (unsigned long long) mp * np * kp * 4096 / 1024);  // HMX VTCM reads (act + weight tiles)
  MM_BUMP(16, (unsigned long long) M * N * 2 / 1024);            // output DDR writes
  MM_BUMP(17, (unsigned long long) N * K / 2 / 1024);            // weight int4 DDR reads
  MM_BUMP(18, (unsigned long long) N * K * 2 / 1024);            // dequantized fp16 VTCM writes
  const bool reuse_activation = (mp > 0 && mp <= mp_chunk);
  // One buffer in both cases: the second one existed to double-buffer the activation DMA prefetch, which
  // the parallel copy+shuffle staging replaced. Halving this frees VTCM for a larger np_chunk.
  const int  activation_buffers = 1;
  // A k-tile is half a pack-64 row, so staging straight into tile layout needs one 64-byte-wide
  // descriptor per (ox, k). Instead DMA the chunk verbatim -- one descriptor, 4 KB+ rows -- and shuffle
  // it in VTCM. Needs K to be a whole number of packs; it always is for NC4HW4 pack-64 activations.
  const bool act_bulk_dma       = !reuse_activation && (K % 64) == 0;
  // The weight buffer is what serializes this op: a panel's dequantization cannot start until the previous
  // panel's HMX has stopped reading the buffer, so the workers (42% busy) and this thread (33% idle) alternate
  // instead of overlapping. Halving the panel and staging two of them keeps the VTCM footprint identical, since
  // the host sizes np so that exactly one panel's weights fill VTCM. Needs the activation staged once up front,
  // otherwise the per-row-tile activation DMA would collide with the next panel's weight chain.
  // Restricted to the block path: the per-channel path chains a per-panel scale DMA behind the weight chain,
  // which the look-ahead panel has no place to put.
  // Double-buffer the weight panels whenever the oy range can be split in two, not only when it
  // exceeds np_chunk. The `np > np_chunk` form let four of the seven matmuls per layer fall through
  // to a single panel: k, v, o and ffn_down all have np = 32 against an np_chunk of 36. That is 47%
  // of the model's weight bytes whose dequantization is then fully exposed instead of overlapped,
  // and it matches the main thread's 9.52 ms of MM_DEQUANT_WAIT at pp512.
  int        weight_panel_np    = np_chunk;
  if (weight_panel_np > np) {
    weight_panel_np = np;
  }
  const int weight_panels =
    (reuse_activation && dequant_in_weight && weight_panel_np >= DSP_Q4_MIN_DOUBLE_BUFFER_NP) ? 2 : 1;
  if (weight_panels == 2) {
    np_chunk = (weight_panel_np / 2) & ~1;
    if (np_chunk < 2) {
      np_chunk = 2;
    }
  }
  // The first panel's dequantization has nothing to hide behind: HMX cannot start until every one of its
  // chunks is in. So the first panel is cut down to one even pair of oy per worker, which dequantizes in a
  // few microseconds, and the second (full) panel dequantizes under its HMX. On v81 a worker dequantizes
  // one oy of kp tiles about 3x faster than HMX consumes it for mp = 16, so one step of ramp is enough.
  int first_panel_np = np_chunk;
  if (weight_panels == 2) {
    const int ramp = 2 * (int) (g_max_num_workers > 1 ? g_max_num_workers : 1);
    if (ramp < first_panel_np) {
      first_panel_np = ramp;
    }
  }
#define Q4_PANEL_NEXT(start) ((start) == 0 ? first_panel_np : (start) + np_chunk)
  // Measured and rejected: moving the store into the worker pool for the block path too. The resource
  // account says this thread is the binding one (on 2B it is 92% busy against a pool at 50%), and the
  // phase table does shift -- MM_STORE_INLINE goes to zero and MM_KERNEL_TOTAL drops 15.7% -- but the
  // end-to-end throughput is flat on all three models (v79, three alternating rounds, -rep 5: 0.6B
  // +0.0%, 2B -0.5%, 4B +0.3%). MM_HMX grows 45% when the store runs alongside it, which eats exactly
  // what the store gave back. Do not re-open this without first explaining that.
  const int    async_output_candidate          = (!dequant_in_weight && mp > 1 && g_max_num_workers > 1);
  // One slot per ox when it fits. Four fixed slots made the main thread queue up behind its own stores
  // as soon as mp exceeded them: on 4B (mp = 16) MM_OUTBUF_WAIT went from 0 to 97 ms, which is more than
  // the 106 ms of MM_STORE_INLINE the async path removes. A slot is only np_chunk * 2 KB -- 18 KB on 4B,
  // 25 KB on 2B -- so covering all of mp costs a few hundred KB, and the budget check below falls back
  // to the synchronous path when even that does not fit.
  int          output_slots                    = 1;
  if (async_output_candidate) {
    output_slots = mp < Q4_MAX_OUTPUT_SLOTS ? mp : Q4_MAX_OUTPUT_SLOTS;
    if (output_slots < 2) {
      output_slots = 2;
    }
  }
  const size_t q4_vtcm_safe_size = 8 * 1024 * 1024 - 16 * 1024;
  // Fixed cost of everything except the output slots, so the slot count can be solved for rather than
  // guessed: VTCM is what decides how deep this stage of the pipeline can be.
  const size_t async_fixed_bytes = (size_t) weight_panels * np_chunk * weight_bytes_per_np +
                                   (size_t) weight_panels * np_chunk * weight_int4_bytes_per_np +
                                   (size_t) activation_buffers * mp_chunk * act_bytes_per_mp +
                                   (size_t) (act_bulk_dma ? mp_chunk * act_bytes_per_mp : 0) + (size_t) np_chunk * 256 +
                                   (size_t) (np_chunk * 64 + 64);
  const size_t output_slot_bytes = (size_t) np_chunk * 1024 * sizeof(__fp16);
  if (async_output_candidate && async_fixed_bytes < q4_vtcm_safe_size) {
    const size_t affordable = (q4_vtcm_safe_size - async_fixed_bytes) / output_slot_bytes;
    if ((size_t) output_slots > affordable) {
      output_slots = (int) affordable;
    }
  }
  // Two slots is the minimum for any overlap at all; below that keep the store on this thread.
  const int async_output_store = async_output_candidate && output_slots >= 2;
  // The synchronous path stores on this thread, and HMX runs asynchronously to it: the products of the
  // next row tile are issued before the current one is stored, so the store runs under them instead of
  // leaving HMX idle. That needs a second output buffer, taken only when VTCM has room for it (the host
  // sized the tiles for one). Microbenchmark on v81: 64 rounds of 16 products + 8 pair stores take 92 us
  // in sequence and 81 us pipelined, against 78 us for the products alone.
  const int sync_output_pingpong =
    !async_output_store && mp > 1 && async_fixed_bytes + 2 * output_slot_bytes <= q4_vtcm_safe_size;
  const int output_buffers     = async_output_store ? output_slots : (sync_output_pingpong ? 2 : 1);
  const int scale_buffer_bytes = np_chunk * 64 + 64;
  // The block path folds the scales into the weight and leaves this at identity, so one buffer is
  // enough; only the per-channel path needs a second one to stay a panel ahead.
  const int scale_buffers      = (async_output_store && !dequant_in_weight) ? 2 : 1;
  MM_BUMP(20, async_output_store);
  MM_BUMP(21, (async_fixed_bytes + (size_t) output_buffers * output_slot_bytes) / 1024);
  MM_BUMP(22, np_chunk);

  // The staged activation goes first so that its base address depends only on (M, K), not on np_chunk.
  // Consecutive matmuls sharing one activation (the Q/K/V and gate/up projections) then find it at the
  // same place and can skip the staging entirely -- see the act_reusable check below.
  uint8_t     *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_buffers * mp_chunk * act_bytes_per_mp);
  // Landing zone for the bulk activation DMA, in the DDR pack-64 layout; the shuffle below turns it into
  // HMX tiles. Only the chunked path needs it, and it is exactly the buffer the removed DMA double-
  // buffering used to occupy, so the VTCM total is unchanged.
  // The chunked path re-stages the activation for every weight panel, and a row tile is scattered over
  // K / 64 packs M * 128 B apart: on 4B that was 665K descriptors of ~4 KB on this thread's one engine,
  // 285 ms of DMA per 512-token prefill, two thirds of the op. When the landing zone can hold two blocks
  // per worker instead, the workers split the packs and each moves its share on its own engine, then
  // shuffles it (copy_shuffle_q4_activation_dma). It needs less VTCM than the verbatim landing zone.
  const int    act_nw = (int) g_max_num_workers > Q4_MAX_WEIGHT_CHUNKS ? Q4_MAX_WEIGHT_CHUNKS : (int) g_max_num_workers;
  const size_t act_worker_bytes = (size_t) act_nw * 2 * Q4_ACT_DMA_PACKS * 4096;
  // The host reserves a second activation-sized buffer for this path (the verbatim landing zone). If what VTCM
  // has left over can hold the workers' small landing blocks, keep that buffer as a second activation chunk
  // instead, so the next chunk is staged while HMX consumes the current one (the 4B ffn_down case, where a
  // chunk is a single row tile and cannot be split in halves).
  const size_t act_chunk_bytes  = (size_t) mp_chunk * act_bytes_per_mp;
  int          act_two_buffers  = 0;
  int          act_blk_packs    = Q4_ACT_DMA_PACKS;
  if (act_bulk_dma && act_nw > 1 && dequant_in_weight && !async_output_store) {
    const size_t vtcm_total = vtcm_manager_get_vtcm_size();
    const size_t limit      = (vtcm_total > 16 * 1024 && vtcm_total - 16 * 1024 < q4_vtcm_safe_size) ?
                                vtcm_total - 16 * 1024 :
                                q4_vtcm_safe_size;
    const size_t used       = async_fixed_bytes + (size_t) output_buffers * output_slot_bytes + 64 * 1024;
    for (int pk = Q4_ACT_DMA_PACKS; pk >= 1; pk /= 2) {
      if (used + (size_t) act_nw * 2 * pk * 4096 <= limit) {
        act_two_buffers = 1;
        act_blk_packs   = pk;
        break;
      }
    }
  }
  const bool act_worker_dma = act_bulk_dma && act_nw > 1 && (act_two_buffers || act_chunk_bytes >= act_worker_bytes);
  __fp16    *vtcm_act_raw =
    act_bulk_dma ?
      (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (act_worker_dma && !act_two_buffers) ? act_worker_bytes : act_chunk_bytes) :
      NULL;
  // The workers' landing blocks: the reshaped landing zone, or their own area beside the second buffer.
  uint8_t *act_blk     = act_two_buffers ?
                           (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, (size_t) act_nw * 2 * act_blk_packs * 4096) :
                           (uint8_t *) vtcm_act_raw;
  __fp16  *vtcm_weight = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_panels * np_chunk * weight_bytes_per_np);
  uint8_t *vtcm_weight_int4 =
    (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, weight_panels * np_chunk * weight_int4_bytes_per_np);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_buffers * np_chunk * 1024 * sizeof(__fp16));
  HVX_Vector *vtcm_hmx_scales = (HVX_Vector *) vtcm_seq_alloc(&vtcm_ptr, np_chunk * 256);
  uint8_t  *vtcm_scales     = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, scale_buffers * scale_buffer_bytes);
  // Landing blocks for the DMA-fed activation staging (copy_shuffle_q4_activation_dma), two per worker, in
  // whatever VTCM the tiles above left over; without room the staging shuffles straight from DDR.
  uint8_t    *vtcm_act_dma    = NULL;
  int         act_dma_packs   = 0;
  if (reuse_activation && dequant_in_weight && g_max_num_workers > 1) {
    const size_t vtcm_total = vtcm_manager_get_vtcm_size();
    const size_t limit      = (vtcm_total > 16 * 1024 && vtcm_total - 16 * 1024 < q4_vtcm_safe_size) ?
                                vtcm_total - 16 * 1024 :
                                q4_vtcm_safe_size;
    const size_t used       = async_fixed_bytes + (size_t) output_buffers * output_slot_bytes + 64 * 1024;
    const int    nw = (int) g_max_num_workers > Q4_MAX_WEIGHT_CHUNKS ? Q4_MAX_WEIGHT_CHUNKS : (int) g_max_num_workers;
    for (int pk = Q4_ACT_DMA_PACKS; pk >= 2; pk /= 2) {
      const size_t need = (size_t) nw * 2 * pk * 4096;
      if (used + need <= limit) {
        act_dma_packs = pk;
        vtcm_act_dma  = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, need);
        break;
      }
    }
  }
  MM_BUMP(26, act_dma_packs);
  // The bulk activation staging sends one 1D descriptor per pack. It cannot use a single 2D descriptor:
  // roi_width and dst_stride would carry act_rows * 128 B and src_stride M * 128 B, and all three are
  // uint16_t (see dma_desc_2d), so from M = 512 they silently truncate. A roi_width of 0 makes the
  // descriptor illegal and the DMA engine faults, which kills the cDSP and reboots v81 outright.
  // `length` is 24-bit, so the 1D chain carries the same 4 KB+ rows with no truncation.
  const int      act_bulk_packs = act_bulk_dma ? kp / 2 : 0;
  dma_desc_1d_t *act_bulk_descs =
    act_bulk_packs > 0 ? (dma_desc_1d_t *) memalign(64, (size_t) act_bulk_packs * sizeof(dma_desc_1d_t)) : NULL;
  if (act_bulk_packs > 0 && act_bulk_descs == NULL) {
    return 1;
  }
  memset(vtcm_scales, 0, scale_buffers * scale_buffer_bytes);
  if (dequant_in_weight) {
    init_identity_q4_output_scales(vtcm_scales, np_chunk);
  }

  MM_ADD(31, mm_total_start);
  const unsigned long long mm_setup_start = MM_T0();
  hmx_manager_enable_execution();
  hmx_init_column_scales(vtcm_hmx_scales, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(vtcm_hmx_scales);

  int pack = 64;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif
  MM_ADD(5, mm_setup_start);
  const unsigned long long     mm_preloop_start = MM_T0();
  worker_synctoken_t           output_store_sync[Q4_MAX_OUTPUT_SLOTS];
  store_q4_output_task_state_t output_store_tasks[Q4_MAX_OUTPUT_SLOTS];
  int                          output_store_active[Q4_MAX_OUTPUT_SLOTS] = { 0 };

  q4_weight_ctx_t  weight_ctx = { b,
                                  b_scale,
                                  vtcm_weight_int4,
                                  vtcm_weight,
                                  (size_t) np_chunk * weight_int4_bytes_per_np,
                                  (size_t) np_chunk * weight_bytes_per_np,
                                  kp,
                                  scale_block_num,
                                  scale_asymmetric,
                                  weight_is_vrmpy,
                                  (int) g_max_num_workers > Q4_MAX_WEIGHT_CHUNKS ? Q4_MAX_WEIGHT_CHUNKS :
                                                                                   (int) g_max_num_workers,
                                  mp };
  q4_panel_state_t panels[2];
  memset(panels, 0, sizeof(panels));
  _Alignas(64) dma_desc_2d_t scale_desc[1];

  // The activation has to be staged before the first panel's weights are consumed, and in the pipelined case also
  // before the second panel's DMA is issued, so do it once here instead of inside the panel loop.
  //
  // The Q/K/V projections (and gate/up) are consecutive commands reading one activation, so the staging is
  // redundant from the second one on. It is skipped only when the immediately preceding command staged the
  // very same (pointer, M, K): requiring adjacency in the command sequence is what makes this safe, since
  // any other op in between may have reused the VTCM.
  // Staging the activation in row-tile groups that each signal on their own lets the first panel start as
  // soon as its first group and first weight chunk are in, instead of after the whole activation: HMX then
  // runs under the rest of the staging and under the panel's dequantization. The panel goes to the pool
  // first so its DMA is already moving while the first group is shuffled.
  copy_shuffle_q4_activation_task_state_t act_tasks[Q4_ACT_GROUPS];
  copy_shuffle_q4_activation_task_state_t act_g0_tasks[Q4_MAX_WEIGHT_CHUNKS];
  worker_synctoken_t                      act_sync[Q4_ACT_GROUPS];
  int                                     act_pending[Q4_ACT_GROUPS] = { 0 };
  int                                     act_group_ox               = mp;
  if (reuse_activation) {
    const int act_reusable = g_act_stage.base == a && g_act_stage.m == M && g_act_stage.k == K &&
                             g_act_stage.cmd_seq + 1 == htp_ops_command_seq();
    const int act_async    = !act_reusable && dequant_in_weight && mp > 1 && g_max_num_workers > 1;
    if (act_async) {
      MM_BUMP(25, (unsigned long long) M * K * sizeof(__fp16) / 1024);
      // Group 0 gates the first HMX product of the call, and one worker staging it alone took 75-110 us at
      // K = 9216 (32 rows x 18 KB): that was most of the 150 us every Qwen3.5-4B ffn_down window spent before
      // HMX started. Split it by packs over all workers and hand it out ahead of the first panel; its pieces
      // are a few microseconds each, so the panel's dequantization starts right behind them.
      const int act_g0_split = vtcm_act_dma != NULL && weight_ctx.max_chunks > 1 && kp >= 4;
      if (act_g0_split) {
        q4_act_stage_submit(act_g0_tasks, &act_sync[0], weight_ctx.max_chunks, vtcm_activation, a, 0, 0, 1, kp, M, ms,
                            vtcm_act_dma, act_dma_packs);
        act_pending[0] = 1;
      }
      int first_end = Q4_PANEL_NEXT(0) < np ? Q4_PANEL_NEXT(0) : np;
      q4_panel_submit(&weight_ctx, &panels[0], 0, first_end, 0, 0);
      // One group per worker: a pool slot frees on pickup and there are only workers + 1 of them, so a
      // finer split makes this thread block in worker_pool_submit until a worker finishes a job.
      int groups = mp < (int) g_max_num_workers ? mp : (int) g_max_num_workers;
      if (groups > Q4_ACT_GROUPS) {
        groups = Q4_ACT_GROUPS;
      }
      // Group 0 is a single row tile so HMX can start after one tile's staging; the rest share the
      // remaining workers. act_group_start maps a row tile back to its group.
      act_group_ox = groups > 1 ? (mp - 1 + groups - 2) / (groups - 1) : mp;
      for (int g = act_g0_split ? 1 : 0;; ++g) {
        const int ox0 = g == 0 ? 0 : 1 + (g - 1) * act_group_ox;
        if (ox0 >= mp) {
          break;
        }
        const int ox1 = g == 0 ? 1 : (ox0 + act_group_ox < mp ? ox0 + act_group_ox : mp);
        worker_pool_synctoken_init(&act_sync[g], 1);
        act_tasks[g].vtcm_activation = vtcm_activation;
        act_tasks[g].a               = a;
        act_tasks[g].base_offset     = ox0 * kp * 32 * 32;
        act_tasks[g].ox_start        = ox0;
        act_tasks[g].ox_end          = ox1;
        act_tasks[g].kp              = kp;
        act_tasks[g].M               = M;
        act_tasks[g].ms              = ms;
        act_tasks[g].raw             = vtcm_act_dma;
        act_tasks[g].raw_packs       = act_dma_packs;
        act_tasks[g].p_begin         = 0;
        act_tasks[g].p_end           = kp / 2;
        act_tasks[g].shared_sync     = &act_sync[g];
        act_pending[g]               = 1;
        worker_pool_job_t job;
        job.fptr = copy_shuffle_q4_activation_worker_loop;
        job.dptr = &act_tasks[g];
        if (worker_pool_submit(NULL, job) != 0) {
          copy_shuffle_q4_activation_worker_loop(&act_tasks[g], 0);
        }
      }
    } else if (!act_reusable) {
      const unsigned long long t = MM_T0();
      MM_BUMP(25, (unsigned long long) M * K * sizeof(__fp16) / 1024);
      copy_shuffle_q4_activation_tiles_pack64_parallel(vtcm_activation, a, 0, 0, mp, kp, M, ms);
      MM_ADD(2, t);
    } else {
      MM_BUMP(23, 1);
    }
    g_act_stage.base    = a;
    g_act_stage.m       = M;
    g_act_stage.k       = K;
    g_act_stage.cmd_seq = htp_ops_command_seq();
  } else {
    g_act_stage.base = NULL;
  }

  MM_ADD(29, mm_preloop_start);
  // Chunked path with worker staging: split the activation buffer in two halves and stage the next chunk
  // into one while HMX consumes the other. Only on the synchronous store path, whose per-chunk final store
  // is what guarantees HMX has finished reading a half before it is refilled.
  const int act_pingpong = !reuse_activation && act_worker_dma && dequant_in_weight && !async_output_store &&
                           (act_two_buffers || mp_chunk >= 2);
  const int act_step     = act_two_buffers ? mp_chunk : (act_pingpong ? mp_chunk / 2 : mp_chunk);
  // Element offset of the second half / buffer from vtcm_activation.
  const int act_buf1_off = act_two_buffers ? (int) (vtcm_act_raw - vtcm_activation) : act_step * kp * 32 * 32;
  copy_shuffle_q4_activation_task_state_t act_pf_tasks[Q4_MAX_WEIGHT_CHUNKS];
  worker_synctoken_t                      act_pf_sync;
  int                                     act_pf_pending = 0, act_pf_ox = -1, act_pf_buf = 0, act_chunk_index = 0;
  int                                     panel_index = 0;
  for (int oy_start = 0; oy_start < np; oy_start = Q4_PANEL_NEXT(oy_start), ++panel_index) {
    int oy_end = Q4_PANEL_NEXT(oy_start);
    if (oy_end > np) {
      oy_end = np;
    }
    const int weight_buf          = weight_panels == 2 ? (panel_index & 1) : 0;
    const int oy_block            = oy_start / np_chunk;
    // The scale buffer is only double-buffered on the per-channel path; the block path keeps one
    // identity buffer. The output slots are indexed by ox alone, so they do not need a per-panel group.
    const int scale_buf_idx       = scale_buffers > 1 ? (oy_block & 1) : 0;
    uint8_t  *current_vtcm_scales = vtcm_scales + scale_buf_idx * scale_buffer_bytes;
    __fp16   *current_weight      = (__fp16 *) ((uint8_t *) vtcm_weight + (size_t) weight_buf * weight_ctx.fp16_stride);
    q4_panel_state_t *panel       = &panels[weight_buf];

    uint32_t scale_dma_ptr = 0;
    if (!dequant_in_weight) {
      memset(&scale_desc[0], 0, sizeof(dma_desc_2d_t));
      scale_desc[0].type       = DMA_DESC_TYPE_2D;
      scale_desc[0].src_bypass = 1;
      scale_desc[0].dst_bypass = 1;
      scale_desc[0].ordered    = 1;
      scale_desc[0].src        = (uint32_t) (b_scale + oy_start * 32 * sizeof(__fp16));
      scale_desc[0].dst        = (uint32_t) current_vtcm_scales;
      scale_desc[0].roi_width  = 32 * sizeof(__fp16);
      scale_desc[0].roi_height = oy_end - oy_start;
      scale_desc[0].src_stride = 32 * sizeof(__fp16);
      scale_desc[0].dst_stride = 64;
      scale_desc[0].next       = 0;
      scale_dma_ptr            = (uint32_t) &scale_desc[0];
    }
    uint32_t tail_dma_ptr = scale_dma_ptr;

    if (!panel->issued || panel->oy_start != oy_start) {
      q4_panel_submit(&weight_ctx, panel, oy_start, oy_end, weight_buf, tail_dma_ptr);
    }
    // The DMA engine belongs to this thread and takes one chain at a time, so it has to be idle before another
    // one is started -- issuing on a busy engine reboots the device. By now this panel's chain was started either
    // just above or a whole panel ago, so the wait is short.
    {
      const unsigned long long t = MM_T0();
      dma_wait_for_idle();
      MM_ADD(0, t);
    }
    // Hand the next panel to the workers while consuming this one: its chain runs and its dequantization jobs
    // land during the HMX below, which is what the second weight buffer is for. It is submitted right after
    // this panel's first products are issued rather than before them: at the top of the panel the previous
    // panel's last store has just drained HMX, so the ~0.5 us per job of submitting ran with HMX idle --
    // 20 ms per Qwen3.5-4B prefill, whose ffn_down windows hand out 8 jobs for every 11 us of HMX.
    int next_oy_start = weight_panels == 2 ? Q4_PANEL_NEXT(oy_start) : np;
    int next_oy_end   = next_oy_start < np ? Q4_PANEL_NEXT(next_oy_start) : np;
    if (next_oy_end > np) {
      next_oy_end = np;
    }
#define Q4_SUBMIT_NEXT_PANEL()                                                                              \
  do {                                                                                                      \
    if (next_oy_start < np) {                                                                               \
      const unsigned long long t_sub = MM_T0();                                                             \
      q4_panel_submit(&weight_ctx, &panels[weight_buf ^ 1], next_oy_start, next_oy_end, weight_buf ^ 1, 0); \
      MM_ADD(30, t_sub);                                                                                    \
      next_oy_start = np;                                                                                   \
    }                                                                                                       \
  } while (0)

    for (int ox_start = 0; ox_start < mp; ox_start += act_step) {
      // Slot 28 bounds the whole ox body, so slot 28 minus the phases inside it is the unattributed
      // time. Closing that gap is what found the hidden dma_wait_for_idle() last round.
      const unsigned long long t_ox   = MM_T0();
      int                      ox_end = ox_start + act_step;
      if (ox_end > mp) ox_end = mp;

      const int act_buf             = act_pingpong ? (act_chunk_index++ & 1) : 0;
      int       current_base_offset = act_buf * act_buf1_off;

      if (!reuse_activation) {
        const unsigned long long t        = MM_T0();
        int                      act_rows = M - ox_start * 32;
        if (act_rows > (ox_end - ox_start) * 32) {
          act_rows = (ox_end - ox_start) * 32;
        }
        if (act_pingpong) {
          // This chunk was usually staged during the previous chunk's HMX; otherwise stage it now. Then
          // start the next chunk (the next panel's first one after the last) into the other half, which
          // the previous chunk's HMX has finished reading: its last store above waited for it.
          if (!(act_pf_pending && act_pf_ox == ox_start && act_pf_buf == act_buf)) {
            q4_act_stage_submit(act_pf_tasks, &act_pf_sync, act_nw, vtcm_activation, a, current_base_offset, ox_start,
                                ox_end, kp, M, ms, act_blk, act_blk_packs);
          }
          worker_pool_synctoken_wait(&act_pf_sync);
          act_pf_pending = 0;
          MM_BUMP(25, (unsigned long long) act_rows * K * sizeof(__fp16) / 1024);
          const int next_ox = ox_end < mp ? ox_end : (oy_end < np ? 0 : -1);
          if (next_ox >= 0) {
            const int next_end = next_ox + act_step < mp ? next_ox + act_step : mp;
            act_pf_buf         = act_buf ^ 1;
            act_pf_ox          = next_ox;
            act_pf_pending     = 1;
            q4_act_stage_submit(act_pf_tasks, &act_pf_sync, act_nw, vtcm_activation, a, act_pf_buf * act_buf1_off,
                                next_ox, next_end, kp, M, ms, act_blk, act_blk_packs);
          }
        } else if (act_worker_dma) {
          copy_shuffle_q4_activation_task_state_t tasks[Q4_MAX_WEIGHT_CHUNKS];
          worker_synctoken_t                      sync;
          q4_act_stage_submit(tasks, &sync, act_nw, vtcm_activation, a, current_base_offset, ox_start, ox_end, kp, M,
                              ms, act_blk, act_blk_packs);
          worker_pool_synctoken_wait(&sync);
          MM_BUMP(25, (unsigned long long) act_rows * K * sizeof(__fp16) / 1024);
        } else if (act_bulk_dma) {
          // One descriptor per pack. A pack's rows for this chunk are contiguous in the DDR pack-64
          // layout (packs are M * 128 B apart), so each transfer is act_rows * 128 B -- the 4 KB+ rows
          // that make staging in bulk worth it, versus the 64 B a tile-layout destination would force.
          const size_t pack_bytes = (size_t) act_rows * 64 * sizeof(int16_t);
          for (int p = 0; p < act_bulk_packs; ++p) {
            dma_desc_1d_t *d = &act_bulk_descs[p];
            memset(d, 0, sizeof(*d));
            d->next       = 0;
            d->length     = (unsigned) pack_bytes;
            d->type       = DMA_DESC_TYPE_1D;
            d->src_bypass = 0;
            d->dst_bypass = 1;
            d->ordered    = 1;
            d->src        = (uint32_t) (a + ((size_t) p * ms + (size_t) ox_start * 32) * 64 * sizeof(int16_t));
            d->dst        = (uint32_t) ((uint8_t *) vtcm_act_raw + (size_t) p * pack_bytes);
            if (p > 0) {
              act_bulk_descs[p - 1].next = (uint32_t) d;
            }
          }
          dma_wait_for_idle();
          dmstart(&act_bulk_descs[0]);
          // Slot 24 is the DMA alone, so the rest of slot 2 is the HVX shuffle. Without the split there
          // is no way to tell a bandwidth problem from an instruction-bound one, and the two call for
          // opposite fixes.
          const unsigned long long t_dma = MM_T0();
          dma_wait_for_idle();
          MM_ADD(24, t_dma);
          MM_BUMP(25, (unsigned long long) act_bulk_packs * pack_bytes / 1024);
          MM_BUMP(26, act_bulk_packs);
          copy_shuffle_q4_activation_tiles_pack64_parallel(vtcm_activation, (const uint8_t *) vtcm_act_raw,
                                                           current_base_offset, 0, ox_end - ox_start, kp, act_rows,
                                                           act_rows);
        } else {
          // No DMA on this path: the shuffle reads DDR directly, 128 bytes at a time with an M * 128
          // stride, so its traffic is the same but every access is its own DRAM row.
          MM_BUMP(25, (unsigned long long) (ox_end - ox_start) * 32 * kp * 64 * sizeof(int16_t) / 1024);
          copy_shuffle_q4_activation_tiles_pack64_parallel(vtcm_activation, a, current_base_offset, ox_start, ox_end,
                                                           kp, M, ms);
        }
        MM_ADD(2, t);
      }

      if (async_output_store) {
        for (int ox = ox_start; ox < ox_end; ++ox) {
          // One slot per ox while they last, so within a panel no ox ever waits. Across panels the same
          // ox reuses its slot, and the wait below is what keeps that safe.
          const int output_buf_idx = ox % output_slots;
          if (output_store_active[output_buf_idx]) {
            const unsigned long long t = MM_T0();
            worker_pool_synctoken_wait(&output_store_sync[output_buf_idx]);
            MM_ADD(4, t);
            output_store_active[output_buf_idx] = 0;
          }
          __fp16* current_vtcm_output = vtcm_output + (size_t)output_buf_idx * np_chunk * 1024;
          int     ox_offset           = current_base_offset + (ox - ox_start) * kp * 32 * 32;
          Q4_WAIT_ACT_GROUP(ox);
          for (int chunk_idx = 0; chunk_idx < panel->valid_chunks; ++chunk_idx) {
            if (!panel->chunk_ready[chunk_idx]) {
              const unsigned long long t = MM_T0();
              worker_pool_synctoken_wait(&panel->sync[chunk_idx]);
              MM_ADD(1, t);
              panel->chunk_ready[chunk_idx] = 1;
            }
            if (panel->chunk_k1[chunk_idx] != kp) {
              continue;  // an earlier k part of the next chunk's oy
            }
            const int                oy_chunk_start = oy_start + panel->chunk_starts[chunk_idx];
            const int                oy_chunk_end   = oy_chunk_start + panel->chunk_counts[chunk_idx];
            const unsigned long long t_hmx          = MM_T0();
            for (int oy = oy_chunk_start; oy < oy_chunk_end; ++oy) {
              const int oy_offset   = (oy - oy_start) * 16 * kp;
              __fp16   *output_tile = current_vtcm_output + (oy - oy_start) * 1024;
              hmx_load_q4_tiles(vtcm_activation + ox_offset, current_weight + oy_offset * 64, kp);
              hmx_consume_accumulator_fp16(output_tile);
            }
            MM_ADD(3, t_hmx);
            Q4_SUBMIT_NEXT_PANEL();
          }
          worker_pool_synctoken_init(&output_store_sync[output_buf_idx], 1);
          output_store_tasks[output_buf_idx].c           = c;
          output_store_tasks[output_buf_idx].vtcm_output = current_vtcm_output;
          output_store_tasks[output_buf_idx].bias        = bias;
          output_store_tasks[output_buf_idx].vtcm_scales = dequant_in_weight ? NULL : current_vtcm_scales;
          output_store_tasks[output_buf_idx].M           = M;
          output_store_tasks[output_buf_idx].ms          = ms;
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
        }
      } else {
        int pending_ox = -1;
        for (int ox = ox_start; ox < ox_end; ++ox) {
          __fp16 *current_vtcm_output = vtcm_output + (sync_output_pingpong ? (size_t) (ox & 1) * np_chunk * 1024 : 0);
          int     ox_offset           = current_base_offset + (ox - ox_start) * kp * 32 * 32;
          Q4_WAIT_ACT_GROUP(ox);
          for (int chunk_idx = 0; chunk_idx < panel->valid_chunks; ++chunk_idx) {
            if (!panel->chunk_ready[chunk_idx]) {
              const unsigned long long t = MM_T0();
              worker_pool_synctoken_wait(&panel->sync[chunk_idx]);
              MM_ADD(1, t);
              panel->chunk_ready[chunk_idx] = 1;
            }
            if (panel->chunk_k1[chunk_idx] != kp) {
              continue;  // an earlier k part of the next chunk's oy
            }
            const int                oy_chunk_start = oy_start + panel->chunk_starts[chunk_idx];
            const int                oy_chunk_end   = oy_chunk_start + panel->chunk_counts[chunk_idx];
            const unsigned long long t_hmx          = MM_T0();
            for (int oy = oy_chunk_start; oy < oy_chunk_end; ++oy) {
              const int oy_offset   = (oy - oy_start) * 16 * kp;
              __fp16   *output_tile = current_vtcm_output + (oy - oy_start) * 1024;
              hmx_load_q4_tiles(vtcm_activation + ox_offset, current_weight + oy_offset * 64, kp);
              hmx_consume_accumulator_fp16(output_tile);
            }
            MM_ADD(3, t_hmx);
            Q4_SUBMIT_NEXT_PANEL();
          }
          // One store for the whole oy range, once every weight chunk's HMX tiles are in
          // VTCM (the output buffer holds np_chunk tiles, so they all coexist). Storing per
          // weight chunk instead never reaches the paired path in store_q4_output_range,
          // because a chunk is often a single oy and a lone tile owns only half of a pack-64
          // 128B c-slot, which degrades to read-modify-write DDR: 60.4ms vs 11.9ms for the
          // same output on a 1053-token prefill.
          // With the ping-pong buffer this stores the previous row tile, whose products were issued a
          // whole row tile ago, while HMX works through the ones just issued.
          const int store_ox = sync_output_pingpong ? pending_ox : ox;
          if (store_ox >= 0) {
            const unsigned long long t = MM_T0();
            store_q4_output_range(
              c, vtcm_output + (sync_output_pingpong ? (size_t) (store_ox & 1) * np_chunk * 1024 : 0), bias,
              dequant_in_weight ? NULL : current_vtcm_scales, M, ms, store_ox, oy_start, oy_start, oy_end, pack);
            MM_ADD(6, t);
          }
          pending_ox = ox;
        }
        if (sync_output_pingpong && pending_ox >= 0) {
          const unsigned long long t = MM_T0();
          store_q4_output_range(c, vtcm_output + (size_t) (pending_ox & 1) * np_chunk * 1024, bias,
                                dequant_in_weight ? NULL : current_vtcm_scales, M, ms, pending_ox, oy_start, oy_start,
                                oy_end, pack);
          MM_ADD(6, t);
        }
      }
      MM_ADD(28, t_ox);
    }
    Q4_SUBMIT_NEXT_PANEL();  // no products were issued (cannot happen with mp >= 1; kept for safety)
#undef Q4_SUBMIT_NEXT_PANEL
    {
      const unsigned long long t = MM_T0();
      q4_panel_drain(panel);
      MM_ADD(7, t);
    }
  }
  {
    const unsigned long long t = MM_T0();
    if (act_pf_pending) {
      worker_pool_synctoken_wait(&act_pf_sync);
      act_pf_pending = 0;
    }
    for (int g = 0; g < Q4_ACT_GROUPS; ++g) {
      if (act_pending[g]) {
        worker_pool_synctoken_wait(&act_sync[g]);
        act_pending[g] = 0;
      }
    }
    // A lookahead panel is still outstanding when np is not a multiple of the pipeline depth.
    q4_panel_drain(&panels[0]);
    q4_panel_drain(&panels[1]);
    for (int i = 0; i < output_buffers; ++i) {
      if (output_store_active[i]) {
        worker_pool_synctoken_wait(&output_store_sync[i]);
      }
    }
    MM_ADD(8, t);
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  free(act_bulk_descs);
#undef Q4_PANEL_NEXT
  MM_ADD(13, mm_total_start);
  return 0;
}
