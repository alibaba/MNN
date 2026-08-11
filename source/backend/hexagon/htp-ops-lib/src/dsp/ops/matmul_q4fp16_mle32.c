#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "dsp/ops.h"

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

#include "HAP_farf.h"
#include "HAP_perf.h"

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (2 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (2 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)


static const __fp16 q4_to_fp16_lut[64] __attribute__((aligned(128))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

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

static inline HVX_Vector load_q4_output_bias(const uint8_t* bias, int oy) {
  const __fp16* bias_ptr = (const __fp16*)bias + oy * 32;
  HVX_Vector vBias_raw = vmemu(bias_ptr);
  HVX_Vector vBias_rot = Q6_V_vror_VR(vBias_raw, 64);
  return Q6_V_valign_VVR(vBias_raw, vBias_rot, 64);
}

static inline HVX_Vector post_q4_output_vec_nobias(HVX_Vector v, HVX_Vector vScale) {
  return Q6_Vhf_vmpy_VhfVhf(Q6_Vh_vdeal_Vh(v), vScale);
}

static inline HVX_Vector post_q4_output_vec_bias(HVX_Vector v, HVX_Vector vScale, HVX_Vector vBias) {
  v = post_q4_output_vec_nobias(v, vScale);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vBias));
}

static inline void store_single_output_tile_mle32(uint8_t* c, __fp16* vtcm_dst, int M, int oy,
                                                  HVX_VectorPred q_low, HVX_Vector vScale,
                                                  HVX_Vector vBias, int has_bias) {
  const int pack_idx = (oy * 32) / 64;
  const int pack_inner = (oy * 32) & 63;
  HVX_VectorPred q = pack_inner == 0 ? q_low : Q6_Q_not_Q(q_low);
  uint8_t* dst_ptr = c + (size_t)(pack_idx * M) * 128;
  const int num_loops = M / 2;
  const int has_tail = M & 1;

  for (int src_xi = 0; src_xi < num_loops; ++src_xi) {
    HVX_Vector v0 = vmem(vtcm_dst + 64 * src_xi);
    v0 = has_bias ? post_q4_output_vec_bias(v0, vScale, vBias) : post_q4_output_vec_nobias(v0, vScale);
    HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
    HVX_Vector v_first = pack_inner == 0 ? v0 : v0_rot;
    HVX_Vector v_second = pack_inner == 0 ? v0_rot : v0;
    vmem(dst_ptr) = Q6_V_vmux_QVV(q, v_first, vmem(dst_ptr));
    vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, v_second, vmem(dst_ptr + 128));
    dst_ptr += 256;
  }
  if (has_tail) {
    HVX_Vector v0 = vmem(vtcm_dst + 64 * num_loops);
    v0 = has_bias ? post_q4_output_vec_bias(v0, vScale, vBias) : post_q4_output_vec_nobias(v0, vScale);
    if (pack_inner != 0) {
      v0 = Q6_V_valign_VVR(v0, v0, 64);
    }
    vmem(dst_ptr) = Q6_V_vmux_QVV(q, v0, vmem(dst_ptr));
  }
}

static inline void store_output_range_mle32(uint8_t* c, __fp16* vtcm_output, const uint8_t* vtcm_scales,
                                            const uint8_t* bias, int M, int oy_start,
                                            int oy_begin, int oy_end) {
  HVX_VectorPred q_low = Q6_Q_vsetq_R(64);
  const int num_loops = M / 2;
  const int has_tail = M & 1;
  int oy = oy_begin;
  while (oy < oy_end) {
    if (((oy * 32) & 63) == 0 && oy + 1 < oy_end) {
      __fp16* vtcm_dst_0 = vtcm_output + (oy - oy_start) * 1024;
      __fp16* vtcm_dst_1 = vtcm_output + (oy + 1 - oy_start) * 1024;
      HVX_Vector vScale_0 = load_q4_output_scale(vtcm_scales, oy_start, oy);
      HVX_Vector vScale_1 = load_q4_output_scale(vtcm_scales, oy_start, oy + 1);
      int pack_idx = (oy * 32) / 64;
      uint8_t* dst_ptr = c + (size_t)(pack_idx * M) * 128;

      if (bias) {
        HVX_Vector vBias_0 = load_q4_output_bias(bias, oy);
        HVX_Vector vBias_1 = load_q4_output_bias(bias, oy + 1);
        for (int src_xi = 0; src_xi < num_loops; ++src_xi) {
          HVX_Vector v0 = post_q4_output_vec_bias(vmem(vtcm_dst_0 + 64 * src_xi), vScale_0, vBias_0);
          HVX_Vector v1 = post_q4_output_vec_bias(vmem(vtcm_dst_1 + 64 * src_xi), vScale_1, vBias_1);
          HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
          HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
          vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
          vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
          dst_ptr += 256;
        }
        if (has_tail) {
          HVX_Vector v0 = post_q4_output_vec_bias(vmem(vtcm_dst_0 + 64 * num_loops), vScale_0, vBias_0);
          HVX_Vector v1 = post_q4_output_vec_bias(vmem(vtcm_dst_1 + 64 * num_loops), vScale_1, vBias_1);
          HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
          vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
        }
      } else {
        for (int src_xi = 0; src_xi < num_loops; ++src_xi) {
          HVX_Vector v0 = post_q4_output_vec_nobias(vmem(vtcm_dst_0 + 64 * src_xi), vScale_0);
          HVX_Vector v1 = post_q4_output_vec_nobias(vmem(vtcm_dst_1 + 64 * src_xi), vScale_1);
          HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
          HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
          vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
          vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
          dst_ptr += 256;
        }
        if (has_tail) {
          HVX_Vector v0 = post_q4_output_vec_nobias(vmem(vtcm_dst_0 + 64 * num_loops), vScale_0);
          HVX_Vector v1 = post_q4_output_vec_nobias(vmem(vtcm_dst_1 + 64 * num_loops), vScale_1);
          HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
          vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
        }
      }
      oy += 2;
    } else {
      __fp16* vtcm_dst = vtcm_output + (oy - oy_start) * 1024;
      HVX_Vector vScale = load_q4_output_scale(vtcm_scales, oy_start, oy);
      HVX_Vector vBias = bias ? load_q4_output_bias(bias, oy) : Q6_V_vzero();
      store_single_output_tile_mle32(c, vtcm_dst, M, oy, q_low, vScale, vBias, bias != NULL);
      ++oy;
    }
  }
}

typedef struct {
  uint8_t* c;
  __fp16* vtcm_output;
  const uint8_t* vtcm_scales;
  const uint8_t* bias;
  int M;
  int oy_start;
  int oy_begin;
  int oy_end;
  worker_synctoken_t* shared_sync;
} store_output_task_state_t;

static void store_output_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  store_output_task_state_t* state = (store_output_task_state_t*)data;
  store_output_range_mle32(state->c, state->vtcm_output, state->vtcm_scales, state->bias,
                           state->M, state->oy_start, state->oy_begin, state->oy_end);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

typedef struct {
  int oy_start;
  int start_idx;
  int count;
  int kp;
  int scale_block_num;
  int q4block_variant;
  const uint8_t* b_scale;
  const uint8_t* vtcm_weight_int4;
  __fp16* vtcm_weight;
  dma_desc_1d_t* depend;
  worker_synctoken_t* shared_sync;
} process_chunk_task_state_t;

static inline void wait_q4_weight_dma_done(const dma_desc_1d_t* depend) {
  volatile uint32_t *ctrl_word = (volatile uint32_t*)&depend->dstate_order_bypass_type_length;
  while (((*ctrl_word >> 31) & 0x1) == 0) {
    asm volatile("nop");
  }
}

static inline void process_q4_weight_chunk(const process_chunk_task_state_t* state) {
  wait_q4_weight_dma_done(state->depend);

  HVX_Vector vlut_cvt = vmemu(q4_to_fp16_lut);

  const int weight_int4_stride = 512 * state->kp;
  const int weight_fp16_stride = 1024 * state->kp;
  const uint8_t* src_oy = state->vtcm_weight_int4 + state->start_idx * weight_int4_stride;
  __fp16* dst_oy = state->vtcm_weight + state->start_idx * weight_fp16_stride;

  if (state->scale_block_num > 1) {
    const int scale_block_bytes = 128;
    for (int local_oy = 0; local_oy < state->count; ++local_oy) {
        const int oy = state->oy_start + state->start_idx + local_oy;
        const uint8_t* src = src_oy;
        __fp16* dst = dst_oy;
        for (int k = 0; k < state->kp; ++k) {
            const int scale_idx = (k * state->scale_block_num) / state->kp;
            const uint8_t* block_scale_ptr = state->b_scale + ((size_t)oy * state->scale_block_num + scale_idx) * scale_block_bytes;
            const HVX_Vector vBlockScale = vmemu(block_scale_ptr);
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
            src += 512;
            dst += 1024;
        }
        src_oy += weight_int4_stride;
        dst_oy += weight_fp16_stride;
    }
    return;
  }

  for (int local_oy = 0; local_oy < state->count; ++local_oy) {
      const uint8_t* src = src_oy;
      __fp16* dst = dst_oy;
      for (int k = 0; k < state->kp; ++k) {
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

static void process_chunk_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  process_chunk_task_state_t* state = (process_chunk_task_state_t*)data;
  process_q4_weight_chunk(state);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

static inline void hmx_load_q4_mle32_tiles(const __fp16* activation, const __fp16* weight, int kp) {
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

typedef struct {
  uint8_t       *c;
  const uint8_t *a;
  const uint8_t *b;
  const uint8_t *b_scale;
  const uint8_t *bias;
  int            M;
  int            K;
  int            N;
  int            mp_max;
  int            np_max;
  int            kp_max;
  int            scale_block_num;
  int            q4block_variant;
  int            np_chunk;
  int            mp_chunk;
  int            weight_bytes_per_np;
  int            weight_int4_bytes_per_np;
  int            act_bytes_per_mp;
  __fp16        *vtcm_weight;
  uint8_t       *vtcm_weight_int4;
  __fp16        *vtcm_activation;
  __fp16        *vtcm_output;
  __fp16        *vtcm_hmx_scales;
  uint8_t       *vtcm_scales;
} MatmulParam;

static int hmx_matmulq4fp16_mle32_part(const MatmulParam *param) {
  uint8_t       *c                    = param->c;
  const uint8_t *a                    = param->a;
  const uint8_t *b                    = param->b;
  const uint8_t *b_scale              = param->b_scale;
  const uint8_t *bias                 = param->bias;
  int            M                    = param->M;
  int            K                    = param->K;
  int            N                    = param->N;
  int            mp_chunk             = param->mp_chunk;
  int            scale_block_num      = param->scale_block_num;
  int            q4block_variant      = param->q4block_variant;
  const int      dequant_in_weight    = scale_block_num > 1;
  int            np_chunk             = param->np_chunk;
  int            weight_bytes_per_np  = param->weight_bytes_per_np;
  int            weight_int4_bytes_per_np = param->weight_int4_bytes_per_np;
  int            act_bytes_per_mp     = param->act_bytes_per_mp;
  __fp16        *vtcm_weight          = param->vtcm_weight;
  uint8_t       *vtcm_weight_int4     = param->vtcm_weight_int4;
  __fp16        *vtcm_activation      = param->vtcm_activation;
  __fp16        *vtcm_output          = param->vtcm_output;
  __fp16        *vtcm_hmx_scales      = param->vtcm_hmx_scales;
  uint8_t       *vtcm_scales          = param->vtcm_scales;
  (void) mp_chunk;
  (void) weight_bytes_per_np;
  (void) weight_int4_bytes_per_np;
  (void) act_bytes_per_mp;

  hmx_manager_enable_execution();
  hmx_unit_acquire();
  hmx_init_column_scales(vtcm_hmx_scales, Q6_V_vsplat_R(0x3c00));
  if (!q4block_variant) {
    hmx_set_output_scales(vtcm_hmx_scales);
  }

  int pack = 64;
#ifdef __HVX_LENGTH__
  pack = __HVX_LENGTH__ / (int32_t)sizeof(int16_t);
#endif

  int np = N / 32;
  int kp = K / 32;
  int act_treat = 0;
  int tileCount = (np + np_chunk - 1) / np_chunk;

  for (int yo = 0; yo < tileCount; ++yo) {
    int oy_start = yo * np_chunk;
    int oy_end = oy_start + np_chunk;
    if (oy_end > np) oy_end = np;

    int weight_dma_count = oy_end - oy_start;
    int NUM_CHUNKS = g_max_num_workers;
    if (NUM_CHUNKS > weight_dma_count) {
      NUM_CHUNKS = weight_dma_count;
    }
    if (NUM_CHUNKS < 1) {
      NUM_CHUNKS = 1;
    }
    int async_store = weight_dma_count > 32;

    int chunk_size = (weight_dma_count + NUM_CHUNKS - 1) / NUM_CHUNKS;
    if (chunk_size > 1) {
      chunk_size = (chunk_size + 1) & ~1;
    }
    int chunk_counts[NUM_CHUNKS];
    int chunk_starts[NUM_CHUNKS];
    int current_start = 0;
    int validChunk = 0;
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        chunk_starts[i] = current_start;
        int end = current_start + chunk_size;
        if (end >= weight_dma_count) {
          end = weight_dma_count;
        }
        chunk_counts[i] = end - current_start;
        validChunk++;
        current_start = end;
        if (current_start >= weight_dma_count) {
          break;
        }
    }

    _Alignas(64) dma_desc_1d_t weight_desc[NUM_CHUNKS];
    _Alignas(64) dma_desc_2d_t scale_desc[1];

    #define SET_WEIGHT_DMA(start_idx, count, desc_idx, next_ptr, pre_index) do { \
        { \
            if (pre_index >= 0) weight_desc[pre_index].next = (uint32_t)&weight_desc[desc_idx]; \
            memset(&weight_desc[desc_idx], 0, sizeof(dma_desc_1d_t)); \
            weight_desc[desc_idx].next       = next_ptr; \
            weight_desc[desc_idx].length     = 16 * 32 * kp * (count); \
            weight_desc[desc_idx].type       = DMA_DESC_TYPE_1D; \
            weight_desc[desc_idx].src_bypass = 1; \
            weight_desc[desc_idx].dst_bypass = 1; \
            weight_desc[desc_idx].ordered    = 1; \
            weight_desc[desc_idx].dstate     = DMA_DESC_DSTATE_PENDING; \
            weight_desc[desc_idx].src        = (uint32_t) (b + (size_t)(oy_start + (start_idx)) * kp * 32 * 16); \
            weight_desc[desc_idx].dst        = (uint32_t) (vtcm_weight_int4 + (start_idx) * 512 * kp); \
        } \
    } while(0)

    // Pipeline with N chunks
    // 1. Issue DMA for chunk 0
    SET_WEIGHT_DMA(chunk_starts[0], chunk_counts[0], 0, 0, -1);

    int safe_kp = kp > 0 ? kp : 1;
    if (!dequant_in_weight) {
      memset(&scale_desc[0], 0, sizeof(dma_desc_2d_t));
      scale_desc[0].type       = DMA_DESC_TYPE_2D;
      scale_desc[0].src_bypass = 1;
      scale_desc[0].dst_bypass = 1;
      scale_desc[0].ordered    = 1;
      scale_desc[0].src        = (uint32_t) (b_scale + (oy_start) * 32 * sizeof(__fp16));
      scale_desc[0].dst        = (uint32_t) (vtcm_scales);
      scale_desc[0].roi_width  = 32 * sizeof(__fp16);
      scale_desc[0].roi_height = weight_dma_count;
      scale_desc[0].src_stride = 32 * sizeof(__fp16);
      scale_desc[0].dst_stride = 64;
      scale_desc[0].next       = 0;
    }
    _Alignas(64) dma_desc_2d_t act_descs[safe_kp];
    if (act_treat == 0) {
      if (M > 1) {
        for (int k = 0; k < kp; ++k) {
          int pack_idx = (k * 32) / pack;
          int pack_inner = (k * 32) % pack;
          uint8_t* dst_addr = (uint8_t*)(vtcm_activation + k * 32 * 32);
          size_t src_offset = (size_t)(pack_idx * M) * pack * sizeof(int16_t) + pack_inner * sizeof(int16_t);
          const uint8_t* src_addr = a + src_offset;
          memset(&act_descs[k], 0, sizeof(dma_desc_2d_t));
          act_descs[k].type       = DMA_DESC_TYPE_2D;
          act_descs[k].src_bypass = 0;
          act_descs[k].dst_bypass = 1;
          act_descs[k].ordered    = 1;
          act_descs[k].src              = (uint32_t) src_addr;
          act_descs[k].dst              = (uint32_t) dst_addr;
          act_descs[k].roi_width        = 32 * sizeof(int16_t);
          act_descs[k].roi_height       = M;
          act_descs[k].src_stride       = pack * sizeof(int16_t);
          act_descs[k].dst_stride       = 32 * sizeof(int16_t);
          if (k > 0) {
            act_descs[k - 1].next = (uint32_t)&act_descs[k];
          }
          if (k == kp - 1 && !dequant_in_weight) {
            act_descs[k].next = (uint32_t)&scale_desc[0];
          }
        }
      } else {
        int k = 0;
        uint8_t* dst_addr = (uint8_t*)(vtcm_activation);
        size_t src_offset = 0;
        const uint8_t* src_addr = a + src_offset;
        memset(&act_descs[k], 0, sizeof(dma_desc_2d_t));
        act_descs[k].type       = DMA_DESC_TYPE_2D;
        act_descs[k].src_bypass = 0;
        act_descs[k].dst_bypass = 1;
        act_descs[k].ordered    = 1;
        act_descs[k].src              = (uint32_t) src_addr;
        act_descs[k].dst              = (uint32_t) dst_addr;
        act_descs[k].roi_width        = 32 * sizeof(int16_t);
        act_descs[k].roi_height       = kp;
        act_descs[k].src_stride       = 32 * sizeof(int16_t);
        act_descs[k].dst_stride       = 32 * 32 * sizeof(int16_t);
        act_descs[k].next = !dequant_in_weight ? (uint32_t)&scale_desc[0] : 0;
      }
    }
    uint32_t act_dma_ptr = 0;
    if (act_treat == 0 && M > 0 && kp > 0) {
      act_dma_ptr = (uint32_t)&act_descs[0];
    } else if (!dequant_in_weight) {
      act_dma_ptr = (uint32_t)(&scale_desc[0]);
    }
    for (int i = 1; i < validChunk; ++i) {
      SET_WEIGHT_DMA(chunk_starts[i], chunk_counts[i], i, 0, i-1);
    }
    weight_desc[validChunk - 1].next = act_dma_ptr;

    dmstart(&weight_desc[0]);
    worker_synctoken_t sync_token[NUM_CHUNKS];
    process_chunk_task_state_t chunk_states[NUM_CHUNKS];
    worker_synctoken_t store_sync_token[NUM_CHUNKS];
    store_output_task_state_t store_states[NUM_CHUNKS];

    for (int i = 0; i < validChunk; ++i) {// Process LUT for chunk i
        worker_pool_synctoken_init(&sync_token[i], 1);
        chunk_states[i].oy_start = oy_start;
        chunk_states[i].start_idx = chunk_starts[i];
        chunk_states[i].count = chunk_counts[i];
        chunk_states[i].kp = kp;
        chunk_states[i].scale_block_num = scale_block_num;
        chunk_states[i].q4block_variant = q4block_variant;
        chunk_states[i].b_scale = b_scale;
        chunk_states[i].vtcm_weight_int4 = vtcm_weight_int4;
        chunk_states[i].vtcm_weight = vtcm_weight;
        chunk_states[i].depend = &weight_desc[i];
        chunk_states[i].shared_sync = &sync_token[i];

        worker_pool_job_t job;
        job.fptr = process_chunk_worker_loop;
        job.dptr = &chunk_states[i];
        worker_pool_submit(NULL, job);
    }

    #undef SET_WEIGHT_DMA

    dma_wait_for_idle();
    {
      if (act_treat == 0) {
        act_treat = 1;
        int act_vecs_per_k = (M + 1) / 2;
        if (act_vecs_per_k > 16) {
          act_vecs_per_k = 16;
        }
        for (int k = 0; k < kp; ++k) {
          __fp16* act_tile = vtcm_activation + k * 1024;
          for (int xi = 0; xi < act_vecs_per_k; ++xi) {
            HVX_Vector* va =  (HVX_Vector*)(act_tile + xi * 64);
            va[0] = Q6_Vh_vshuff_Vh(va[0]);
          }
        }
      }
      if (q4block_variant) {
        hmx_set_output_scales(vtcm_hmx_scales);
      }
      for (int i = 0; i < validChunk; ++i) {// Process LUT for chunk i
        int start_idx = chunk_starts[i];
        int count = chunk_counts[i];
        worker_pool_synctoken_wait(&sync_token[i]);

        __fp16* weight_tile = vtcm_weight + start_idx * 1024 * kp;
        __fp16* output_tile = vtcm_output + start_idx * 1024;
        for (int local_oy = 0; local_oy < count; ++local_oy) {
          hmx_load_q4_mle32_tiles(vtcm_activation, weight_tile, kp);
          __fp16* vtcm_dst = output_tile;
          hmx_consume_accumulator_fp16(vtcm_dst);
          weight_tile += 1024 * kp;
          output_tile += 1024;
        }
        if (async_store) {
          worker_pool_synctoken_init(&store_sync_token[i], 1);
          store_states[i].c = c;
          store_states[i].vtcm_output = vtcm_output;
          store_states[i].vtcm_scales = vtcm_scales;
          store_states[i].bias = bias;
          store_states[i].M = M;
          store_states[i].oy_start = oy_start;
          store_states[i].oy_begin = oy_start + start_idx;
          store_states[i].oy_end = oy_start + start_idx + count;
          store_states[i].shared_sync = &store_sync_token[i];

          worker_pool_job_t store_job;
          store_job.fptr = store_output_worker_loop;
          store_job.dptr = &store_states[i];
          worker_pool_submit(NULL, store_job);
        } else {
          store_output_range_mle32(c, vtcm_output, vtcm_scales, bias, M, oy_start,
                                   oy_start + start_idx, oy_start + start_idx + count);
        }
      }
      if (async_store) {
        for (int i = 0; i < validChunk; ++i) {
          worker_pool_synctoken_wait(&store_sync_token[i]);
        }
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_matmulq4fp16_mle32_entry(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale,
                                        const uint8_t * bias, int M, int K, int N, int mp_max, int np_max,
                                        int kp_max, int scale_block_num, int scale_asymmetric,
                                        int q4block_variant) {
  if (scale_block_num <= 0) scale_block_num = 1;
  (void)scale_asymmetric;
  const int dequant_in_weight = scale_block_num > 1;
  MatmulParam param = {
    .c = c,
    .a = a,
    .b = b,
    .b_scale = b_scale,
    .bias = bias,
    .M = M,
    .K = K,
    .N = N,
    .mp_max = mp_max,
    .np_max = np_max,
    .kp_max = kp_max,
    .scale_block_num = scale_block_num,
    .q4block_variant = q4block_variant,
  };

  param.np_chunk = param.np_max;
  if (param.np_chunk == 0) param.np_chunk = 1;
  param.mp_chunk = param.mp_max;
  if (param.mp_chunk == 0) param.mp_chunk = 1;
  if (param.np_chunk > 1 && param.np_chunk % 2!=0) param.np_chunk -= 1;

  param.weight_bytes_per_np = 32 * param.K * sizeof(int16_t);
  param.weight_int4_bytes_per_np = 32 * param.K / 2;
  param.act_bytes_per_mp = 32 * param.K * sizeof(int16_t);

  uint8_t *vtcm_ptr = (uint8_t *) vtcm_manager_get_vtcm_base();
  param.vtcm_weight = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * param.weight_bytes_per_np);
  param.vtcm_weight_int4 = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * param.weight_int4_bytes_per_np);
  param.vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.mp_chunk * param.act_bytes_per_mp);
  param.vtcm_output = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.mp_chunk * param.np_chunk * OUTPUT_AREA_SIZE);
  param.vtcm_hmx_scales = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  param.vtcm_scales = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * 64 + 64);
  memset(param.vtcm_scales, 0, param.np_chunk * 64 + 64);
  if (dequant_in_weight) {
    init_identity_q4_output_scales(param.vtcm_scales, param.np_chunk);
  }

  return hmx_matmulq4fp16_mle32_part(&param);
}

int hmx_matmulq4fp16_mle32(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale, const uint8_t * bias,
                           int M, int K, int N, int mp_max, int np_max, int kp_max, int scale_block_num,
                           int scale_asymmetric) {
  return hmx_matmulq4fp16_mle32_entry(c, a, b, b_scale, bias, M, K, N, mp_max, np_max, kp_max,
                                      scale_block_num, scale_asymmetric, 0);
}
