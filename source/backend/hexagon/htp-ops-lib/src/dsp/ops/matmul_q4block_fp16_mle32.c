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

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (2 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (2 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)

static const __fp16 q4_to_fp16_lut[64] __attribute__((aligned(128))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static volatile uint32_t g_q4block_scale_touch_sink = 0;

static inline void touch_q4block_scale_buffer_once(const uint8_t* scale_buffer, size_t bytes) {
  enum { MAX_TOUCHED_SCALE_BUFFERS = 2048 };
  static const uint8_t* touched_scale_buffers[MAX_TOUCHED_SCALE_BUFFERS];
  static int touched_scale_buffer_count = 0;
  for (int i = 0; i < touched_scale_buffer_count; ++i) {
    if (touched_scale_buffers[i] == scale_buffer) {
      return;
    }
  }
  if (touched_scale_buffer_count < MAX_TOUCHED_SCALE_BUFFERS) {
    touched_scale_buffers[touched_scale_buffer_count++] = scale_buffer;
  }
  uint32_t acc = 0;
  for (size_t off = 0; off + sizeof(uint32_t) <= bytes; off += sizeof(uint32_t)) {
    acc ^= *(const volatile uint32_t*)(scale_buffer + off);
  }
  if (bytes > 0) {
    acc ^= *(const volatile uint32_t*)(scale_buffer + bytes - sizeof(uint32_t));
  }
  g_q4block_scale_touch_sink ^= acc;
}

static inline HVX_Vector load_q4_output_bias(const uint8_t* bias, int oy) {
  const __fp16* bias_ptr = (const __fp16*)bias + oy * 32;
  HVX_Vector vBias_raw = vmemu(bias_ptr);
  HVX_Vector vBias_rot = Q6_V_vror_VR(vBias_raw, 64);
  return Q6_V_valign_VVR(vBias_raw, vBias_rot, 64);
}

static inline HVX_Vector post_q4_output_vec_identity_nobias(HVX_Vector v) {
  return Q6_Vh_vdeal_Vh(v);
}

static inline HVX_Vector post_q4_output_vec_identity_bias(HVX_Vector v, HVX_Vector vBias) {
  v = post_q4_output_vec_identity_nobias(v);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vBias));
}

typedef struct {
  __fp16* reduced_output;
  __fp16* vtcm_output;
  __fp16* vtcm_packed_scales;
  int oy_start;
  int oy_begin;
  int oy_end;
  int scale_pair_count;
  int clear_output;
  worker_synctoken_t* shared_sync;
} reduce_q4block_task_state_t;

typedef struct {
  int start_idx;
  int count;
  int kp;
  const uint8_t* vtcm_weight_int4;
  __fp16* vtcm_weight;
  dma_desc_1d_t* depend;
  worker_synctoken_t* shared_sync;
} process_chunk_task_state_t;

static inline int matmul_q4block_scale_output_passes(int scale_block_num) {
  if (scale_block_num <= 1) {
    return 0;
  }
  return (scale_block_num + 31) / 32;
}

static inline void wait_q4_weight_dma_done(const dma_desc_1d_t* depend) {
  volatile uint32_t* ctrl_word = (volatile uint32_t*)&depend->dstate_order_bypass_type_length;
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
  if (kp == 48) {
    hmx_load_tiles_fp16(activation, weight, 32);
    hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 16);
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

static inline void cache_q4block_m1_activation_rows(__fp16* activation_cache, const __fp16* activation, int kp) {
  const HVX_Vector vEvenMask = Q6_V_vsplat_R(0x0000ffff);
  HVX_Vector* cache_vec = (HVX_Vector*)activation_cache;
  for (int k = 0; k < kp; ++k) {
    const __fp16* act_tile = activation + k * 1024;
    cache_vec[k] = Q6_V_vand_VV(Q6_Vh_vshuff_Vh(vmem(act_tile)), vEvenMask);
  }
}

static inline void prepare_q4block_m1_scale_rows_cached(__fp16* activation, const __fp16* activation_cache,
                                                        int kp, int scale_block_num,
                                                        int scale_begin, int scale_count) {
  const HVX_Vector vZero = Q6_V_vzero();
  const HVX_Vector* cache_vec = (const HVX_Vector*)activation_cache;
  const int k_per_scale = kp / scale_block_num;
  for (int local_scale_idx = 0; local_scale_idx < scale_count; ++local_scale_idx) {
    const int scale_idx = scale_begin + local_scale_idx;
    const int k_begin = scale_idx * k_per_scale;
    const int k_end = k_begin + k_per_scale;
    const int dst_vec_idx = local_scale_idx >> 1;
    const int use_high_half = local_scale_idx & 1;
    for (int k = k_begin; k < k_end; ++k) {
      HVX_Vector* act_vec = (HVX_Vector*)(activation + k * 1024);
      for (int i = 0; i < 16; ++i) {
        act_vec[i] = vZero;
      }
      act_vec[dst_vec_idx] = use_high_half ? Q6_Vw_vasl_VwR(cache_vec[k], 16) : cache_vec[k];
    }
  }
}

static inline void accumulate_q4block_m1_packed_sum_range(__fp16* reduced_output, __fp16* vtcm_output,
                                                              __fp16* vtcm_packed_scales, int oy_start,
                                                              int oy_begin, int oy_end,
                                                              int scale_pair_count, int clear_output) {
  for (int oy = oy_begin; oy < oy_end; ++oy) {
    __fp16* vtcm_dst = vtcm_output + (oy - oy_start) * 1024;
    HVX_Vector* reduced_vec = (HVX_Vector*)(reduced_output + (oy - oy_start) * 64);
    HVX_Vector v = clear_output ? Q6_V_vzero() : reduced_vec[0];
    const HVX_Vector* partial_vec = (const HVX_Vector*)vtcm_dst;
    const HVX_Vector* scale_vec = (const HVX_Vector*)(vtcm_packed_scales + (oy - oy_begin) * 1024);
    if (scale_pair_count == 16) {
#define ACC_Q4BLOCK_PAIR(IDX) do { \
        HVX_Vector vScaled = Q6_Vhf_vmpy_VhfVhf(partial_vec[(IDX)], scale_vec[(IDX)]); \
        v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vScaled)); \
      } while (0)
      ACC_Q4BLOCK_PAIR(0);
      ACC_Q4BLOCK_PAIR(1);
      ACC_Q4BLOCK_PAIR(2);
      ACC_Q4BLOCK_PAIR(3);
      ACC_Q4BLOCK_PAIR(4);
      ACC_Q4BLOCK_PAIR(5);
      ACC_Q4BLOCK_PAIR(6);
      ACC_Q4BLOCK_PAIR(7);
      ACC_Q4BLOCK_PAIR(8);
      ACC_Q4BLOCK_PAIR(9);
      ACC_Q4BLOCK_PAIR(10);
      ACC_Q4BLOCK_PAIR(11);
      ACC_Q4BLOCK_PAIR(12);
      ACC_Q4BLOCK_PAIR(13);
      ACC_Q4BLOCK_PAIR(14);
      ACC_Q4BLOCK_PAIR(15);
#undef ACC_Q4BLOCK_PAIR
    } else {
      for (int pair_idx = 0; pair_idx < scale_pair_count; ++pair_idx) {
        HVX_Vector vScaled = Q6_Vhf_vmpy_VhfVhf(partial_vec[pair_idx], scale_vec[pair_idx]);
        v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vScaled));
      }
    }
    reduced_vec[0] = v;
  }
}

static inline HVX_Vector finish_q4block_m1_packed_sum_vec(HVX_Vector v) {
  const HVX_Vector vEvenMask = Q6_V_vsplat_R(0x0000ffff);
  HVX_Vector vHigh = Q6_Vuw_vlsr_VuwR(v, 16);
  v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, vHigh));
  return Q6_V_vand_VV(v, vEvenMask);
}

static inline HVX_Vector post_q4block_m1_reduced_vec(HVX_Vector v, const uint8_t* bias, int oy) {
  if (bias) {
    return post_q4_output_vec_identity_bias(v, load_q4_output_bias(bias, oy));
  }
  return post_q4_output_vec_identity_nobias(v);
}

static inline void store_q4block_m1_reduced_output_range(uint8_t* c, __fp16* reduced_output,
                                                         const uint8_t* bias, int oy_start,
                                                         int oy_begin, int oy_end) {
  HVX_VectorPred q_low = Q6_Q_vsetq_R(64);
  int oy = oy_begin;
  while (oy < oy_end) {
    if (((oy * 32) & 63) == 0 && oy + 1 < oy_end) {
      HVX_Vector v0 = finish_q4block_m1_packed_sum_vec(vmem(reduced_output + (oy - oy_start) * 64));
      HVX_Vector v1 = finish_q4block_m1_packed_sum_vec(vmem(reduced_output + (oy + 1 - oy_start) * 64));
      v0 = post_q4block_m1_reduced_vec(v0, bias, oy);
      v1 = post_q4block_m1_reduced_vec(v1, bias, oy + 1);
      HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
      int pack_idx = (oy * 32) / 64;
      uint8_t* dst_ptr = c + (size_t)pack_idx * 128;
      vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
      ++oy;
    } else {
      HVX_Vector v = finish_q4block_m1_packed_sum_vec(vmem(reduced_output + (oy - oy_start) * 64));
      v = post_q4block_m1_reduced_vec(v, bias, oy);
      const int pack_idx = (oy * 32) / 64;
      const int pack_inner = (oy * 32) & 63;
      HVX_VectorPred q = pack_inner == 0 ? q_low : Q6_Q_not_Q(q_low);
      if (pack_inner != 0) {
        v = Q6_V_valign_VVR(v, v, 64);
      }
      uint8_t* dst_ptr = c + (size_t)pack_idx * 128;
      vmem(dst_ptr) = Q6_V_vmux_QVV(q, v, vmem(dst_ptr));
    }
    ++oy;
  }
}

static void reduce_q4block_worker_loop(void* data, int _worker_index) {
  (void)_worker_index;
  reduce_q4block_task_state_t* state = (reduce_q4block_task_state_t*)data;
  accumulate_q4block_m1_packed_sum_range(state->reduced_output, state->vtcm_output, state->vtcm_packed_scales,
                                         state->oy_start, state->oy_begin, state->oy_end,
                                         state->scale_pair_count, state->clear_output);
  worker_pool_synctoken_jobdone(state->shared_sync);
}

typedef struct {
  uint8_t       *c;
  const uint8_t *a;
  const uint8_t *b;
  const uint8_t *b_scale;
  const uint8_t *bias;
  int            K;
  int            N;
  int            scale_block_num;
  int            np_chunk;
  int            act_bytes_per_mp;
  __fp16        *vtcm_weight;
  uint8_t       *vtcm_weight_int4;
  __fp16        *vtcm_activation;
  __fp16        *vtcm_activation_cache;
  __fp16        *vtcm_output;
  __fp16        *vtcm_reduced_output;
  __fp16        *vtcm_packed_scales;
  __fp16        *vtcm_hmx_scales;
} MatmulParam;

static int hmx_matmulq4fp16_mle32_part(const MatmulParam *param) {
  uint8_t       *c                    = param->c;
  const uint8_t *a                    = param->a;
  const uint8_t *b                    = param->b;
  const uint8_t *b_scale              = param->b_scale;
  const uint8_t *bias                 = param->bias;
  int            K                    = param->K;
  int            N                    = param->N;
  int            scale_block_num      = param->scale_block_num;
  const int      scale_output_passes  = matmul_q4block_scale_output_passes(scale_block_num);
  int            np_chunk             = param->np_chunk;
  int            act_bytes_per_mp     = param->act_bytes_per_mp;
  __fp16        *vtcm_weight          = param->vtcm_weight;
  uint8_t       *vtcm_weight_int4     = param->vtcm_weight_int4;
  __fp16        *vtcm_activation      = param->vtcm_activation;
  __fp16        *vtcm_activation_cache = param->vtcm_activation_cache;
  __fp16        *vtcm_output          = param->vtcm_output;
  __fp16        *vtcm_reduced_output  = param->vtcm_reduced_output;
  __fp16        *vtcm_packed_scales   = param->vtcm_packed_scales;
  __fp16        *vtcm_hmx_scales      = param->vtcm_hmx_scales;

  hmx_manager_enable_execution();
  hmx_unit_acquire();
  hmx_init_column_scales(vtcm_hmx_scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(vtcm_hmx_scales);

  int np = N / 32;
  int kp = K / 32;
  int act_treat = 0;
  int tileCount = (np + np_chunk - 1) / np_chunk;
  const int scale_pair_num_all = (scale_block_num + 1) >> 1;
  const uint8_t* host_packed_scales_base = b_scale + (size_t)np * scale_block_num * 128;
  touch_q4block_scale_buffer_once(b_scale, (size_t)np * scale_block_num * 128);
  touch_q4block_scale_buffer_once(host_packed_scales_base, (size_t)np * scale_pair_num_all * 128);

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

    _Alignas(64) dma_desc_2d_t act_desc;
    if (act_treat == 0) {
      memset(&act_desc, 0, sizeof(act_desc));
      act_desc.type       = DMA_DESC_TYPE_2D;
      act_desc.src_bypass = 0;
      act_desc.dst_bypass = 1;
      act_desc.ordered    = 1;
      act_desc.src        = (uint32_t)a;
      act_desc.dst        = (uint32_t)vtcm_activation;
      act_desc.roi_width  = 32 * sizeof(int16_t);
      act_desc.roi_height = kp;
      act_desc.src_stride = 32 * sizeof(int16_t);
      act_desc.dst_stride = 32 * 32 * sizeof(int16_t);
      act_desc.next       = 0;
    }
    uint32_t act_dma_ptr = 0;
    if (act_treat == 0 && kp > 0) {
      memset(vtcm_activation, 0, (size_t)act_bytes_per_mp);
      act_dma_ptr = (uint32_t)&act_desc;
    }
    for (int i = 1; i < validChunk; ++i) {
      SET_WEIGHT_DMA(chunk_starts[i], chunk_counts[i], i, 0, i-1);
    }

    weight_desc[validChunk - 1].next = act_dma_ptr;
    dmstart(&weight_desc[0]);

    worker_synctoken_t sync_token[NUM_CHUNKS];
    process_chunk_task_state_t chunk_states[NUM_CHUNKS];
    worker_synctoken_t reduce_sync_token[NUM_CHUNKS];
    reduce_q4block_task_state_t reduce_states[NUM_CHUNKS];
    int chunk_weight_ready[NUM_CHUNKS];
    for (int i = 0; i < validChunk; ++i) {
        chunk_weight_ready[i] = 0;
        worker_pool_synctoken_init(&sync_token[i], 1);
        chunk_states[i].start_idx = chunk_starts[i];
        chunk_states[i].count = chunk_counts[i];
        chunk_states[i].kp = kp;
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
        cache_q4block_m1_activation_rows(vtcm_activation_cache, vtcm_activation, kp);
      }
      __fp16* vtcm_reduced_output_block = vtcm_reduced_output;
      int reduce_pending[NUM_CHUNKS];
      for (int i = 0; i < validChunk; ++i) {
        reduce_pending[i] = 0;
      }
      for (int pass = 0; pass < scale_output_passes; ++pass) {
        const int scale_begin = pass * 32;
        int scale_count = scale_block_num - scale_begin;
        if (scale_count > 32) {
          scale_count = 32;
        }
        const int k_per_scale = kp / scale_block_num;
        const int k_begin = scale_begin * k_per_scale;
        const int k_count = scale_count * k_per_scale;
        const int scale_pair_count = (scale_count + 1) >> 1;
        prepare_q4block_m1_scale_rows_cached(vtcm_activation, vtcm_activation_cache, kp,
                                             scale_block_num, scale_begin, scale_count);
        __fp16* vtcm_output_pass = vtcm_output + (size_t)pass * np_chunk * 1024;
        __fp16* vtcm_packed_scales_pass = vtcm_packed_scales + (size_t)(pass & 1) * np_chunk * 1024;
        const int scale_pair_num = (scale_block_num + 1) >> 1;
        const int scale_pair_begin = scale_begin >> 1;
        _Alignas(64) dma_desc_2d_t scale_desc;
        memset(&scale_desc, 0, sizeof(scale_desc));
        scale_desc.type       = DMA_DESC_TYPE_2D;
        scale_desc.src_bypass = 1;
        scale_desc.dst_bypass = 1;
        scale_desc.ordered    = 1;
        scale_desc.src        = (uint32_t)(host_packed_scales_base + ((size_t)oy_start * scale_pair_num + scale_pair_begin) * 128);
        scale_desc.dst        = (uint32_t)vtcm_packed_scales_pass;
        scale_desc.roi_width  = scale_pair_count * 128;
        scale_desc.roi_height = weight_dma_count;
        scale_desc.src_stride = scale_pair_num * 128;
        scale_desc.dst_stride = 1024 * sizeof(int16_t);
        scale_desc.next       = 0;
        dmstart(&scale_desc);
        int scale_dma_pending = 1;
        for (int i = 0; i < validChunk; ++i) {
          int start_idx = chunk_starts[i];
          int count = chunk_counts[i];
          if (reduce_pending[i]) {
            worker_pool_synctoken_wait(&reduce_sync_token[i]);
            reduce_pending[i] = 0;
          }
          if (!chunk_weight_ready[i]) {
            worker_pool_synctoken_wait(&sync_token[i]);
            chunk_weight_ready[i] = 1;
          }

          __fp16* weight_tile = vtcm_weight + start_idx * 1024 * kp;
          __fp16* chunk_output = vtcm_output_pass + start_idx * 1024;
          __fp16* chunk_scales = vtcm_packed_scales_pass + start_idx * 1024;
          __fp16* output_tile = chunk_output;
          for (int local_oy = 0; local_oy < count; ++local_oy) {
            hmx_load_q4_mle32_tiles(vtcm_activation + k_begin * 1024,
                                    weight_tile + k_begin * 1024, k_count);
            hmx_consume_accumulator_fp16(output_tile);
            weight_tile += 1024 * kp;
            output_tile += 1024;
          }
          if (scale_dma_pending) {
            dma_wait_for_idle();
            scale_dma_pending = 0;
          }
          worker_pool_synctoken_init(&reduce_sync_token[i], 1);
          reduce_states[i].reduced_output = vtcm_reduced_output_block + start_idx * 64;
          reduce_states[i].vtcm_output = chunk_output;
          reduce_states[i].vtcm_packed_scales = chunk_scales;
          reduce_states[i].oy_start = oy_start + start_idx;
          reduce_states[i].oy_begin = oy_start + start_idx;
          reduce_states[i].oy_end = oy_start + start_idx + count;
          reduce_states[i].scale_pair_count = scale_pair_count;
          reduce_states[i].clear_output = pass == 0;
          reduce_states[i].shared_sync = &reduce_sync_token[i];

          worker_pool_job_t reduce_job;
          reduce_job.fptr = reduce_q4block_worker_loop;
          reduce_job.dptr = &reduce_states[i];
          worker_pool_submit(NULL, reduce_job);
          reduce_pending[i] = 1;
        }
      }
      for (int i = 0; i < validChunk; ++i) {
        if (reduce_pending[i]) {
          worker_pool_synctoken_wait(&reduce_sync_token[i]);
        }
      }
      for (int i = 0; i < validChunk; ++i) {
        int start_idx = chunk_starts[i];
        int count = chunk_counts[i];
        store_q4block_m1_reduced_output_range(c, vtcm_reduced_output_block + start_idx * 64, bias,
                                              oy_start + start_idx,
                                              oy_start + start_idx, oy_start + start_idx + count);
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_matmulq4fp16_mle32_entry(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale,
                                        const uint8_t * bias, int M, int K, int N, int mp_max, int np_max,
                                        int kp_max, int scale_block_num, int scale_asymmetric) {
  (void)M;
  (void)mp_max;
  if (scale_block_num <= 1) return -1;
  (void)scale_asymmetric;
  const int scale_output_passes = matmul_q4block_scale_output_passes(scale_block_num);
  MatmulParam param = {
    .c = c,
    .a = a,
    .b = b,
    .b_scale = b_scale,
    .bias = bias,
    .K = K,
    .N = N,
    .scale_block_num = scale_block_num,
  };

  param.np_chunk = np_max;
  if (param.np_chunk == 0) param.np_chunk = 1;
  if (param.np_chunk > 1 && param.np_chunk % 2!=0) param.np_chunk -= 1;

  const int weight_bytes_per_np = 32 * param.K * sizeof(int16_t);
  const int weight_int4_bytes_per_np = 32 * param.K / 2;
  param.act_bytes_per_mp = 32 * param.K * sizeof(int16_t);

  uint8_t *vtcm_ptr = (uint8_t *) vtcm_manager_get_vtcm_base();
  param.vtcm_weight = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * weight_bytes_per_np);
  param.vtcm_weight_int4 = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * weight_int4_bytes_per_np);
  param.vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.act_bytes_per_mp);
  param.vtcm_activation_cache = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, kp_max * 128);
  int output_partition_count = scale_output_passes > 1 ? scale_output_passes : 1;
  param.vtcm_output = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr,
                                                param.np_chunk * OUTPUT_AREA_SIZE * output_partition_count);
  param.vtcm_reduced_output = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * 128);
  int scale_partition_count = scale_output_passes > 1 ? 2 : 1;
  param.vtcm_packed_scales = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, param.np_chunk * OUTPUT_AREA_SIZE *
                                                       scale_partition_count);
  param.vtcm_hmx_scales = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  return hmx_matmulq4fp16_mle32_part(&param);
}

int hmx_matmulq4blockfp16_mle32(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale,
                                const uint8_t * bias, int M, int K, int N, int mp_max, int np_max, int kp_max,
                                int scale_block_num, int scale_asymmetric) {
  return hmx_matmulq4fp16_mle32_entry(c, a, b, b_scale, bias, M, K, N, mp_max, np_max, kp_max,
                                      scale_block_num, scale_asymmetric);
}
