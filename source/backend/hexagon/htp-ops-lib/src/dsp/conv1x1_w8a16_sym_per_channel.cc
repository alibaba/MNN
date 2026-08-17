#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_utils.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "im2col_convolution_fp16_internal.h"

static inline void apply_output_post_fp16_w8(HVX_Vector* v, HVX_Vector* v_rot, HVX_Vector vBias,
                                             int hasBias, int relu, int relu6,
                                             HVX_Vector vZero, HVX_Vector vRelu6) {
    if (hasBias) {
        *v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*v, vBias));
        *v_rot = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*v_rot, vBias));
    }
    if (relu || relu6) {
        *v = Q6_Vhf_vmax_VhfVhf(*v, vZero);
        *v_rot = Q6_Vhf_vmax_VhfVhf(*v_rot, vZero);
        if (relu6) {
            HVX_VectorPred qGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(*v, vRelu6);
            HVX_VectorPred qRotGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(*v_rot, vRelu6);
            *v = Q6_V_vmux_QVV(qGtRelu6, vRelu6, *v);
            *v_rot = Q6_V_vmux_QVV(qRotGtRelu6, vRelu6, *v_rot);
        }
    }
}

static inline HVX_Vector load_w8_output_scale(__fp16* scale_work, const __fp16* scale) {
    memcpy(scale_work, scale, 32 * sizeof(__fp16));
    memset(scale_work + 32, 0, 32 * sizeof(__fp16));
    HVX_Vector vScaleRaw = vmemu(scale_work);
    HVX_Vector vScaleRot = Q6_V_vror_VR(vScaleRaw, 64);
    return Q6_V_valign_VVR(vScaleRaw, vScaleRot, 64);
}

static inline void init_w8_hmx_scale_bias(void* out_scales, const __fp16* column_scales,
                                          const __fp16* column_biases) {
    uint32_t* words = (uint32_t*)out_scales;
    const uint16_t* scaleBits = (const uint16_t*)column_scales;
    if (column_biases != nullptr) {
        const uint16_t* biasBits = (const uint16_t*)column_biases;
        for (int col = 0; col < 32; ++col) {
            words[col] = (uint32_t)scaleBits[col] | ((uint32_t)biasBits[col] << 16);
        }
    } else {
        for (int col = 0; col < 32; ++col) {
            words[col] = (uint32_t)scaleBits[col];
        }
    }
    HVX_Vector* pv = (HVX_Vector*)(words + 32);
    *pv = Q6_V_vzero();
}

static inline void init_w8_hmx_identity_bias(void *out_scales, const __fp16 *column_biases) {
  uint32_t       *words    = (uint32_t *) out_scales;
  const uint16_t *biasBits = (const uint16_t *) column_biases;
  for (int col = 0; col < 32; ++col) {
    words[col] = 0x3c00U | (column_biases ? ((uint32_t) biasBits[col] << 16) : 0U);
  }
  HVX_Vector *pv = (HVX_Vector *) (words + 32);
  *pv            = Q6_V_vzero();
}

static inline void hmx_compute_tile_w8a16_fp16(const __fp16* activation, const __fp16* weight,
                                               int kp, __fp16* vtcm_output) {
    if (kp == 32) {
        hmx_load_tiles_fp16(activation, weight, 32);
    } else if (kp == 64) {
        hmx_load_tiles_fp16(activation, weight, 32);
        hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
    } else if (kp == 88) {
        hmx_load_tiles_fp16(activation, weight, 32);
        hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
        hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 24);
    } else if (kp == 96) {
        hmx_load_tiles_fp16(activation, weight, 32);
        hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
        hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 32);
    } else if (kp == 128) {
        hmx_load_tiles_fp16(activation, weight, 32);
        hmx_load_tiles_fp16(activation + 32 * 1024, weight + 32 * 1024, 32);
        hmx_load_tiles_fp16(activation + 64 * 1024, weight + 64 * 1024, 32);
        hmx_load_tiles_fp16(activation + 96 * 1024, weight + 96 * 1024, 32);
    } else if ((kp & 31) == 0) {
        for (int k = 0; k < kp; k += 32) {
            hmx_load_tiles_fp16(activation + (size_t)k * 1024, weight + (size_t)k * 1024, 32);
        }
    } else {
        for (int k = 0; k < kp; k += 32) {
            int kend = k + 32;
            if (kend > kp) {
                kend = kp;
            }
            hmx_load_tiles_fp16(activation + (size_t)k * 1024, weight + (size_t)k * 1024, kend - k);
        }
    }
    hmx_consume_accumulator_fp16(vtcm_output);
}

static inline void apply_output_post_fp16_scaled(HVX_Vector* v, HVX_Vector* v_rot, HVX_Vector vScale,
                                                 HVX_Vector vBias, int hasBias, int relu, int relu6,
                                                 HVX_Vector vZero, HVX_Vector vRelu6) {
    *v = Q6_Vhf_vmpy_VhfVhf(*v, vScale);
    *v_rot = Q6_Vhf_vmpy_VhfVhf(*v_rot, vScale);
    apply_output_post_fp16_w8(v, v_rot, vBias, hasBias, relu, relu6, vZero, vRelu6);
}

static inline int store_output_tile_fp16_scaled(uint8_t* dst, const __fp16* vtcm_output, const __fp16* scale,
                                                __fp16* scale_work, const __fp16* bias, int M, int ox, int oy,
                                                int pack, int relu, int relu6, int outputBytes) {
    int pack_idx = (oy * 32) / pack;
    int pack_inner = (oy * 32) % pack;

    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi < 0) valid_xi = 0;
    int xi_limit = valid_xi & ~1;

    HVX_Vector* src_ptr = (HVX_Vector*)vtcm_output;
    size_t c_offset = (size_t)(pack_idx * M + ox * 32) * 128;
    if (outputBytes > 0 && c_offset + (size_t)valid_xi * 128 > (size_t)outputBytes) {
        return AEE_EBADPARM;
    }
    uint8_t* dst_ptr = dst + c_offset;
    HVX_Vector vScale = scale ? load_w8_output_scale(scale_work, scale) : Q6_V_vzero();
    HVX_Vector vBias = bias ? load_w8_output_scale(scale_work, bias) : Q6_V_vzero();
    const int hasBias = bias != nullptr;
    const __fp16 relu6Value = (__fp16)6.0f;
    HVX_Vector vZero = Q6_V_vzero();
    HVX_Vector vRelu6 = Q6_Vh_vsplat_R(*(uint16_t*)&relu6Value);

    HVX_VectorPred q = pack_inner == 0 ? Q6_Q_vsetq_R(64) : Q6_Q_not_Q(Q6_Q_vsetq_R(64));
    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
        HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
        HVX_Vector vLoadRot = Q6_V_valign_VVR(vLoad, vLoad, 64);
        if (scale) {
            apply_output_post_fp16_scaled(&vLoad, &vLoadRot, vScale, vBias, hasBias, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&vLoad, &vLoadRot, vBias, hasBias, relu, relu6, vZero, vRelu6);
        }

        HVX_Vector vFirst = pack_inner == 0 ? vLoad : vLoadRot;
        HVX_Vector vSecond = pack_inner == 0 ? vLoadRot : vLoad;
        HVX_Vector vOld0 = pack_inner == 0 ? vZero : vmem(dst_ptr);
        HVX_Vector vOld1 = pack_inner == 0 ? vZero : vmem(dst_ptr + 128);
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, vFirst, vOld0);
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q, vSecond, vOld1);
        dst_ptr += 256;
    }
    if (xi < valid_xi) {
        HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
        HVX_Vector vLoadRot = Q6_V_valign_VVR(vLoad, vLoad, 64);
        if (scale) {
            apply_output_post_fp16_scaled(&vLoad, &vLoadRot, vScale, vBias, hasBias, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&vLoad, &vLoadRot, vBias, hasBias, relu, relu6, vZero, vRelu6);
        }
        if (pack_inner != 0) {
            vLoad = vLoadRot;
        }
        HVX_Vector vOld = pack_inner == 0 ? vZero : vmem(dst_ptr);
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, vLoad, vOld);
    }
    return AEE_SUCCESS;
}

static inline int store_output_tile_pair_fp16_scaled(uint8_t* dst, const __fp16* vtcm_output0,
                                                     const __fp16* vtcm_output1, const __fp16* scale0,
                                                     const __fp16* scale1, __fp16* scale_work,
                                                     const __fp16* bias0, const __fp16* bias1,
                                                     int M, int ox, int oy, int relu, int relu6,
                                                     int outputBytes) {
    int valid_xi = M - ox * 32;
    if (valid_xi > 32) valid_xi = 32;
    if (valid_xi < 0) valid_xi = 0;
    int xi_limit = valid_xi & ~1;

    HVX_Vector* src0_ptr = (HVX_Vector*)vtcm_output0;
    HVX_Vector* src1_ptr = (HVX_Vector*)vtcm_output1;
    size_t c_offset = (size_t)(((oy * 32) / 64) * M + ox * 32) * 128;
    if (outputBytes > 0 && c_offset + (size_t)valid_xi * 128 > (size_t)outputBytes) {
        return AEE_EBADPARM;
    }
    uint8_t* dst_ptr = dst + c_offset;
    HVX_Vector vScale0 = scale0 ? load_w8_output_scale(scale_work, scale0) : Q6_V_vzero();
    HVX_Vector vBias0 = bias0 ? load_w8_output_scale(scale_work, bias0) : Q6_V_vzero();
    HVX_Vector vScale1 = scale1 ? load_w8_output_scale(scale_work, scale1) : Q6_V_vzero();
    HVX_Vector vBias1 = bias1 ? load_w8_output_scale(scale_work, bias1) : Q6_V_vzero();
    const int hasBias0 = bias0 != nullptr;
    const int hasBias1 = bias1 != nullptr;
    const __fp16 relu6Value = (__fp16)6.0f;
    HVX_Vector vZero = Q6_V_vzero();
    HVX_Vector vRelu6 = Q6_Vh_vsplat_R(*(uint16_t*)&relu6Value);
    HVX_VectorPred q_low = Q6_Q_vsetq_R(64);

    int xi = 0;
    for (; xi < xi_limit; xi += 2) {
        HVX_Vector v0 = Q6_Vh_vdeal_Vh(*src0_ptr++);
        HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
        HVX_Vector v1 = Q6_Vh_vdeal_Vh(*src1_ptr++);
        HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
        if (scale0) {
            apply_output_post_fp16_scaled(&v0, &v0_rot, vScale0, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&v0, &v0_rot, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        }
        if (scale1) {
            apply_output_post_fp16_scaled(&v1, &v1_rot, vScale1, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&v1, &v1_rot, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
        dst_ptr += 256;
    }
    if (xi < valid_xi) {
        HVX_Vector v0 = Q6_Vh_vdeal_Vh(*src0_ptr++);
        HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
        HVX_Vector v1 = Q6_Vh_vdeal_Vh(*src1_ptr++);
        HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
        if (scale0) {
            apply_output_post_fp16_scaled(&v0, &v0_rot, vScale0, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&v0, &v0_rot, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        }
        if (scale1) {
            apply_output_post_fp16_scaled(&v1, &v1_rot, vScale1, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        } else {
            apply_output_post_fp16_w8(&v1, &v1_rot, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        }
        vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    }
    return AEE_SUCCESS;
}

static inline void convert_w8_tile_to_fp16_unscaled(const int8_t* src, __fp16* dst) {
    const HVX_Vector* srcVec = (const HVX_Vector*)src;
    HVX_Vector* dstVec = (HVX_Vector*)dst;
    for (int i = 0; i < 8; ++i) {
        HVX_Vector vWeight = vmem(srcVec + i);
        HVX_VectorPair extended = Q6_Wh_vsxt_Vb(vWeight);
        HVX_VectorPair converted = Q6_W_vcombine_VV(Q6_Vhf_vcvt_Vh(Q6_V_hi_W(extended)),
                                                     Q6_Vhf_vcvt_Vh(Q6_V_lo_W(extended)));
        dstVec[2 * i] = Q6_V_lo_W(converted);
        dstVec[2 * i + 1] = Q6_V_hi_W(converted);
    }
}

static inline void convert_w8_tile_to_fp16_scaled(const int8_t *src, __fp16 *dst, HVX_Vector vScale, HVX_Vector vOffset,
                                                  int scale_asymmetric) {
  convert_w8_tile_to_fp16_unscaled(src, dst);
  HVX_Vector *dstVec = (HVX_Vector *) dst;
  for (int i = 0; i < 16; ++i) {
    dstVec[i] = Q6_Vhf_vmpy_VhfVhf(dstVec[i], vScale);
    if (scale_asymmetric) {
      dstVec[i] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(dstVec[i], vOffset));
    }
  }
}

typedef struct {
    const int8_t* vtcm_weight_int8;
    __fp16* vtcm_weight;
    int start_idx;
    int count;
    int kp;
    int                  oy_start;
    int                  scale_block_num;
    int                  scale_asymmetric;
    const __fp16        *scales;
    const dma_desc_1d_t* depend;
    worker_synctoken_t* shared_sync;
} HmxW8WeightDmaConvertTask;

typedef struct {
  // Scale-region descriptor, chained ahead of weight_desc (ordered): when a worker
  // sees its weight descriptor's dstate, the scale DMA for the whole oy-chunk is
  // already complete, so converts read scales from the VTCM staging buffer.
  alignas(64) dma_desc_1d_t scale_desc;
  alignas(64) dma_desc_1d_t weight_desc[32];
  worker_synctoken_t        sync_token[32];
  HmxW8WeightDmaConvertTask tasks[32];
  int                       chunk_starts[32];
  int                       chunk_counts[32];
  int                       chunk_ready[32];
  int                       valid_chunks;
} HmxW8WeightDmaConvertAsync;

static inline void wait_w8_weight_dma_done(const dma_desc_1d_t* depend) {
    volatile uint32_t *ctrl_word = (volatile uint32_t*)&depend->dstate_order_bypass_type_length;
    while (((*ctrl_word >> 31) & 0x1) == 0) {
        asm volatile("nop");
    }
}

static void convert_w8_weight_dma_chunk_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxW8WeightDmaConvertTask* task = (HmxW8WeightDmaConvertTask*)data;
    wait_w8_weight_dma_done(task->depend);
    const int kernel_block_size = 32 * 32;
    const size_t tileSize = (size_t)task->kp * kernel_block_size;
    const int8_t* src_oy = task->vtcm_weight_int8 + (size_t)task->start_idx * tileSize;
    __fp16* dst_oy = task->vtcm_weight + (size_t)task->start_idx * tileSize;
    for (int local_oy = 0; local_oy < task->count; ++local_oy) {
      const int     local_oy_idx     = task->start_idx + local_oy;
      const int8_t *src              = src_oy;
      __fp16       *dst              = dst_oy;
      int           cached_scale_idx = -1;
      HVX_Vector    vScale           = Q6_V_vzero();
      HVX_Vector    vOffset          = Q6_V_vzero();
      for (int k = 0; k < task->kp; ++k) {
        if (task->scale_block_num > 1 || task->scale_asymmetric) {
          const int scale_idx = (k * task->scale_block_num) / task->kp;
          if (scale_idx != cached_scale_idx) {
            const int     scale_unit = task->scale_asymmetric ? 128 : 64;
            const __fp16 *scale =
              task->scales + ((size_t) local_oy_idx * task->scale_block_num + scale_idx) * scale_unit;
            vScale           = vmemu(scale);
            vOffset          = task->scale_asymmetric ? vmemu(scale + 64) : Q6_V_vzero();
            cached_scale_idx = scale_idx;
          }
          convert_w8_tile_to_fp16_scaled(src, dst, vScale, vOffset, task->scale_asymmetric);
        } else {
          convert_w8_tile_to_fp16_unscaled(src, dst);
        }
        src += kernel_block_size;
        dst += kernel_block_size;
      }
        src_oy += tileSize;
        dst_oy += tileSize;
    }
    worker_pool_synctoken_jobdone(task->shared_sync);
}

static void start_weight_tiles_w8a16_sym_per_channel_dma(HmxW8WeightDmaConvertAsync *async, __fp16 *vtcm_weight,
                                                         int8_t *vtcm_weight_int8, const int8_t *src_weight,
                                                         int oy_start, int oy_end, int kp, const __fp16 *scales,
                                                         int scale_block_num, int scale_asymmetric,
                                                         __fp16 *vtcm_scale_staging) {
  const int    kernel_block_size = 32 * 32;
  const size_t tileSize          = (size_t) kp * kernel_block_size;
  const int    tileCount         = oy_end - oy_start;
  async->valid_chunks            = 0;
  if (tileCount <= 0) {
    return;
  }

  int task_count = (int) g_max_num_workers;
  if (task_count > 32) {
    task_count = 32;
  }
  if (task_count > tileCount) {
    task_count = tileCount;
  }
  if (task_count < 1) {
    task_count = 1;
  }
  int chunk_size = (tileCount + task_count - 1) / task_count;
  if (chunk_size > 1) {
    chunk_size = (chunk_size + 1) & ~1;
  }

  int current_start = 0;
  for (int i = 0; i < task_count && current_start < tileCount; ++i) {
    int end = current_start + chunk_size;
    if (end > tileCount) {
      end = tileCount;
    }
    async->chunk_starts[async->valid_chunks] = current_start;
    async->chunk_counts[async->valid_chunks] = end - current_start;
    async->chunk_ready[async->valid_chunks]  = 0;
    current_start                            = end;
    ++async->valid_chunks;
  }

  for (int i = 0; i < async->valid_chunks; ++i) {
    memset(&async->weight_desc[i], 0, sizeof(dma_desc_1d_t));
    async->weight_desc[i].next       = (i + 1 < async->valid_chunks) ? (uint32_t) &async->weight_desc[i + 1] : 0;
    async->weight_desc[i].length     = (uint32_t) ((size_t) async->chunk_counts[i] * tileSize);
    async->weight_desc[i].type       = DMA_DESC_TYPE_1D;
    async->weight_desc[i].src_bypass = 1;
    async->weight_desc[i].dst_bypass = 1;
    async->weight_desc[i].ordered    = 1;
    async->weight_desc[i].dstate     = DMA_DESC_DSTATE_PENDING;
    async->weight_desc[i].src = (uint32_t) (src_weight + (size_t) (oy_start + async->chunk_starts[i]) * tileSize);
    async->weight_desc[i].dst = (uint32_t) (vtcm_weight_int8 + (size_t) async->chunk_starts[i] * tileSize);
  }

  // Stage the contiguous scale region [oy_start, oy_end) into VTCM as the first
  // descriptor of the ordered chain: the weight descriptors' dstate then implies
  // the scale DMA is complete, and converts read scales from VTCM instead of
  // scattered uncached vmemu DDR reads (per-transfer latency dominates wconv).
  const int  scale_unit   = scale_asymmetric ? 128 : 64;
  const bool stage_scales = vtcm_scale_staging != nullptr;
  if (stage_scales) {
    memset(&async->scale_desc, 0, sizeof(dma_desc_1d_t));
    async->scale_desc.next       = (uint32_t) &async->weight_desc[0];
    async->scale_desc.length     = (uint32_t) ((size_t) tileCount * scale_block_num * scale_unit * sizeof(int16_t));
    async->scale_desc.type       = DMA_DESC_TYPE_1D;
    async->scale_desc.src_bypass = 1;
    async->scale_desc.dst_bypass = 1;
    async->scale_desc.ordered    = 1;
    async->scale_desc.dstate     = DMA_DESC_DSTATE_PENDING;
    async->scale_desc.src        = (uint32_t) (scales + (size_t) oy_start * scale_block_num * scale_unit);
    async->scale_desc.dst        = (uint32_t) vtcm_scale_staging;
  }
  dmstart(stage_scales ? &async->scale_desc : &async->weight_desc[0]);

  for (int i = 0; i < async->valid_chunks; ++i) {
    worker_pool_synctoken_init(&async->sync_token[i], 1);
    async->tasks[i].vtcm_weight_int8 = vtcm_weight_int8;
    async->tasks[i].vtcm_weight      = vtcm_weight;
    async->tasks[i].start_idx        = async->chunk_starts[i];
    async->tasks[i].count            = async->chunk_counts[i];
    async->tasks[i].kp               = kp;
    async->tasks[i].oy_start         = oy_start;
    async->tasks[i].scale_block_num  = scale_block_num;
    async->tasks[i].scale_asymmetric = scale_asymmetric;
    async->tasks[i].scales           = (vtcm_scale_staging != nullptr) ? vtcm_scale_staging : scales;
    async->tasks[i].depend           = &async->weight_desc[i];
    async->tasks[i].shared_sync      = &async->sync_token[i];
    worker_pool_job_t job;
    job.fptr = convert_w8_weight_dma_chunk_worker;
    job.dptr = &async->tasks[i];
    if (worker_pool_submit(NULL, job) != 0) {
      convert_w8_weight_dma_chunk_worker(&async->tasks[i], 0);
    }
  }
  dma_wait_for_idle();
}

static inline void wait_weight_chunk_w8a16_sym_per_channel(HmxW8WeightDmaConvertAsync* async, int chunk_idx) {
    if (!async->chunk_ready[chunk_idx]) {
        worker_pool_synctoken_wait(&async->sync_token[chunk_idx]);
        async->chunk_ready[chunk_idx] = 1;
    }
}

static inline void wait_all_weight_chunks_w8a16_sym_per_channel(HmxW8WeightDmaConvertAsync* async) {
    for (int i = 0; i < async->valid_chunks; ++i) {
        wait_weight_chunk_w8a16_sym_per_channel(async, i);
    }
}

static inline int compute_store_tiles_w8a16_sym_per_channel(
  uint8_t *dst, const uint8_t *bias, const HmxIm2ColConvParam *params, int M, int pack, int kp, int ox_start,
  int ox_end, int oy_start, int oy_end, __fp16 *vtcm_activation, __fp16 *vtcm_weight, __fp16 *vtcm_output,
  __fp16 *vtcm_scales, const __fp16 *scales, int scale_block_num, int scale_asymmetric) {
  for (int ox = ox_start; ox < ox_end; ++ox) {
    const int local_ox = ox - ox_start;
    __fp16   *act_tile = vtcm_activation + (size_t) local_ox * kp * 1024;
    for (int oy = oy_start; oy < oy_end; ++oy) {
      const int local_oy    = oy - oy_start;
      __fp16   *weight_tile = vtcm_weight + (size_t) local_oy * kp * 1024;
      if (pack == 64 && oy + 1 < oy_end && ((oy * 32) & 63) == 0) {
        __fp16       *weight_tile_next = vtcm_weight + (size_t) (local_oy + 1) * kp * 1024;
        __fp16       *vtcm_output_next = vtcm_output + 1024;
        const __fp16 *biasPtr0         = bias ? ((const __fp16 *) bias) + oy * 32 : nullptr;
        const __fp16 *biasPtr1         = bias ? ((const __fp16 *) bias) + (oy + 1) * 32 : nullptr;
        if (scale_block_num > 1 || scale_asymmetric) {
          init_w8_hmx_identity_bias(vtcm_scales, biasPtr0);
        } else {
          init_w8_hmx_scale_bias(vtcm_scales, scales + (size_t) oy * 32, biasPtr0);
        }
        hmx_set_output_scales(vtcm_scales);
        hmx_compute_tile_w8a16_fp16(act_tile, weight_tile, kp, vtcm_output);
        if (scale_block_num > 1 || scale_asymmetric) {
          init_w8_hmx_identity_bias(vtcm_scales, biasPtr1);
        } else {
          init_w8_hmx_scale_bias(vtcm_scales, scales + (size_t) (oy + 1) * 32, biasPtr1);
        }
        hmx_set_output_scales(vtcm_scales);
        hmx_compute_tile_w8a16_fp16(act_tile, weight_tile_next, kp, vtcm_output_next);
        int storeRet =
          store_output_tile_pair_fp16_scaled(dst, vtcm_output, vtcm_output_next, nullptr, nullptr, vtcm_scales, nullptr,
                                             nullptr, M, ox, oy, params->relu, params->relu6, params->outputBytes);
        if (storeRet != AEE_SUCCESS) {
          return storeRet;
        }
        ++oy;
        continue;
      }
      const __fp16 *biasPtr = bias ? ((const __fp16 *) bias) + oy * 32 : nullptr;
      if (scale_block_num > 1 || scale_asymmetric) {
        init_w8_hmx_identity_bias(vtcm_scales, biasPtr);
      } else {
        init_w8_hmx_scale_bias(vtcm_scales, scales + (size_t) oy * 32, biasPtr);
      }
      hmx_set_output_scales(vtcm_scales);
      hmx_compute_tile_w8a16_fp16(act_tile, weight_tile, kp, vtcm_output);
      int storeRet = store_output_tile_fp16_scaled(dst, vtcm_output, nullptr, vtcm_scales, nullptr, M, ox, oy, pack,
                                                   params->relu, params->relu6, params->outputBytes);
      if (storeRet != AEE_SUCCESS) {
        return storeRet;
      }
    }
  }
  return AEE_SUCCESS;
}

int hmx_matmul_w8a16_block_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const uint8_t *bias,
                                const HmxIm2ColConvParam *params) {
  const Im2ColParameter *p     = &params->im2col;
  const int              batch = params->batch > 0 ? params->batch : 1;
  const int              pack  = p->packCUnit;
  const int              M     = batch * p->oh * p->ow;
  const int              N     = params->oc;
  const int     kp = p->kernelCountUnit > 0 ? p->kernelCountUnit : (p->kernelX * p->kernelY * ((p->ic + 31) / 32));
  const int     np = (N + 31) / 32;
  const int     mp = (M + 31) / 32;
  const int     scale_block_num  = params->scaleBlockNum > 0 ? params->scaleBlockNum : 1;
  const int     scale_asymmetric = params->scaleAsymmetric != 0;
  const int8_t *src_weight       = (const int8_t *) weight;
  const __fp16 *scales           = (const __fp16 *) (weight + (size_t) np * kp * 1024);

  int np_chunk = params->np > 0 ? params->np : 1;
  int mp_chunk = params->mp > 0 ? params->mp : 1;
  if (np_chunk > np) {
    np_chunk = np;
  }
  if (mp_chunk > mp) {
    mp_chunk = mp;
  }
  const int     ox_chunk_count        = (mp + mp_chunk - 1) / mp_chunk;
  const int     oy_chunk_count        = (np + np_chunk - 1) / np_chunk;
  const int64_t activation_outer_cost = (int64_t) mp + (int64_t) ox_chunk_count * np;
  const int64_t weight_outer_cost     = (int64_t) oy_chunk_count * mp + (int64_t) np;
  const bool    reuse_activation      = activation_outer_cost <= weight_outer_cost;

  uint8_t  *vtcm_ptr           = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16   *vtcm_weight        = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) np_chunk * kp * 1024 * sizeof(int16_t));
  int8_t   *vtcm_weight_int8   = (int8_t *) vtcm_seq_alloc(&vtcm_ptr, (size_t) np_chunk * kp * 1024 * sizeof(int8_t));
  __fp16   *vtcm_activation    = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) mp_chunk * kp * 1024 * sizeof(int16_t));
  __fp16   *vtcm_output        = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 4096);
  __fp16   *vtcm_scales        = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  // Stage the contiguous per-oy scale region for each oy-chunk into VTCM so the
  // convert workers read scales from VTCM instead of scattered uncached DDR
  // vmemu reads. Host-side sizing (chooseIm2ColTileShape) accounts for this.
  const int scale_unit         = scale_asymmetric ? 128 : 64;
  __fp16   *vtcm_scale_staging = nullptr;
  if (scale_block_num > 1 || scale_asymmetric) {
    vtcm_scale_staging =
      (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, (size_t) np_chunk * scale_block_num * scale_unit * sizeof(int16_t));
  }

  // Authoritative VTCM bounds check: vtcm_seq_alloc is an unchecked bump
  // allocator, so a host/DSP sizing drift would let the DMA below write past
  // the end of VTCM (cDSP hang / device reboot). Guard before any DMA.
  const uintptr_t vtcm_begin = (uintptr_t) vtcm_manager_get_vtcm_base();
  const size_t    vtcm_size  = (size_t) vtcm_manager_get_vtcm_size();
  const uintptr_t vtcm_end   = vtcm_begin + vtcm_size;
  if (vtcm_begin == 0 || vtcm_end < vtcm_begin || (uintptr_t) vtcm_ptr > vtcm_end) {
    return AEE_ENOMEMORY;
  }

  hmx_manager_enable_execution();
  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
  hmx_unit_acquire();
  hmx_set_output_scales(vtcm_scales);

  if (reuse_activation) {
    for (int ox_start = 0; ox_start < mp; ox_start += mp_chunk) {
      const int ox_end = (ox_start + mp_chunk > mp) ? mp : (ox_start + mp_chunk);
      hmx_im2col_fill_activation_tiles(vtcm_activation, src, p, ox_start, ox_end - ox_start, kp, batch);
      for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
        const int                  oy_end = (oy_start + np_chunk > np) ? np : (oy_start + np_chunk);
        HmxW8WeightDmaConvertAsync async  = {};
        start_weight_tiles_w8a16_sym_per_channel_dma(&async, vtcm_weight, vtcm_weight_int8, src_weight, oy_start,
                                                     oy_end, kp, scales, scale_block_num, scale_asymmetric,
                                                     vtcm_scale_staging);
        for (int chunk_idx = 0; chunk_idx < async.valid_chunks; ++chunk_idx) {
          wait_weight_chunk_w8a16_sym_per_channel(&async, chunk_idx);
          const int chunk_oy_start = oy_start + async.chunk_starts[chunk_idx];
          const int chunk_oy_end   = chunk_oy_start + async.chunk_counts[chunk_idx];
          __fp16   *chunk_weight   = vtcm_weight + (size_t) async.chunk_starts[chunk_idx] * kp * 1024;
          int       ret            = compute_store_tiles_w8a16_sym_per_channel(
            dst, bias, params, M, pack, kp, ox_start, ox_end, chunk_oy_start, chunk_oy_end, vtcm_activation,
            chunk_weight, vtcm_output, vtcm_scales, scales, scale_block_num, scale_asymmetric);
          if (ret != AEE_SUCCESS) {
            wait_all_weight_chunks_w8a16_sym_per_channel(&async);
            hmx_unit_release();
            hmx_manager_disable_execution();
            return ret;
          }
        }
      }
    }
  } else {
    for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
      const int                  oy_end = (oy_start + np_chunk > np) ? np : (oy_start + np_chunk);
      HmxW8WeightDmaConvertAsync async  = {};
      start_weight_tiles_w8a16_sym_per_channel_dma(&async, vtcm_weight, vtcm_weight_int8, src_weight, oy_start, oy_end,
                                                   kp, scales, scale_block_num, scale_asymmetric, vtcm_scale_staging);

      for (int ox_start = 0; ox_start < mp; ox_start += mp_chunk) {
        const int ox_end = (ox_start + mp_chunk > mp) ? mp : (ox_start + mp_chunk);
        hmx_im2col_fill_activation_tiles(vtcm_activation, src, p, ox_start, ox_end - ox_start, kp, batch);
        for (int chunk_idx = 0; chunk_idx < async.valid_chunks; ++chunk_idx) {
          wait_weight_chunk_w8a16_sym_per_channel(&async, chunk_idx);
          const int chunk_oy_start = oy_start + async.chunk_starts[chunk_idx];
          const int chunk_oy_end   = chunk_oy_start + async.chunk_counts[chunk_idx];
          __fp16   *chunk_weight   = vtcm_weight + (size_t) async.chunk_starts[chunk_idx] * kp * 1024;
          int       ret            = compute_store_tiles_w8a16_sym_per_channel(
            dst, bias, params, M, pack, kp, ox_start, ox_end, chunk_oy_start, chunk_oy_end, vtcm_activation,
            chunk_weight, vtcm_output, vtcm_scales, scales, scale_block_num, scale_asymmetric);
          if (ret != AEE_SUCCESS) {
            wait_all_weight_chunks_w8a16_sym_per_channel(&async);
            hmx_unit_release();
            hmx_manager_disable_execution();
            return ret;
          }
        }
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

int hmx_conv1x1_direct_w8a16_sym_per_channel(uint8_t *dst, const uint8_t *src, const uint8_t *weight,
                                             const uint8_t *bias, const HmxIm2ColConvParam *params) {
  return hmx_matmul_w8a16_block_fp16(dst, src, weight, bias, params);
}
