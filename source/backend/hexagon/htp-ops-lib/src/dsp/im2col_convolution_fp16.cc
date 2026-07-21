#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_utils.h"
#include "dsp/dma_utils.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

static inline int store_output_tile_fp16(uint8_t* dst, const __fp16* vtcm_output, const __fp16* bias,
                                         int M, int ox, int oy, int pack, int relu, int relu6,
                                         int outputBytes) {
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
    HVX_Vector vBias = Q6_V_vzero();
    if (bias) {
        vBias = *((HVX_Vector*)bias);
    }
    const __fp16 relu6Value = (__fp16)6.0f;
    HVX_Vector vZero = Q6_V_vzero();
    HVX_Vector vRelu6 = Q6_Vh_vsplat_R(*(uint16_t*)&relu6Value);

    HVX_VectorPred q = pack_inner == 0 ? Q6_Q_vsetq_R(64) : Q6_Q_not_Q(Q6_Q_vsetq_R(64));
    int xi = 0;
    if (pack_inner == 0) {
        for (; xi < xi_limit; xi += 2) {
            HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
            HVX_Vector vLoadRot = Q6_V_valign_VVR(vLoad, vLoad, 64);
            if (bias) {
                vLoad = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoad, vBias));
                vLoadRot = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoadRot, vBias));
            }
            if (relu || relu6) {
                vLoad = Q6_Vhf_vmax_VhfVhf(vLoad, vZero);
                vLoadRot = Q6_Vhf_vmax_VhfVhf(vLoadRot, vZero);
                if (relu6) {
                    HVX_VectorPred qLoadGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoad, vRelu6);
                    HVX_VectorPred qLoadRotGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoadRot, vRelu6);
                    vLoad = Q6_V_vmux_QVV(qLoadGtRelu6, vRelu6, vLoad);
                    vLoadRot = Q6_V_vmux_QVV(qLoadRotGtRelu6, vRelu6, vLoadRot);
                }
            }
            Q6_vmem_QRIV(q, (HVX_Vector*)dst_ptr, vLoad);
            Q6_vmem_QRIV(q, (HVX_Vector*)(dst_ptr + 128), vLoadRot);
            dst_ptr += 256;
        }
        if (xi < valid_xi) {
            HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
            if (bias) {
                vLoad = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoad, vBias));
            }
            if (relu || relu6) {
                vLoad = Q6_Vhf_vmax_VhfVhf(vLoad, vZero);
                if (relu6) {
                    HVX_VectorPred qLoadGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoad, vRelu6);
                    vLoad = Q6_V_vmux_QVV(qLoadGtRelu6, vRelu6, vLoad);
                }
            }
            Q6_vmem_QRIV(q, (HVX_Vector*)dst_ptr, vLoad);
        }
        return AEE_SUCCESS;
    }
    for (; xi < xi_limit; xi += 2) {
        HVX_Vector vLoad = Q6_Vh_vdeal_Vh(*src_ptr++);
        HVX_Vector vLoadRot = Q6_V_valign_VVR(vLoad, vLoad, 64);
        if (bias) {
            vLoad = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoad, vBias));
            vLoadRot = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoadRot, vBias));
        }
        if (relu || relu6) {
            vLoad = Q6_Vhf_vmax_VhfVhf(vLoad, vZero);
            vLoadRot = Q6_Vhf_vmax_VhfVhf(vLoadRot, vZero);
            if (relu6) {
                HVX_VectorPred qLoadGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoad, vRelu6);
                HVX_VectorPred qLoadRotGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoadRot, vRelu6);
                vLoad = Q6_V_vmux_QVV(qLoadGtRelu6, vRelu6, vLoad);
                vLoadRot = Q6_V_vmux_QVV(qLoadRotGtRelu6, vRelu6, vLoadRot);
            }
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
        if (bias) {
            vLoad = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(vLoad, vBias));
        }
        if (relu || relu6) {
            vLoad = Q6_Vhf_vmax_VhfVhf(vLoad, vZero);
            if (relu6) {
                HVX_VectorPred qLoadGtRelu6 = Q6_Q_vcmp_gt_VhfVhf(vLoad, vRelu6);
                vLoad = Q6_V_vmux_QVV(qLoadGtRelu6, vRelu6, vLoad);
            }
        }
        if (pack_inner != 0) {
            HVX_Vector vLoadRot = Q6_V_valign_VVR(vLoad, vLoad, 64);
            vLoad = vLoadRot;
        }
        HVX_Vector vOld = pack_inner == 0 ? vZero : vmem(dst_ptr);
        vmem(dst_ptr) = Q6_V_vmux_QVV(q, vLoad, vOld);
    }
    return AEE_SUCCESS;
}

static inline void apply_output_post_fp16(HVX_Vector* v, HVX_Vector* v_rot, HVX_Vector vBias,
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

static inline int store_output_tile_pair_fp16(uint8_t* dst, const __fp16* vtcm_output0,
                                              const __fp16* vtcm_output1, const __fp16* bias0,
                                              const __fp16* bias1, int M, int ox, int oy,
                                              int relu, int relu6, int outputBytes) {
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
    HVX_Vector vBias0 = bias0 ? *((HVX_Vector*)bias0) : Q6_V_vzero();
    HVX_Vector vBias1 = bias1 ? *((HVX_Vector*)bias1) : Q6_V_vzero();
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
        apply_output_post_fp16(&v0, &v0_rot, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        apply_output_post_fp16(&v1, &v1_rot, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
        vmem(dst_ptr + 128) = Q6_V_vmux_QVV(q_low, v0_rot, v1);
        dst_ptr += 256;
    }
    if (xi < valid_xi) {
        HVX_Vector v0 = Q6_Vh_vdeal_Vh(*src0_ptr++);
        HVX_Vector v0_rot = Q6_V_valign_VVR(v0, v0, 64);
        HVX_Vector v1 = Q6_Vh_vdeal_Vh(*src1_ptr++);
        HVX_Vector v1_rot = Q6_V_valign_VVR(v1, v1, 64);
        apply_output_post_fp16(&v0, &v0_rot, vBias0, hasBias0, relu, relu6, vZero, vRelu6);
        apply_output_post_fp16(&v1, &v1_rot, vBias1, hasBias1, relu, relu6, vZero, vRelu6);
        vmem(dst_ptr) = Q6_V_vmux_QVV(q_low, v0, v1_rot);
    }
    return AEE_SUCCESS;
}

static inline void compute_hmx_tile_fp16(const __fp16* act_tile, const __fp16* weight_tile,
                                         int kp, __fp16* vtcm_output) {
    for (int kk = 0; kk < kp; kk += 32) {
        int kend = kk + 32;
        if (kend > kp) {
            kend = kp;
        }
        hmx_load_tiles_fp16(act_tile + (size_t)kk * 1024, weight_tile + (size_t)kk * 1024, kend - kk);
    }
    hmx_consume_accumulator_fp16(vtcm_output);
}

static inline size_t input_block_offset_fp16(const Im2ColParameter* p, int ob, int sy, int sx, int icBlock) {
    const int block_channel = icBlock * 32;
    const int pack_idx = block_channel / p->packCUnit;
    const int pack_inner = block_channel % p->packCUnit;
    const size_t elem_offset = (size_t)pack_idx * p->srcZStep +
                               (size_t)(ob * p->ih + sy) * p->srcYStep +
                               (size_t)sx * p->packCUnit +
                               (size_t)pack_inner;
    return elem_offset * sizeof(__fp16);
}

static inline void fill_im2col_activation_kk_range(__fp16* tile_base, const uint8_t* src,
                                                   const Im2ColParameter* p, int tileStart, int validCount,
                                                   int kp, int kk_start, int kk_end,
                                                   int ic_blocks, int plane) {
    (void)kp;
    int ob_list[32];
    int base_y_list[32];
    int base_x_list[32];
    for (int m = 0; m < validCount; ++m) {
        const int global_m = tileStart + m;
        const int ob = global_m / plane;
        const int plane_offset = global_m - ob * plane;
        const int oy = plane_offset / p->ow;
        const int ox = plane_offset - oy * p->ow;
        ob_list[m] = ob;
        base_y_list[m] = oy * p->strideY - p->padY;
        base_x_list[m] = ox * p->strideX - p->padX;
    }
    for (int kk = kk_start; kk < kk_end; ++kk) {
        const int kernel_index = kk / ic_blocks;
        const int ic_block = kk % ic_blocks;
        const int ky = kernel_index / p->kernelX;
        const int kx = kernel_index % p->kernelX;
        const int ky_offset = ky * p->dilateY;
        const int kx_offset = kx * p->dilateX;
        const int src_stride_bytes = p->strideX * p->packCUnit * (int)sizeof(__fp16);
        int m = 0;
        while (m < validCount) {
            const int ob = ob_list[m];
            const int sy = base_y_list[m] + ky_offset;
            const int sx = base_x_list[m] + kx_offset;
            if (sy < 0 || sy >= p->ih || sx < 0 || sx >= p->iw) {
                memset(tile_base + (size_t)kk * 1024 + (size_t)m * 32, 0, 32 * sizeof(__fp16));
                ++m;
                continue;
            }

            int run = 1;
            while (m + run < validCount) {
                const int next_idx = m + run;
                const int next_sy = base_y_list[next_idx] + ky_offset;
                const int next_sx = base_x_list[next_idx] + kx_offset;
                if (ob_list[next_idx] != ob || next_sy != sy || next_sx != sx + run * p->strideX ||
                    next_sx < 0 || next_sx >= p->iw) {
                    break;
                }
                ++run;
            }

            const uint8_t* src_ptr = src + input_block_offset_fp16(p, ob, sy, sx, ic_block);
            __fp16* dst_ptr = tile_base + (size_t)kk * 1024 + (size_t)m * 32;
            if (run > 1 && src_stride_bytes > 0) {
                l2fetch(src_ptr, src_stride_bytes, 32 * sizeof(__fp16), run, 0);
            }
            for (int r = 0; r < run; ++r) {
                memcpy(dst_ptr + (size_t)r * 32, src_ptr + (size_t)r * src_stride_bytes, 32 * sizeof(__fp16));
            }
            m += run;
        }
        if (validCount < 32) {
            memset(tile_base + (size_t)kk * 1024 + (size_t)validCount * 32, 0,
                   (size_t)(32 - validCount) * 32 * sizeof(__fp16));
        }
        for (int i = 0; i < 16; ++i) {
            HVX_Vector* va = (HVX_Vector*)(tile_base + (size_t)kk * 1024 + i * 64);
            va[0] = Q6_Vh_vshuff_Vh(va[0]);
        }
    }
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    size_t tile_elems;
    int tile_start;
    int tile_count;
    int totalM;
    int kp;
    int ic_blocks;
    int plane;
    worker_synctoken_t sync_ctx;
} HmxIm2ColFillTaskState;

typedef struct {
    HmxIm2ColFillTaskState* state;
    int kk_start;
    int kk_end;
} HmxIm2ColFillTask;

static void fill_im2col_activation_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2ColFillTask* task = (HmxIm2ColFillTask*)data;
    HmxIm2ColFillTaskState* state = task->state;
    for (int local_tile = 0; local_tile < state->tile_count; ++local_tile) {
        __fp16* tile_base = state->vtcm_activation + (size_t)local_tile * state->tile_elems;
        const int tileStart = (state->tile_start + local_tile) * 32;
        int validCount = state->totalM - tileStart;
        if (validCount > 32) {
            validCount = 32;
        }
        if (validCount <= 0) {
            continue;
        }
        fill_im2col_activation_kk_range(tile_base, state->src, state->p, tileStart,
                                        validCount, state->kp, task->kk_start, task->kk_end,
                                        state->ic_blocks, state->plane);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    size_t tile_elems;
    int tile_start;
    int tile_count;
    int kp;
    int batch;
    int totalM;
    int pair_count;
    worker_synctoken_t sync_ctx;
} HmxIm2Col1x1DirectTaskState;

typedef struct {
    HmxIm2Col1x1DirectTaskState* state;
    int unit_start;
    int unit_end;
} HmxIm2Col1x1DirectTask;

static inline void fill_im2col_activation_1x1_pack64_direct_unit(__fp16* vtcm_activation, const uint8_t* src,
                                                                 const Im2ColParameter* p, size_t tile_elems,
                                                                 int tile_start, int totalM,
                                                                 int local_tile, int pair_idx) {
    const int kk = pair_idx * 2;
    __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
    __fp16* tile0 = tile_base + (size_t)kk * 1024;
    __fp16* tile1 = tile0 + 1024;
    const int tileStart = (tile_start + local_tile) * 32;
    int validRows = totalM - tileStart;
    if (validRows > 32) {
        validRows = 32;
    }
    if (validRows <= 0) {
        return;
    }
    if (validRows < 32) {
        memset(tile0, 0, 1024 * sizeof(__fp16));
        memset(tile1, 0, 1024 * sizeof(__fp16));
    }
    const uint8_t* src_base = src + ((size_t)pair_idx * p->srcZStep +
                                     (size_t)tileStart * p->packCUnit) * sizeof(__fp16);
    int r = 0;
    for (; r <= validRows - 2; r += 2) {
        HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
        HVX_Vector v1 = vmem((const HVX_Vector*)(src_base + (size_t)(r + 1) * p->packCUnit * sizeof(__fp16)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
    if (r < validRows) {
        HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
}

static void fill_im2col_activation_1x1_pack64_direct_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2Col1x1DirectTask* task = (HmxIm2Col1x1DirectTask*)data;
    HmxIm2Col1x1DirectTaskState* state = task->state;
    for (int unit = task->unit_start; unit < task->unit_end; ++unit) {
        const int local_tile = unit / state->pair_count;
        const int pair_idx = unit - local_tile * state->pair_count;
        fill_im2col_activation_1x1_pack64_direct_unit(state->vtcm_activation, state->src, state->p,
                                                      state->tile_elems, state->tile_start, state->totalM,
                                                      local_tile, pair_idx);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_1x1_pack64_direct_parallel(__fp16* vtcm_activation, const uint8_t* src,
                                                              const Im2ColParameter* p,
                                                              int tile_start, int tile_count, int kp, int batch) {
    const int pair_count = kp / 2;
    const int unit_count = tile_count * pair_count;
    int task_count = (int)g_max_num_workers;
    if (task_count > unit_count) {
        task_count = unit_count;
    }
    if (task_count <= 1) {
        return;
    }

    HmxIm2Col1x1DirectTaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_elems = (size_t)kp * 1024;
    state.tile_start = tile_start;
    state.tile_count = tile_count;
    state.kp = kp;
    state.batch = batch;
    state.totalM = batch * p->oh * p->ow;
    state.pair_count = pair_count;

    HmxIm2Col1x1DirectTask* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2Col1x1DirectTask, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_1x1_pack64_direct_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int unit_start = (int)((int64_t)i * unit_count / task_count);
        const int unit_end = (int)((int64_t)(i + 1) * unit_count / task_count);
        tasks[i].state = &state;
        tasks[i].unit_start = unit_start;
        tasks[i].unit_end = unit_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    int tile_start;
    int totalM;
    worker_synctoken_t sync_ctx;
} HmxIm2Col1x1Kp1TaskState;

typedef struct {
    HmxIm2Col1x1Kp1TaskState* state;
    int tile_begin;
    int tile_end;
} HmxIm2Col1x1Kp1Task;

static inline void fill_im2col_activation_1x1_pack64_kp1_direct_unit(__fp16* vtcm_activation, const uint8_t* src,
                                                                     const Im2ColParameter* p,
                                                                     int tile_start, int totalM,
                                                                     int local_tile) {
    __fp16* tile0 = vtcm_activation + (size_t)local_tile * 1024;
    const int tileStart = (tile_start + local_tile) * 32;
    int validRows = totalM - tileStart;
    if (validRows > 32) {
        validRows = 32;
    }
    if (validRows <= 0) {
        return;
    }
    if (validRows < 32) {
        memset(tile0, 0, 1024 * sizeof(__fp16));
    }
    const uint8_t* src_base = src + (size_t)tileStart * p->packCUnit * sizeof(__fp16);
    int r = 0;
    for (; r <= validRows - 2; r += 2) {
        HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
        HVX_Vector v1 = vmem((const HVX_Vector*)(src_base + (size_t)(r + 1) * p->packCUnit * sizeof(__fp16)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
    }
    if (r < validRows) {
        HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
        HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
    }
}

static void fill_im2col_activation_1x1_pack64_kp1_direct_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2Col1x1Kp1Task* task = (HmxIm2Col1x1Kp1Task*)data;
    HmxIm2Col1x1Kp1TaskState* state = task->state;
    for (int local_tile = task->tile_begin; local_tile < task->tile_end; ++local_tile) {
        fill_im2col_activation_1x1_pack64_kp1_direct_unit(state->vtcm_activation, state->src, state->p,
                                                          state->tile_start, state->totalM,
                                                          local_tile);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_1x1_pack64_kp1_direct_parallel(__fp16* vtcm_activation, const uint8_t* src,
                                                                  const Im2ColParameter* p,
                                                                  int tile_start, int tile_count, int batch) {
    int task_count = (int)g_max_num_workers;
    if (task_count > tile_count) {
        task_count = tile_count;
    }
    if (task_count <= 1) {
        return;
    }

    HmxIm2Col1x1Kp1TaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_start = tile_start;
    state.totalM = batch * p->oh * p->ow;

    HmxIm2Col1x1Kp1Task* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2Col1x1Kp1Task, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_1x1_pack64_kp1_direct_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int tile_begin = (int)((int64_t)i * tile_count / task_count);
        const int tile_end = (int)((int64_t)(i + 1) * tile_count / task_count);
        tasks[i].state = &state;
        tasks[i].tile_begin = tile_begin;
        tasks[i].tile_end = tile_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

static inline void store_pack64_im2col_row_pair(__fp16* tile0, __fp16* tile1, int r,
                                                const uint8_t* src0, const uint8_t* src1);

static inline bool use_pack64_1x1_strided_fast(const Im2ColParameter* p, int kp, int batch, int ic_blocks) {
    return batch == 1 && p->packCUnit == 64 && p->kernelX == 1 && p->kernelY == 1 &&
           p->dilateX == 1 && p->dilateY == 1 && p->padX == 0 && p->padY == 0 &&
           kp == ic_blocks && (p->strideX != 1 || p->strideY != 1);
}

static inline void fill_im2col_activation_1x1_pack64_strided_one_tile(__fp16* vtcm_activation,
                                                                      const uint8_t* src,
                                                                      const Im2ColParameter* p,
                                                                      int tile_start, int totalM,
                                                                      int local_tile, int kp) {
    const size_t tile_elems = (size_t)kp * 1024;
    __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
    const int tileStart = (tile_start + local_tile) * 32;
    int validRows = totalM - tileStart;
    if (validRows > 32) {
        validRows = 32;
    }
    if (validRows <= 0) {
        return;
    }
    const int oy = tileStart / p->ow;
    const int ox = tileStart - oy * p->ow;
    const bool fullRowTile = ox + validRows - 1 < p->ow;
    const int paired_kp = kp & ~1;
    for (int kk = 0; kk < paired_kp; kk += 2) {
        __fp16* tile0 = tile_base + (size_t)kk * 1024;
        __fp16* tile1 = tile0 + 1024;
        if (validRows < 32) {
            memset(tile0, 0, 1024 * sizeof(__fp16));
            memset(tile1, 0, 1024 * sizeof(__fp16));
        }
        if (fullRowTile) {
            const int sy = oy * p->strideY;
            const int sxBase = ox * p->strideX;
            for (int r = 0; r <= validRows - 2; r += 2) {
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, sy, sxBase + r * p->strideX, kk);
                const uint8_t* src1 = src0 + (size_t)p->strideX * p->packCUnit * sizeof(__fp16);
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
            }
            if (validRows & 1) {
                const int r = validRows - 1;
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, sy, sxBase + r * p->strideX, kk);
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
            }
            continue;
        }
        for (int r = 0; r <= validRows - 2; r += 2) {
            const int out0 = tileStart + r;
            const int out1 = out0 + 1;
            const int oy0 = out0 / p->ow;
            const int ox0 = out0 - oy0 * p->ow;
            const int oy1 = out1 / p->ow;
            const int ox1 = out1 - oy1 * p->ow;
            const uint8_t* src0 = src + input_block_offset_fp16(p, 0, oy0 * p->strideY, ox0 * p->strideX, kk);
            const uint8_t* src1 = src + input_block_offset_fp16(p, 0, oy1 * p->strideY, ox1 * p->strideX, kk);
            store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
        }
        if (validRows & 1) {
            const int r = validRows - 1;
            const int out0 = tileStart + r;
            const int oy0 = out0 / p->ow;
            const int ox0 = out0 - oy0 * p->ow;
            const uint8_t* src0 = src + input_block_offset_fp16(p, 0, oy0 * p->strideY, ox0 * p->strideX, kk);
            store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
        }
    }
    if (paired_kp < kp) {
        __fp16* tile0 = tile_base + (size_t)paired_kp * 1024;
        if (validRows < 32) {
            memset(tile0, 0, 1024 * sizeof(__fp16));
        }
        if (fullRowTile) {
            const int sy = oy * p->strideY;
            const int sxBase = ox * p->strideX;
            for (int r = 0; r <= validRows - 2; r += 2) {
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, sy, sxBase + r * p->strideX, paired_kp);
                const uint8_t* src1 = src0 + (size_t)p->strideX * p->packCUnit * sizeof(__fp16);
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, src1);
            }
            if (validRows & 1) {
                const int r = validRows - 1;
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, sy, sxBase + r * p->strideX, paired_kp);
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, nullptr);
            }
        } else {
            for (int r = 0; r <= validRows - 2; r += 2) {
                const int out0 = tileStart + r;
                const int out1 = out0 + 1;
                const int oy0 = out0 / p->ow;
                const int ox0 = out0 - oy0 * p->ow;
                const int oy1 = out1 / p->ow;
                const int ox1 = out1 - oy1 * p->ow;
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, oy0 * p->strideY, ox0 * p->strideX, paired_kp);
                const uint8_t* src1 = src + input_block_offset_fp16(p, 0, oy1 * p->strideY, ox1 * p->strideX, paired_kp);
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, src1);
            }
            if (validRows & 1) {
                const int r = validRows - 1;
                const int out0 = tileStart + r;
                const int oy0 = out0 / p->ow;
                const int ox0 = out0 - oy0 * p->ow;
                const uint8_t* src0 = src + input_block_offset_fp16(p, 0, oy0 * p->strideY, ox0 * p->strideX, paired_kp);
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, nullptr);
            }
        }
    }
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    int tile_start;
    int totalM;
    int kp;
    worker_synctoken_t sync_ctx;
} HmxIm2Col1x1StridedTaskState;

typedef struct {
    HmxIm2Col1x1StridedTaskState* state;
    int tile_begin;
    int tile_end;
} HmxIm2Col1x1StridedTask;

static void fill_im2col_activation_1x1_pack64_strided_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2Col1x1StridedTask* task = (HmxIm2Col1x1StridedTask*)data;
    HmxIm2Col1x1StridedTaskState* state = task->state;
    for (int local_tile = task->tile_begin; local_tile < task->tile_end; ++local_tile) {
        fill_im2col_activation_1x1_pack64_strided_one_tile(state->vtcm_activation, state->src, state->p,
                                                           state->tile_start, state->totalM,
                                                           local_tile, state->kp);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_1x1_pack64_strided_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                                            const Im2ColParameter* p,
                                                            int tile_start, int tile_count, int kp, int batch) {
    int task_count = (int)g_max_num_workers;
    if (task_count > tile_count) {
        task_count = tile_count;
    }
    if (task_count <= 1) {
        const int totalM = batch * p->oh * p->ow;
        for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
            fill_im2col_activation_1x1_pack64_strided_one_tile(vtcm_activation, src, p, tile_start,
                                                               totalM, local_tile, kp);
        }
        return;
    }
    HmxIm2Col1x1StridedTaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_start = tile_start;
    state.totalM = batch * p->oh * p->ow;
    state.kp = kp;
    HmxIm2Col1x1StridedTask* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2Col1x1StridedTask, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_1x1_pack64_strided_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int tile_begin = (int)((int64_t)i * tile_count / task_count);
        const int tile_end = (int)((int64_t)(i + 1) * tile_count / task_count);
        tasks[i].state = &state;
        tasks[i].tile_begin = tile_begin;
        tasks[i].tile_end = tile_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

static void fill_im2col_activation_1x1_pack64_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                                    const Im2ColParameter* p,
                                                    int tile_start, int tile_count, int kp, int batch) {
    const size_t tile_elems = (size_t)kp * 1024;
    const int totalM = batch * p->oh * p->ow;
    const int plane = p->oh * p->ow;
    const int paired_kp = kp & ~1;
    const bool direct_plane = batch == 1 && p->oh == p->ih && p->ow == p->iw &&
                              p->srcYStep == p->iw * p->packCUnit;
    const int ic_blocks = (p->ic + 31) / 32;
    if (use_pack64_1x1_strided_fast(p, kp, batch, ic_blocks)) {
        fill_im2col_activation_1x1_pack64_strided_tiles(vtcm_activation, src, p, tile_start,
                                                        tile_count, kp, batch);
        return;
    }
    if (direct_plane && kp == 1 && g_max_num_workers > 1 && tile_count >= 2) {
        fill_im2col_activation_1x1_pack64_kp1_direct_parallel(vtcm_activation, src, p, tile_start,
                                                              tile_count, batch);
        return;
    }
    if (direct_plane && paired_kp == kp && paired_kp >= 2 && g_max_num_workers > 1 &&
        tile_count * (paired_kp / 2) >= 2) {
        fill_im2col_activation_1x1_pack64_direct_parallel(vtcm_activation, src, p, tile_start,
                                                          tile_count, kp, batch);
        return;
    }
    for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
        __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
        const int tileStart = (tile_start + local_tile) * 32;
        int validRows = totalM - tileStart;
        if (validRows > 32) {
            validRows = 32;
        }
        if (validRows <= 0) {
            continue;
        }

        for (int kk = 0; kk < paired_kp; kk += 2) {
            __fp16* tile0 = tile_base + (size_t)kk * 1024;
            __fp16* tile1 = tile0 + 1024;
            if (validRows < 32) {
                memset(tile0, 0, 1024 * sizeof(__fp16));
                memset(tile1, 0, 1024 * sizeof(__fp16));
            }
            if (direct_plane) {
                const uint8_t* src_base = src + ((size_t)(kk / 2) * p->srcZStep +
                                                 (size_t)tileStart * p->packCUnit) * sizeof(__fp16);
                int r = 0;
                for (; r <= validRows - 2; r += 2) {
                    HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
                    HVX_Vector v1 = vmem((const HVX_Vector*)(src_base + (size_t)(r + 1) * p->packCUnit * sizeof(__fp16)));
                    HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
                    vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                    vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
                }
                if (r < validRows) {
                    HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
                    HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
                    vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                    vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
                }
                continue;
            }
            int r = 0;
            for (; r <= validRows - 2; r += 2) {
                const int global0 = tileStart + r;
                const int global1 = global0 + 1;
                const int ob0 = global0 / plane;
                const int ob1 = global1 / plane;
                const int plane0 = global0 - ob0 * plane;
                const int plane1 = global1 - ob1 * plane;
                const int oy0 = plane0 / p->ow;
                const int oy1 = plane1 / p->ow;
                const int ox0 = plane0 - oy0 * p->ow;
                const int ox1 = plane1 - oy1 * p->ow;
                const int sy0 = oy0 * p->strideY - p->padY;
                const int sx0 = ox0 * p->strideX - p->padX;
                const int sy1 = oy1 * p->strideY - p->padY;
                const int sx1 = ox1 * p->strideX - p->padX;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob0, sy0, sx0, kk)
                                          : nullptr;
                const uint8_t* src1 = (sy1 >= 0 && sy1 < p->ih && sx1 >= 0 && sx1 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob1, sy1, sx1, kk)
                                          : nullptr;
                HVX_Vector v0 = src0 ? vmem((const HVX_Vector*)src0) : Q6_V_vzero();
                HVX_Vector v1 = src1 ? vmem((const HVX_Vector*)src1) : Q6_V_vzero();
                HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
                vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
            }
            if (r < validRows) {
                const int global0 = tileStart + r;
                const int ob0 = global0 / plane;
                const int plane0 = global0 - ob0 * plane;
                const int oy0 = plane0 / p->ow;
                const int ox0 = plane0 - oy0 * p->ow;
                const int sy0 = oy0 * p->strideY - p->padY;
                const int sx0 = ox0 * p->strideX - p->padX;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob0, sy0, sx0, kk)
                                          : nullptr;
                HVX_Vector v0 = src0 ? vmem((const HVX_Vector*)src0) : Q6_V_vzero();
                HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
                vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
            }
        }
        if (paired_kp < kp) {
            if (direct_plane) {
                const int kk = paired_kp;
                __fp16* tile0 = tile_base + (size_t)kk * 1024;
                if (validRows < 32) {
                    memset(tile0, 0, 1024 * sizeof(__fp16));
                }
                const uint8_t* src_base = src + ((size_t)(kk / 2) * p->srcZStep +
                                                 (size_t)tileStart * p->packCUnit) * sizeof(__fp16);
                int r = 0;
                for (; r <= validRows - 2; r += 2) {
                    HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
                    HVX_Vector v1 = vmem((const HVX_Vector*)(src_base + (size_t)(r + 1) * p->packCUnit * sizeof(__fp16)));
                    HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
                    vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                }
                if (r < validRows) {
                    HVX_Vector v0 = vmem((const HVX_Vector*)(src_base + (size_t)r * p->packCUnit * sizeof(__fp16)));
                    HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
                    vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
                }
            } else {
                fill_im2col_activation_kk_range(tile_base, src, p, tileStart, validRows, kp,
                                                paired_kp, kp, kp, plane);
            }
        }
    }
}

static inline const uint8_t* input_pack64_block0_ptr(const uint8_t* src, const Im2ColParameter* p,
                                                     int ob, int sy, int sx) {
    return src + ((size_t)(ob * p->ih + sy) * p->srcYStep +
                  (size_t)sx * p->packCUnit) * sizeof(__fp16);
}

static void fill_im2col_activation_pack64_ic1_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                                    const Im2ColParameter* p, int tile_start,
                                                    int tile_count, int kp, int batch) {
    const size_t tile_elems = (size_t)kp * 1024;
    const int totalM = batch * p->oh * p->ow;
    const int plane = p->oh * p->ow;
    const int kernel_count = p->kernelX * p->kernelY;
    for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
        __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
        const int tileStart = (tile_start + local_tile) * 32;
        int validRows = totalM - tileStart;
        if (validRows > 32) {
            validRows = 32;
        }
        if (validRows <= 0) {
            continue;
        }

        int ob_list[32];
        int base_y_list[32];
        int base_x_list[32];
        for (int m = 0; m < validRows; ++m) {
            const int global_m = tileStart + m;
            const int ob = global_m / plane;
            const int plane_offset = global_m - ob * plane;
            const int oy = plane_offset / p->ow;
            const int ox = plane_offset - oy * p->ow;
            ob_list[m] = ob;
            base_y_list[m] = oy * p->strideY - p->padY;
            base_x_list[m] = ox * p->strideX - p->padX;
        }

        for (int kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
            const int ky = kernel_index / p->kernelX;
            const int kx = kernel_index - ky * p->kernelX;
            const int ky_offset = ky * p->dilateY;
            const int kx_offset = kx * p->dilateX;
            __fp16* tile0 = tile_base + (size_t)kernel_index * 1024;
            if (validRows < 32) {
                memset(tile0, 0, 1024 * sizeof(__fp16));
            }
            int r = 0;
            for (; r <= validRows - 2; r += 2) {
                const int sy0 = base_y_list[r] + ky_offset;
                const int sx0 = base_x_list[r] + kx_offset;
                const int sy1 = base_y_list[r + 1] + ky_offset;
                const int sx1 = base_x_list[r + 1] + kx_offset;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? input_pack64_block0_ptr(src, p, ob_list[r], sy0, sx0)
                                          : nullptr;
                const uint8_t* src1 = (sy1 >= 0 && sy1 < p->ih && sx1 >= 0 && sx1 < p->iw)
                                          ? input_pack64_block0_ptr(src, p, ob_list[r + 1], sy1, sx1)
                                          : nullptr;
                HVX_Vector v0 = src0 ? vmem((const HVX_Vector*)src0) : Q6_V_vzero();
                HVX_Vector v1 = src1 ? vmem((const HVX_Vector*)src1) : Q6_V_vzero();
                HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
                vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
            }
            if (r < validRows) {
                const int sy0 = base_y_list[r] + ky_offset;
                const int sx0 = base_x_list[r] + kx_offset;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? input_pack64_block0_ptr(src, p, ob_list[r], sy0, sx0)
                                          : nullptr;
                HVX_Vector v0 = src0 ? vmem((const HVX_Vector*)src0) : Q6_V_vzero();
                HVX_VectorPair vp = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, 64);
                vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
            }
        }
    }
}

static inline void store_pack64_im2col_row_pair(__fp16* tile0, __fp16* tile1, int r,
                                                const uint8_t* src0, const uint8_t* src1) {
    HVX_Vector v0 = src0 ? vmem((const HVX_Vector*)src0) : Q6_V_vzero();
    HVX_Vector v1 = src1 ? vmem((const HVX_Vector*)src1) : Q6_V_vzero();
    HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
    vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
    if (tile1) {
        vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
}

static void fill_im2col_activation_pack64_1d_unit_range(__fp16* vtcm_activation, const uint8_t* src,
                                                        const Im2ColParameter* p,
                                                        int tile_start, int tile_count, int kp, int batch,
                                                        int ic_blocks, int plane,
                                                        int unit_start, int unit_end) {
    const size_t tile_elems = (size_t)kp * 1024;
    const int totalM = batch * plane;
    const int kernel_count = p->kernelY;
    const int pairs_per_kernel = (ic_blocks + 1) / 2;
    const int pair_count = kernel_count * pairs_per_kernel;
    const int unit_count = tile_count * pair_count;
    if (unit_start < 0) {
        unit_start = 0;
    }
    if (unit_end > unit_count) {
        unit_end = unit_count;
    }

    int local_tile = unit_start / pair_count;
    int pair_start = unit_start - local_tile * pair_count;
    while (local_tile < tile_count && unit_start < unit_end) {
        __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
        const int tileStart = (tile_start + local_tile) * 32;
        int validRows = totalM - tileStart;
        if (validRows > 32) {
            validRows = 32;
        }
        if (validRows <= 0) {
            unit_start = (local_tile + 1) * pair_count;
            ++local_tile;
            pair_start = 0;
            continue;
        }

        int pair_end = unit_end - local_tile * pair_count;
        if (pair_end > pair_count) {
            pair_end = pair_count;
        }
        for (int pair_idx = pair_start; pair_idx < pair_end; ++pair_idx) {
            const int ky = pair_idx / pairs_per_kernel;
            const int ic_block = (pair_idx - ky * pairs_per_kernel) * 2;
            const int kk = ky * ic_blocks + ic_block;
            __fp16* tile0 = tile_base + (size_t)kk * 1024;
            __fp16* tile1 = (ic_block + 1 < ic_blocks) ? tile0 + 1024 : nullptr;
            if (validRows < 32) {
                memset(tile0, 0, 1024 * sizeof(__fp16));
                if (tile1) {
                    memset(tile1, 0, 1024 * sizeof(__fp16));
                }
            }

            if ((plane & 1) == 0) {
                int ob = tileStart / plane;
                int oy = tileStart - ob * plane;
                const int src_y_stride_bytes = p->srcYStep * (int)sizeof(__fp16);
                for (int r = 0; r <= validRows - 2; r += 2) {
                    const int sy0 = oy - p->padY + ky;
                    const uint8_t* src0 = nullptr;
                    const uint8_t* src1 = nullptr;
                    if (sy0 >= 0 && sy0 + 1 < p->ih) {
                        src0 = src + input_block_offset_fp16(p, ob, sy0, 0, ic_block);
                        src1 = src0 + src_y_stride_bytes;
                    } else {
                        if (sy0 >= 0 && sy0 < p->ih) {
                            src0 = src + input_block_offset_fp16(p, ob, sy0, 0, ic_block);
                        }
                        const int sy1 = sy0 + 1;
                        if (sy1 >= 0 && sy1 < p->ih) {
                            src1 = src + input_block_offset_fp16(p, ob, sy1, 0, ic_block);
                        }
                    }
                    store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
                    oy += 2;
                    if (oy >= plane) {
                        oy -= plane;
                        ++ob;
                    }
                }
            } else {
                int ob = tileStart / plane;
                int oy = tileStart - ob * plane;
                for (int r = 0; r <= validRows - 2; r += 2) {
                    const int sy0 = oy - p->padY + ky;
                    int ob1 = ob;
                    int oy1 = oy + 1;
                    if (oy1 >= plane) {
                        oy1 = 0;
                        ++ob1;
                    }
                    const int sy1 = oy1 - p->padY + ky;
                    const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih)
                                              ? src + input_block_offset_fp16(p, ob, sy0, 0, ic_block)
                                              : nullptr;
                    const uint8_t* src1 = (sy1 >= 0 && sy1 < p->ih)
                                              ? src + input_block_offset_fp16(p, ob1, sy1, 0, ic_block)
                                              : nullptr;
                    store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
                    oy += 2;
                    if (oy >= plane) {
                        oy -= plane;
                        ++ob;
                    }
                }
                if (validRows & 1) {
                    const int r = validRows - 1;
                    const int sy0 = oy - p->padY + ky;
                    const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih)
                                              ? src + input_block_offset_fp16(p, ob, sy0, 0, ic_block)
                                              : nullptr;
                    store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
                }
            }
        }
        unit_start = (local_tile + 1) * pair_count;
        ++local_tile;
        pair_start = 0;
    }
}

static void fill_im2col_activation_pack64_unit_range(__fp16* vtcm_activation, const uint8_t* src,
                                                     const Im2ColParameter* p,
                                                     int tile_start, int tile_count, int kp, int batch,
                                                     int ic_blocks, int plane,
                                                     int unit_start, int unit_end) {
    const size_t tile_elems = (size_t)kp * 1024;
    const int totalM = batch * p->oh * p->ow;
    const int kernel_count = p->kernelX * p->kernelY;
    const int pairs_per_kernel = (ic_blocks + 1) / 2;
    const int pair_count = kernel_count * pairs_per_kernel;
    const int unit_count = tile_count * pair_count;
    if (unit_start < 0) {
        unit_start = 0;
    }
    if (unit_end > unit_count) {
        unit_end = unit_count;
    }
    int local_tile = unit_start / pair_count;
    int pair_start = unit_start - local_tile * pair_count;
    while (local_tile < tile_count && unit_start < unit_end) {
        __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
        const int tileStart = (tile_start + local_tile) * 32;
        int validRows = totalM - tileStart;
        if (validRows > 32) {
            validRows = 32;
        }
        if (validRows <= 0) {
            unit_start = (local_tile + 1) * pair_count;
            ++local_tile;
            pair_start = 0;
            continue;
        }

        int ob_list[32];
        int base_y_list[32];
        int base_x_list[32];
        for (int m = 0; m < validRows; ++m) {
            const int global_m = tileStart + m;
            const int ob = global_m / plane;
            const int plane_offset = global_m - ob * plane;
            const int oy = plane_offset / p->ow;
            const int ox = plane_offset - oy * p->ow;
            ob_list[m] = ob;
            base_y_list[m] = oy * p->strideY - p->padY;
            base_x_list[m] = ox * p->strideX - p->padX;
        }

        int pair_end = unit_end - local_tile * pair_count;
        if (pair_end > pair_count) {
            pair_end = pair_count;
        }
        for (int pair_idx = pair_start; pair_idx < pair_end; ++pair_idx) {
            const int kernel_index = pair_idx / pairs_per_kernel;
            const int ic_block = (pair_idx - kernel_index * pairs_per_kernel) * 2;
            const int ky = kernel_index / p->kernelX;
            const int kx = kernel_index - ky * p->kernelX;
            const int ky_offset = ky * p->dilateY;
            const int kx_offset = kx * p->dilateX;
            const int kk = kernel_index * ic_blocks + ic_block;
            __fp16* tile0 = tile_base + (size_t)kk * 1024;
            __fp16* tile1 = (ic_block + 1 < ic_blocks) ? tile0 + 1024 : nullptr;
            if (validRows < 32) {
                memset(tile0, 0, 1024 * sizeof(__fp16));
                if (tile1) {
                    memset(tile1, 0, 1024 * sizeof(__fp16));
                }
            }
            int r = 0;
            for (; r <= validRows - 2; r += 2) {
                const int sy0 = base_y_list[r] + ky_offset;
                const int sx0 = base_x_list[r] + kx_offset;
                const int sy1 = base_y_list[r + 1] + ky_offset;
                const int sx1 = base_x_list[r + 1] + kx_offset;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob_list[r], sy0, sx0, ic_block)
                                          : nullptr;
                const uint8_t* src1 = (sy1 >= 0 && sy1 < p->ih && sx1 >= 0 && sx1 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob_list[r + 1], sy1, sx1, ic_block)
                                          : nullptr;
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
            }
            if (r < validRows) {
                const int sy0 = base_y_list[r] + ky_offset;
                const int sx0 = base_x_list[r] + kx_offset;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob_list[r], sy0, sx0, ic_block)
                                          : nullptr;
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
            }
        }
        unit_start = (local_tile + 1) * pair_count;
        ++local_tile;
        pair_start = 0;
    }
}

static inline bool use_pack64_3x3s1p1_ic1_fast(const Im2ColParameter* p, int kp, int batch,
                                                int ic_blocks, int plane) {
    return batch == 1 && p->packCUnit == 64 && ic_blocks == 1 && kp == 9 && p->kernelX == 3 &&
           p->kernelY == 3 && p->strideX == 1 && p->strideY == 1 && p->dilateX == 1 &&
           p->dilateY == 1 && p->padX == 1 && p->padY == 1 && p->ow == p->iw &&
           p->oh == p->ih && plane == p->oh * p->ow && (p->ow & 31) == 0;
}

static inline bool use_pack64_3x3s1p1_fast(const Im2ColParameter* p, int kp, int batch,
                                           int ic_blocks, int plane) {
    return batch == 1 && p->packCUnit == 64 && kp == 9 * ic_blocks && p->kernelX == 3 &&
           p->kernelY == 3 && p->strideX == 1 && p->strideY == 1 && p->dilateX == 1 &&
           p->dilateY == 1 && p->padX == 1 && p->padY == 1 && p->ow == p->iw &&
           p->oh == p->ih && plane == p->oh * p->ow;
}

static inline void fill_im2col_activation_pack64_3x3s1p1_one_tile(__fp16* vtcm_activation,
                                                                   const uint8_t* src,
                                                                   const Im2ColParameter* p,
                                                                   int tile_start, int local_tile,
                                                                   int ic_blocks) {
    const int kp = 9 * ic_blocks;
    __fp16* tile_base = vtcm_activation + (size_t)local_tile * kp * 1024;
    const int tileStart = (tile_start + local_tile) * 32;
    int validRows = p->oh * p->ow - tileStart;
    if (validRows > 32) {
        validRows = 32;
    }
    if (validRows <= 0) {
        return;
    }
    const int oy = tileStart / p->ow;
    const int ox = tileStart - oy * p->ow;
    const bool fullRowTile = ox + validRows - 1 < p->ow;
    const int pairs_per_kernel = (ic_blocks + 1) / 2;
    for (int kernel_index = 0; kernel_index < 9; ++kernel_index) {
        const int ky = kernel_index / 3;
        const int kx = kernel_index - ky * 3;
        const int sy = oy + ky - 1;
        const int sxBase = ox + kx - 1;
        for (int pair_idx = 0; pair_idx < pairs_per_kernel; ++pair_idx) {
            const int ic_block = pair_idx * 2;
            const int kk = kernel_index * ic_blocks + ic_block;
            __fp16* tile0 = tile_base + (size_t)kk * 1024;
            __fp16* tile1 = (ic_block + 1 < ic_blocks) ? tile0 + 1024 : nullptr;
            if (validRows < 32) {
                memset(tile0, 0, 1024 * sizeof(__fp16));
                if (tile1 != nullptr) {
                    memset(tile1, 0, 1024 * sizeof(__fp16));
                }
            }
            if (fullRowTile && sy >= 0 && sy < p->ih && sxBase >= 0 && sxBase + validRows - 1 < p->iw) {
                const uint8_t* src_base = src + input_block_offset_fp16(p, 0, sy, sxBase, ic_block);
                for (int r = 0; r <= validRows - 2; r += 2) {
                    const uint8_t* src0 = src_base + (size_t)r * p->packCUnit * sizeof(__fp16);
                    const uint8_t* src1 = src0 + (size_t)p->packCUnit * sizeof(__fp16);
                    store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
                }
                if (validRows & 1) {
                    const int r = validRows - 1;
                    const uint8_t* src0 = src_base + (size_t)r * p->packCUnit * sizeof(__fp16);
                    store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
                }
                continue;
            }
            int ob = 0;
            int y = oy;
            int x = ox;
            for (int r = 0; r <= validRows - 2; r += 2) {
                const int sy0 = y + ky - 1;
                const int sx0 = x + kx - 1;
                int y1 = y;
                int x1 = x + 1;
                if (x1 >= p->ow) {
                    x1 = 0;
                    ++y1;
                }
                const int sy1 = y1 + ky - 1;
                const int sx1 = x1 + kx - 1;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob, sy0, sx0, ic_block)
                                          : nullptr;
                const uint8_t* src1 = (sy1 >= 0 && sy1 < p->ih && sx1 >= 0 && sx1 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob, sy1, sx1, ic_block)
                                          : nullptr;
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, src1);
                x += 2;
                if (x >= p->ow) {
                    if (p->ow == 1) {
                        x = 0;
                        y += 2;
                    } else {
                        x -= p->ow;
                        ++y;
                    }
                }
            }
            if (validRows & 1) {
                const int r = validRows - 1;
                const int sy0 = y + ky - 1;
                const int sx0 = x + kx - 1;
                const uint8_t* src0 = (sy0 >= 0 && sy0 < p->ih && sx0 >= 0 && sx0 < p->iw)
                                          ? src + input_block_offset_fp16(p, ob, sy0, sx0, ic_block)
                                          : nullptr;
                store_pack64_im2col_row_pair(tile0, tile1, r, src0, nullptr);
            }
        }
    }
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    int tile_start;
    int ic_blocks;
    worker_synctoken_t sync_ctx;
} HmxIm2Col3x3GeneralFastTaskState;

typedef struct {
    HmxIm2Col3x3GeneralFastTaskState* state;
    int tile_begin;
    int tile_end;
} HmxIm2Col3x3GeneralFastTask;

static void fill_im2col_activation_pack64_3x3s1p1_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2Col3x3GeneralFastTask* task = (HmxIm2Col3x3GeneralFastTask*)data;
    HmxIm2Col3x3GeneralFastTaskState* state = task->state;
    for (int local_tile = task->tile_begin; local_tile < task->tile_end; ++local_tile) {
        fill_im2col_activation_pack64_3x3s1p1_one_tile(state->vtcm_activation, state->src,
                                                       state->p, state->tile_start, local_tile,
                                                       state->ic_blocks);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_pack64_3x3s1p1_tiles(__fp16* vtcm_activation,
                                                        const uint8_t* src,
                                                        const Im2ColParameter* p,
                                                        int tile_start, int tile_count,
                                                        int ic_blocks) {
    int task_count = (int)g_max_num_workers;
    if (task_count > tile_count) {
        task_count = tile_count;
    }
    if (task_count <= 1) {
        for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
            fill_im2col_activation_pack64_3x3s1p1_one_tile(vtcm_activation, src, p,
                                                           tile_start, local_tile, ic_blocks);
        }
        return;
    }
    HmxIm2Col3x3GeneralFastTaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_start = tile_start;
    state.ic_blocks = ic_blocks;
    HmxIm2Col3x3GeneralFastTask* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2Col3x3GeneralFastTask, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_pack64_3x3s1p1_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int tile_begin = (int)((int64_t)i * tile_count / task_count);
        const int tile_end = (int)((int64_t)(i + 1) * tile_count / task_count);
        tasks[i].state = &state;
        tasks[i].tile_begin = tile_begin;
        tasks[i].tile_end = tile_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

static inline void fill_im2col_activation_pack64_3x3s1p1_ic1_one_tile(__fp16* vtcm_activation,
                                                                       const uint8_t* src,
                                                                       const Im2ColParameter* p,
                                                                       int tile_start, int local_tile) {
    __fp16* tile_base = vtcm_activation + (size_t)local_tile * 9 * 1024;
    const int tileStart = (tile_start + local_tile) * 32;
    const int oy = tileStart / p->ow;
    const int ox = tileStart - oy * p->ow;
    for (int kernel_index = 0; kernel_index < 9; ++kernel_index) {
        const int ky = kernel_index / 3;
        const int kx = kernel_index - ky * 3;
        const int sy = oy + ky - 1;
        __fp16* tile0 = tile_base + (size_t)kernel_index * 1024;
        if (sy < 0 || sy >= p->ih) {
            memset(tile0, 0, 1024 * sizeof(__fp16));
            continue;
        }
        const int sxBase = ox + kx - 1;
        if (sxBase >= 0 && sxBase + 31 < p->iw) {
            const uint8_t* src_base = input_pack64_block0_ptr(src, p, 0, sy, sxBase);
            for (int r = 0; r < 32; r += 2) {
                const uint8_t* src0 = src_base + (size_t)r * p->packCUnit * sizeof(__fp16);
                const uint8_t* src1 = src0 + (size_t)p->packCUnit * sizeof(__fp16);
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, src1);
            }
        } else {
            for (int r = 0; r < 32; r += 2) {
                const int sx0 = sxBase + r;
                const int sx1 = sx0 + 1;
                const uint8_t* src0 = (sx0 >= 0 && sx0 < p->iw) ? input_pack64_block0_ptr(src, p, 0, sy, sx0) : nullptr;
                const uint8_t* src1 = (sx1 >= 0 && sx1 < p->iw) ? input_pack64_block0_ptr(src, p, 0, sy, sx1) : nullptr;
                store_pack64_im2col_row_pair(tile0, nullptr, r, src0, src1);
            }
        }
    }
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    int tile_start;
    worker_synctoken_t sync_ctx;
} HmxIm2Col3x3FastTaskState;

typedef struct {
    HmxIm2Col3x3FastTaskState* state;
    int tile_begin;
    int tile_end;
} HmxIm2Col3x3FastTask;

static void fill_im2col_activation_pack64_3x3s1p1_ic1_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2Col3x3FastTask* task = (HmxIm2Col3x3FastTask*)data;
    HmxIm2Col3x3FastTaskState* state = task->state;
    for (int local_tile = task->tile_begin; local_tile < task->tile_end; ++local_tile) {
        fill_im2col_activation_pack64_3x3s1p1_ic1_one_tile(state->vtcm_activation, state->src,
                                                           state->p, state->tile_start, local_tile);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_pack64_3x3s1p1_ic1_tiles(__fp16* vtcm_activation,
                                                            const uint8_t* src,
                                                            const Im2ColParameter* p,
                                                            int tile_start, int tile_count) {
    int task_count = (int)g_max_num_workers;
    if (task_count > tile_count) {
        task_count = tile_count;
    }
    if (task_count <= 1) {
        for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
            fill_im2col_activation_pack64_3x3s1p1_ic1_one_tile(vtcm_activation, src, p, tile_start, local_tile);
        }
        return;
    }
    HmxIm2Col3x3FastTaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_start = tile_start;
    HmxIm2Col3x3FastTask* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2Col3x3FastTask, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_pack64_3x3s1p1_ic1_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int tile_begin = (int)((int64_t)i * tile_count / task_count);
        const int tile_end = (int)((int64_t)(i + 1) * tile_count / task_count);
        tasks[i].state = &state;
        tasks[i].tile_begin = tile_begin;
        tasks[i].tile_end = tile_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

typedef struct {
    __fp16* vtcm_activation;
    const uint8_t* src;
    const Im2ColParameter* p;
    int tile_start;
    int tile_count;
    int kp;
    int batch;
    int ic_blocks;
    int plane;
    worker_synctoken_t sync_ctx;
} HmxIm2ColPack64TaskState;

typedef struct {
    HmxIm2ColPack64TaskState* state;
    int unit_start;
    int unit_end;
} HmxIm2ColPack64Task;

static void fill_im2col_activation_pack64_worker(void* data, int worker_index) {
    (void)worker_index;
    HmxIm2ColPack64Task* task = (HmxIm2ColPack64Task*)data;
    HmxIm2ColPack64TaskState* state = task->state;
    if (state->p->packCUnit == 64 && state->p->kernelX == 1 && state->p->iw == 1 &&
        state->p->ow == 1 && state->p->strideY == 1 && state->p->dilateX == 1 && state->p->dilateY == 1) {
        fill_im2col_activation_pack64_1d_unit_range(state->vtcm_activation, state->src, state->p,
                                                    state->tile_start, state->tile_count, state->kp,
                                                    state->batch, state->ic_blocks, state->plane,
                                                    task->unit_start, task->unit_end);
    } else {
        fill_im2col_activation_pack64_unit_range(state->vtcm_activation, state->src, state->p,
                                                 state->tile_start, state->tile_count, state->kp,
                                                 state->batch, state->ic_blocks, state->plane,
                                                 task->unit_start, task->unit_end);
    }
    worker_pool_synctoken_jobdone(&state->sync_ctx);
}

static void fill_im2col_activation_pack64_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                                const Im2ColParameter* p,
                                                int tile_start, int tile_count, int kp, int batch,
                                                int ic_blocks, int plane) {
    if (use_pack64_3x3s1p1_ic1_fast(p, kp, batch, ic_blocks, plane)) {
        fill_im2col_activation_pack64_3x3s1p1_ic1_tiles(vtcm_activation, src, p, tile_start, tile_count);
        return;
    }
    if (use_pack64_3x3s1p1_fast(p, kp, batch, ic_blocks, plane)) {
        fill_im2col_activation_pack64_3x3s1p1_tiles(vtcm_activation, src, p, tile_start, tile_count, ic_blocks);
        return;
    }
    const int kernel_count = p->kernelX * p->kernelY;
    const int pairs_per_kernel = (ic_blocks + 1) / 2;
    const int pair_count = kernel_count * pairs_per_kernel;
    const int unit_count = tile_count * pair_count;
    int task_count = 1;
    if (unit_count >= 9 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        if (task_count > unit_count) {
            task_count = unit_count;
        }
    }
    if (task_count <= 1) {
        if (p->packCUnit == 64 && p->kernelX == 1 && p->iw == 1 &&
            p->ow == 1 && p->strideY == 1 && p->dilateX == 1 && p->dilateY == 1) {
            fill_im2col_activation_pack64_1d_unit_range(vtcm_activation, src, p, tile_start, tile_count,
                                                        kp, batch, ic_blocks, plane, 0, unit_count);
        } else {
            fill_im2col_activation_pack64_unit_range(vtcm_activation, src, p, tile_start, tile_count,
                                                     kp, batch, ic_blocks, plane, 0, unit_count);
        }
        return;
    }

    HmxIm2ColPack64TaskState state = {};
    state.vtcm_activation = vtcm_activation;
    state.src = src;
    state.p = p;
    state.tile_start = tile_start;
    state.tile_count = tile_count;
    state.kp = kp;
    state.batch = batch;
    state.ic_blocks = ic_blocks;
    state.plane = plane;

    HmxIm2ColPack64Task* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2ColPack64Task, task_count);
    worker_pool_job_t job;
    job.fptr = fill_im2col_activation_pack64_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int i = 0; i < task_count; ++i) {
        const int unit_start = (int)((int64_t)i * unit_count / task_count);
        const int unit_end = (int)((int64_t)(i + 1) * unit_count / task_count);
        tasks[i].state = &state;
        tasks[i].unit_start = unit_start;
        tasks[i].unit_end = unit_end;
        job.dptr = tasks + i;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

static void fill_im2col_activation_tiles(__fp16* vtcm_activation, const uint8_t* src,
                                         const Im2ColParameter* p,
                                         int tile_start, int tile_count, int kp, int batch) {
    const size_t tile_elems = (size_t)kp * 1024;
    const int ic_blocks = (p->ic + 31) / 32;
    const int totalM = batch * p->oh * p->ow;
    const int plane = p->oh * p->ow;
    if (p->packCUnit == 64 && p->kernelX == 1 && p->kernelY == 1 &&
        p->dilateX == 1 && p->dilateY == 1 && kp == ic_blocks) {
        fill_im2col_activation_1x1_pack64_tiles(vtcm_activation, src, p, tile_start, tile_count, kp, batch);
        return;
    }
    if (p->packCUnit == 64 && ic_blocks == 1 && kp == p->kernelX * p->kernelY) {
        if (kp >= 9 && g_max_num_workers > 1) {
            fill_im2col_activation_pack64_tiles(vtcm_activation, src, p, tile_start, tile_count, kp, batch,
                                                ic_blocks, plane);
        } else {
            fill_im2col_activation_pack64_ic1_tiles(vtcm_activation, src, p, tile_start, tile_count, kp, batch);
        }
        return;
    }
    if (p->packCUnit == 64 && kp == p->kernelX * p->kernelY * ic_blocks) {
        fill_im2col_activation_pack64_tiles(vtcm_activation, src, p, tile_start, tile_count, kp, batch,
                                            ic_blocks, plane);
        return;
    }
    int task_count = 1;
    if (kp >= 9 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        if (task_count > kp) {
            task_count = kp;
        }
    }
    if (task_count > 1) {
        HmxIm2ColFillTaskState state = {};
        state.vtcm_activation = vtcm_activation;
        state.src = src;
        state.p = p;
        state.tile_elems = tile_elems;
        state.tile_start = tile_start;
        state.tile_count = tile_count;
        state.totalM = totalM;
        state.kp = kp;
        state.ic_blocks = ic_blocks;
        state.plane = plane;

        HmxIm2ColFillTask* tasks = WORKER_POOL_STACK_ALLOC(HmxIm2ColFillTask, task_count);
        worker_pool_job_t job;
        job.fptr = fill_im2col_activation_worker;
        worker_pool_synctoken_init(&state.sync_ctx, task_count);
        const int kk_per_task = (kp + task_count - 1) / task_count;
        for (int i = 0; i < task_count; ++i) {
            const int kk_start = i * kk_per_task;
            int kk_end = kk_start + kk_per_task;
            if (kk_end > kp) {
                kk_end = kp;
            }
            tasks[i].state = &state;
            tasks[i].kk_start = kk_start;
            tasks[i].kk_end = kk_end;
            job.dptr = tasks + i;
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&state.sync_ctx);
        return;
    }
    for (int local_tile = 0; local_tile < tile_count; ++local_tile) {
        __fp16* tile_base = vtcm_activation + (size_t)local_tile * tile_elems;
        const int tileStart = (tile_start + local_tile) * 32;
        int validCount = totalM - tileStart;
        if (validCount > 32) {
            validCount = 32;
        }
        if (validCount <= 0) {
            continue;
        }

        fill_im2col_activation_kk_range(tile_base, src, p, tileStart, validCount, kp, 0, kp, ic_blocks, plane);
    }
}

static void fill_weight_tiles_fp16(__fp16* vtcm_weight, const __fp16* src_weight, const HmxIm2ColConvParam* params,
                                   int oy_start, int oy_end, int kp) {
    (void)params;
    const int kernel_block_size = 32 * 32;
    const size_t tileSize = (size_t)kp * kernel_block_size;
    const int tileCount = oy_end - oy_start;
    const size_t bytesPerTile = tileSize * sizeof(__fp16);
    if (tileCount <= 0 || bytesPerTile == 0) {
        return;
    }

    if (bytesPerTile < 1024 || bytesPerTile >= (1u << 24)) {
        for (int oy = oy_start; oy < oy_end; ++oy) {
            const int local_oy = oy - oy_start;
            memcpy(vtcm_weight + (size_t)local_oy * tileSize, src_weight + (size_t)oy * tileSize, bytesPerTile);
        }
        return;
    }

    static const int kMaxWeightDmaDescs = 64;
    _Alignas(64) dma_desc_1d_t descs[kMaxWeightDmaDescs];
    for (int base = 0; base < tileCount; base += kMaxWeightDmaDescs) {
        int count = tileCount - base;
        if (count > kMaxWeightDmaDescs) {
            count = kMaxWeightDmaDescs;
        }
        for (int i = 0; i < count; ++i) {
            const int tile = base + i;
            memset(&descs[i], 0, sizeof(descs[i]));
            descs[i].next = (i + 1 < count) ? (uint32_t)&descs[i + 1] : 0;
            descs[i].length = (uint32_t)bytesPerTile;
            descs[i].type = DMA_DESC_TYPE_1D;
            descs[i].src_bypass = 1;
            descs[i].dst_bypass = 1;
            descs[i].ordered = 1;
            descs[i].dstate = DMA_DESC_DSTATE_PENDING;
            descs[i].src = (uint32_t)(src_weight + (size_t)(oy_start + tile) * tileSize);
            descs[i].dst = (uint32_t)(vtcm_weight + (size_t)tile * tileSize);
        }
        dma_wait_for_idle();
        dmstart(&descs[0]);
        dma_wait_for_idle();
    }
}

static inline int compute_store_tiles_fp16(uint8_t* dst, const uint8_t* bias,
                                           const HmxIm2ColConvParam* params, int M, int pack, int kp,
                                           int ox_start, int ox_end, int oy_start, int oy_end,
                                           __fp16* vtcm_activation, __fp16* vtcm_weight,
                                           __fp16* vtcm_output) {
    for (int ox = ox_start; ox < ox_end; ++ox) {
        const int local_ox = ox - ox_start;
        __fp16* act_tile = vtcm_activation + (size_t)local_ox * kp * 1024;
        for (int oy = oy_start; oy < oy_end; ++oy) {
            const int local_oy = oy - oy_start;
            __fp16* weight_tile = vtcm_weight + (size_t)local_oy * kp * 1024;
            if (pack == 64 && oy + 1 < oy_end && ((oy * 32) & 63) == 0) {
                __fp16* weight_tile_next = vtcm_weight + (size_t)(local_oy + 1) * kp * 1024;
                __fp16* vtcm_output_next = vtcm_output + 1024;
                compute_hmx_tile_fp16(act_tile, weight_tile, kp, vtcm_output);
                compute_hmx_tile_fp16(act_tile, weight_tile_next, kp, vtcm_output_next);
                const __fp16* biasPtr0 = bias ? ((const __fp16*)bias) + oy * 32 : nullptr;
                const __fp16* biasPtr1 = bias ? ((const __fp16*)bias) + (oy + 1) * 32 : nullptr;
                int storeRet = store_output_tile_pair_fp16(dst, vtcm_output, vtcm_output_next,
                                                           biasPtr0, biasPtr1, M, ox, oy,
                                                           params->relu, params->relu6,
                                                           params->outputBytes);
                if (storeRet != AEE_SUCCESS) {
                    return storeRet;
                }
                ++oy;
                continue;
            }
            compute_hmx_tile_fp16(act_tile, weight_tile, kp, vtcm_output);
            const __fp16* biasPtr = bias ? ((const __fp16*)bias) + oy * 32 : nullptr;
            int storeRet = store_output_tile_fp16(dst, vtcm_output, biasPtr, M, ox, oy, pack,
                                                  params->relu, params->relu6,
                                                  params->outputBytes);
            if (storeRet != AEE_SUCCESS) {
                return storeRet;
            }
        }
    }
    return AEE_SUCCESS;
}

extern "C" {

int hmx_im2col_convolution_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const uint8_t *bias,
                                const HmxIm2ColConvParam* params) {
    const Im2ColParameter* p = &params->im2col;
    const int batch = params->batch > 0 ? params->batch : 1;
    const int pack = p->packCUnit;
    const int M = batch * p->oh * p->ow;
    const int N = params->oc;
    const int kp = p->kernelCountUnit > 0 ? p->kernelCountUnit : (p->kernelX * p->kernelY * ((p->ic + 31) / 32));
    const int np = (N + 31) / 32;
    const int mp = (M + 31) / 32;

    int np_chunk = params->np > 0 ? params->np : 1;
    int mp_chunk = params->mp > 0 ? params->mp : 1;
    if (np_chunk > np) np_chunk = np;
    if (mp_chunk > mp) mp_chunk = mp;
    const int ox_chunk_count = (mp + mp_chunk - 1) / mp_chunk;
    const int oy_chunk_count = (np + np_chunk - 1) / np_chunk;
    const int64_t activation_outer_cost = (int64_t)mp + (int64_t)ox_chunk_count * np;
    const int64_t weight_outer_cost = (int64_t)oy_chunk_count * mp + (int64_t)np;
    const bool reuse_activation = activation_outer_cost <= weight_outer_cost;

    uint8_t *vtcm_ptr = (uint8_t *)vtcm_manager_get_vtcm_base();
    __fp16* vtcm_weight = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, (size_t)np_chunk * kp * 1024 * sizeof(int16_t));
    __fp16* vtcm_activation = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, (size_t)mp_chunk * kp * 1024 * sizeof(int16_t));
    __fp16* vtcm_output = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 4096);
    __fp16* vtcm_scales = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 256);

    hmx_manager_enable_execution();
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
    hmx_unit_acquire();
    hmx_set_output_scales(vtcm_scales);

    if (reuse_activation) {
        for (int ox_start = 0; ox_start < mp; ox_start += mp_chunk) {
            const int ox_end = (ox_start + mp_chunk > mp) ? mp : (ox_start + mp_chunk);
            fill_im2col_activation_tiles(vtcm_activation, src, p, ox_start, ox_end - ox_start, kp, batch);
            for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
                const int oy_end = (oy_start + np_chunk > np) ? np : (oy_start + np_chunk);
                fill_weight_tiles_fp16(vtcm_weight, (const __fp16*)weight, params, oy_start, oy_end, kp);
                int ret = compute_store_tiles_fp16(dst, bias, params, M, pack, kp, ox_start, ox_end,
                                                   oy_start, oy_end, vtcm_activation, vtcm_weight,
                                                   vtcm_output);
                if (ret != AEE_SUCCESS) {
                    hmx_unit_release();
                    hmx_manager_disable_execution();
                    return ret;
                }
            }
        }
    } else {
        for (int oy_start = 0; oy_start < np; oy_start += np_chunk) {
            const int oy_end = (oy_start + np_chunk > np) ? np : (oy_start + np_chunk);
            fill_weight_tiles_fp16(vtcm_weight, (const __fp16*)weight, params, oy_start, oy_end, kp);

            for (int ox_start = 0; ox_start < mp; ox_start += mp_chunk) {
                const int ox_end = (ox_start + mp_chunk > mp) ? mp : (ox_start + mp_chunk);
                fill_im2col_activation_tiles(vtcm_activation, src, p, ox_start, ox_end - ox_start, kp, batch);
                int ret = compute_store_tiles_fp16(dst, bias, params, M, pack, kp, ox_start, ox_end,
                                                   oy_start, oy_end, vtcm_activation, vtcm_weight,
                                                   vtcm_output);
                if (ret != AEE_SUCCESS) {
                    hmx_unit_release();
                    hmx_manager_disable_execution();
                    return ret;
                }
            }
        }
    }

    hmx_unit_release();
    hmx_manager_disable_execution();
    return 0;
}

AEEResult htp_ops_im2col_convolution_fp16(uint8_t* output, uint8_t* input, uint8_t* weight, uint8_t* bias,
                                          const HmxIm2ColConvParam* params) {
    return hmx_im2col_convolution_fp16(output, input, weight, bias, params);
}

int htp_ops_conv1x1_direct_fp16(uint8_t* output, uint8_t* input, uint8_t* weight, uint8_t* bias,
                                const HmxIm2ColConvParam* params) {
    return hmx_im2col_convolution_fp16(output, input, weight, bias, params);
}

} // extern "C"
