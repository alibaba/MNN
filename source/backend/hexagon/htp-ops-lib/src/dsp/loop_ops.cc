#include "dsp/mmap_mgr.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <string.h>
#include <remote.h>
#include "region_ops.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_math.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

extern "C" {

AEEResult htp_ops_binary_blit(uint8_t* dst, const uint8_t* src0, const uint8_t* src1,
                              uint8_t* region, int32_t regionCount, int32_t bytes, int32_t opType);

static inline int32_t loop_read_int32(const uint8_t* ptr) {
  int32_t val;
  memcpy(&val, ptr, sizeof(int32_t));
  return val;
}

static inline _Float16 htp_ops_loop_binary_apply_fp16(_Float16 a, _Float16 b, int32_t opType) {
    switch (opType) {
        case 1: return a + b;
        case 2: return a - b;
        case 3: return a * b;
        case 11: {
            float v = (float)a - (float)b;
            return (_Float16)(v * v);
        }
        case 4: return a / b;
        case 5: return a > b ? a : b;
        case 6: return a < b ? a : b;
        case 7: {
            float a_f = (float)a;
            float b_f = (float)b;
            float sig_b = 1.0f / (1.0f + expf(-b_f));
            return (_Float16)(a_f * b_f * sig_b);
        }
        default: return a;
    }
}

static inline int32_t htp_ops_loop_binary_apply_int32(int32_t a, int32_t b, int32_t opType) {
    switch (opType) {
        case 1: return a + b;
        case 2: return a - b;
        case 3: return a * b;
        case 11: {
            int32_t v = a - b;
            return v * v;
        }
        default: return a;
    }
}

static inline HVX_Vector htp_ops_loop_binary_mul_silu_fp16_vec(HVX_Vector v0, HVX_Vector v1) {
    HVX_Vector zero_v = Q6_V_vzero();
    HVX_Vector one_v = Q6_Vh_vsplat_R(0x3c00);
    HVX_VectorPred q_v1_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v1);
    HVX_Vector neg_v1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(zero_v, v1));
    HVX_Vector z = Q6_V_vmux_QVV(q_v1_lt_0, v1, neg_v1);
    HVX_Vector log2e_v = Q6_Vh_vsplat_R(0x3dc5);
    HVX_Vector exp_arg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(z, log2e_v));
    HVX_Vector exp_val = hvx_my_exp2_vhf(exp_arg);
    HVX_Vector denom = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(one_v, exp_val));
    HVX_Vector inv_denom = hvx_my_inv_vhf(denom);
    HVX_Vector two_v = Q6_Vh_vsplat_R(0x4000);
    HVX_Vector dy = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(denom, inv_denom));
    HVX_Vector two_minus_dy = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(two_v, dy));
    inv_denom = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(inv_denom, two_minus_dy));
    HVX_Vector num = Q6_V_vmux_QVV(q_v1_lt_0, exp_val, one_v);
    HVX_Vector q_sig_v = Q6_Vqf16_vmpy_VhfVhf(num, inv_denom);
    HVX_Vector q_v1_sig_v = Q6_Vqf16_vmpy_Vqf16Vhf(q_sig_v, v1);
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(q_v1_sig_v, v0));
}

static inline HVX_Vector htp_ops_loop_binary_apply_vec(HVX_Vector a, HVX_Vector b, int32_t opType) {
    switch (opType) {
        case 1:
            return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
        case 2:
            return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
        case 3:
            return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
        case 11: {
            HVX_Vector sub = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
            return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sub, sub));
        }
        case 4: {
            HVX_Vector inv_b = hvx_my_inv_vhf(b);
            return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, inv_b));
        }
        case 5:
            return Q6_Vhf_vmax_VhfVhf(a, b);
        case 6:
            return Q6_Vhf_vmin_VhfVhf(a, b);
        case 7:
            return htp_ops_loop_binary_mul_silu_fp16_vec(a, b);
        default:
            return a;
    }
}

static inline bool htp_ops_loop_binary_row_fast_fp16(uint8_t* dstY, const uint8_t* src0Y,
                                                    const uint8_t* src1Y, const HtpOpsLoopParam* lp,
                                                    int32_t opType, int32_t bytes) {
    if (bytes != 2 || lp->dstStrideXYZ[2] != 2 || lp->sizeXYZ[2] <= 0) {
        return false;
    }
    const bool src0Contig = lp->src0StrideXYZ[2] == 2;
    const bool src1Contig = lp->src1StrideXYZ[2] == 2;
    const bool src0Scalar = lp->src0StrideXYZ[2] == 0;
    const bool src1Scalar = lp->src1StrideXYZ[2] == 0;
    if (!((src0Contig || src0Scalar) && (src1Contig || src1Scalar)) || (src0Scalar && src1Scalar)) {
        return false;
    }

    const int size = lp->sizeXYZ[2];
    const int vecElems = __HVX_LENGTH__ / (int)sizeof(__fp16);
    const int vecEnd = size & -vecElems;
    __fp16* dst = (__fp16*)dstY;
    const __fp16* src0 = (const __fp16*)src0Y;
    const __fp16* src1 = (const __fp16*)src1Y;
    int x = 0;

    if (src0Contig && src1Scalar) {
        const uint16_t scalarBits = *(const uint16_t*)src1;
        HVX_Vector v1 = Q6_Vh_vsplat_R(scalarBits);
        if (opType == 4) {
            v1 = hvx_my_inv_vhf(v1);
            for (; x < vecEnd; x += vecElems) {
                HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + x));
                HVX_Vector vr = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v1));
                vmemu((HVX_Vector*)(dst + x)) = vr;
            }
        } else {
            for (; x < vecEnd; x += vecElems) {
                HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + x));
                HVX_Vector vr = htp_ops_loop_binary_apply_vec(v0, v1, opType);
                vmemu((HVX_Vector*)(dst + x)) = vr;
            }
        }
        for (; x < size; ++x) {
            dst[x] = htp_ops_loop_binary_apply_fp16(src0[x], src1[0], opType);
        }
        return true;
    }

    if (src0Scalar && src1Contig) {
        const uint16_t scalarBits = *(const uint16_t*)src0;
        HVX_Vector v0 = Q6_Vh_vsplat_R(scalarBits);
        for (; x < vecEnd; x += vecElems) {
            HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + x));
            HVX_Vector vr = htp_ops_loop_binary_apply_vec(v0, v1, opType);
            vmemu((HVX_Vector*)(dst + x)) = vr;
        }
        for (; x < size; ++x) {
            dst[x] = htp_ops_loop_binary_apply_fp16(src0[0], src1[x], opType);
        }
        return true;
    }

    if (src0Contig && src1Contig) {
        for (; x < vecEnd; x += vecElems) {
            HVX_Vector v0 = vmemu((const HVX_Vector*)(src0 + x));
            HVX_Vector v1 = vmemu((const HVX_Vector*)(src1 + x));
            HVX_Vector vr = htp_ops_loop_binary_apply_vec(v0, v1, opType);
            vmemu((HVX_Vector*)(dst + x)) = vr;
        }
        for (; x < size; ++x) {
            dst[x] = htp_ops_loop_binary_apply_fp16(src0[x], src1[x], opType);
        }
        return true;
    }

    return false;
}

static inline void htp_ops_loop_binary_region(uint8_t* dstBase, const uint8_t* src0Base,
                                              const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                              int32_t opType, int32_t bytes, bool continuous,
                                              int optX, int optY, int optZ) {
    const int optXBytes = optX * bytes;
    if (continuous) {
        for (int z = 0; z < optZ; ++z) {
            const uint8_t* src0Z = src0Base + z * lp->src0StrideXYZ[0];
            const uint8_t* src1Z = src1Base + z * lp->src1StrideXYZ[0];
            uint8_t* dstZ = dstBase + z * lp->dstStrideXYZ[0];
            for (int y = 0; y < optY; ++y) {
                const uint8_t* src0Y = src0Z + y * lp->src0StrideXYZ[1];
                const uint8_t* src1Y = src1Z + y * lp->src1StrideXYZ[1];
                uint8_t* dstY = dstZ + y * lp->dstStrideXYZ[1];
                int x = 0;
                if (bytes == 2) {
                    const int vecBytes = __HVX_LENGTH__;
                    const int vecEndBytes = optXBytes & -vecBytes;
                    for (; x < vecEndBytes; x += vecBytes) {
                        HVX_Vector v0 = vmemu((const HVX_Vector*)(src0Y + x));
                        HVX_Vector v1 = vmemu((const HVX_Vector*)(src1Y + x));
                        vmemu((HVX_Vector*)(dstY + x)) = htp_ops_loop_binary_apply_vec(v0, v1, opType);
                    }
                }
                if (bytes == 2) {
                    __fp16* dstFp16 = (__fp16*)(dstY + x);
                    const __fp16* src0Fp16 = (const __fp16*)(src0Y + x);
                    const __fp16* src1Fp16 = (const __fp16*)(src1Y + x);
                    for (int i = 0; x + i * bytes < optXBytes; ++i) {
                        dstFp16[i] = htp_ops_loop_binary_apply_fp16(src0Fp16[i], src1Fp16[i], opType);
                    }
                } else if (bytes == 4) {
                    int32_t* dstI32 = (int32_t*)dstY;
                    const int32_t* src0I32 = (const int32_t*)src0Y;
                    const int32_t* src1I32 = (const int32_t*)src1Y;
                    for (int i = 0; i < optX; ++i) {
                        dstI32[i] = htp_ops_loop_binary_apply_int32(src0I32[i], src1I32[i], opType);
                    }
                } else {
                    for (; x < optXBytes; x += bytes) {
                        memcpy(dstY + x, src0Y + x, bytes);
                    }
                }
            }
        }
        return;
    }

    for (int z = 0; z < lp->sizeXYZ[0]; ++z) {
        const uint8_t* src0Z = src0Base + z * lp->src0StrideXYZ[0];
        const uint8_t* src1Z = src1Base + z * lp->src1StrideXYZ[0];
        uint8_t* dstZ = dstBase + z * lp->dstStrideXYZ[0];
        for (int y = 0; y < lp->sizeXYZ[1]; ++y) {
            const uint8_t* src0Y = src0Z + y * lp->src0StrideXYZ[1];
            const uint8_t* src1Y = src1Z + y * lp->src1StrideXYZ[1];
            uint8_t* dstY = dstZ + y * lp->dstStrideXYZ[1];
            if (htp_ops_loop_binary_row_fast_fp16(dstY, src0Y, src1Y, lp, opType, bytes)) {
                continue;
            }
            for (int x = 0; x < lp->sizeXYZ[2]; ++x) {
                const uint8_t* s0 = src0Y + x * lp->src0StrideXYZ[2];
                const uint8_t* s1 = src1Y + x * lp->src1StrideXYZ[2];
                uint8_t* d = dstY + x * lp->dstStrideXYZ[2];
                if (bytes == 2) {
                    *(__fp16*)d = htp_ops_loop_binary_apply_fp16(*(const __fp16*)s0, *(const __fp16*)s1, opType);
                } else if (bytes == 4) {
                    *(int32_t*)d = htp_ops_loop_binary_apply_int32(*(const int32_t*)s0, *(const int32_t*)s1, opType);
                } else {
                    memcpy(d, s0, bytes);
                }
            }
        }
    }
}

typedef struct {
    uint8_t* dst;
    uint8_t* src0;
    uint8_t* src1;
    uint8_t* iter0;
    uint8_t* iter1;
    uint8_t* iter2;
    const HtpOpsLoopParam* lp;
    int32_t opType;
    int32_t bytes;
    bool continuous;
    int optX;
    int optY;
    int optZ;
    worker_synctoken_t sync_ctx;
} HtpOpsLoopBinaryTaskState;

typedef struct {
    HtpOpsLoopBinaryTaskState* state;
    int begin;
    int end;
} HtpOpsLoopBinaryFixedTask;

static inline void htp_ops_loop_binary_run_iter(HtpOpsLoopBinaryTaskState* state, int iter) {
    const HtpOpsLoopParam* lp = state->lp;
    const uint8_t* srcIter0 = state->iter0;
    const uint8_t* srcIter1 = state->iter1;
    const uint8_t* srcIter2 = state->iter2;

    int32_t it0 = srcIter0 ? loop_read_int32(srcIter0 + iter * sizeof(int32_t)) : iter;
    int32_t it1 = srcIter1 ? loop_read_int32(srcIter1 + iter * sizeof(int32_t)) : iter;
    int32_t it2 = srcIter2 ? loop_read_int32(srcIter2 + iter * sizeof(int32_t)) : iter;

    int32_t outOff = (int32_t)it0 * lp->cmdSteps[0] + lp->cmdViewOffset[0];
    if (outOff < 0 || outOff >= lp->outputElementSize) {
        return;
    }
    int32_t in0Off = (int32_t)it1 * lp->cmdSteps[1] + lp->cmdViewOffset[1];
    if (in0Off < 0 || in0Off >= lp->input0Size) {
        return;
    }
    int32_t in1Off = (int32_t)it2 * lp->cmdSteps[2] + lp->cmdViewOffset[2];
    if (in1Off < 0 || in1Off >= lp->input1Size) {
        return;
    }

    uint8_t* dstBase = state->dst + (int64_t)outOff * state->bytes;
    const uint8_t* src0Base = state->src0 + (int64_t)in0Off * state->bytes;
    const uint8_t* src1Base = state->src1 + (int64_t)in1Off * state->bytes;
    htp_ops_loop_binary_region(dstBase, src0Base, src1Base, lp, state->opType, state->bytes,
                               state->continuous, state->optX, state->optY, state->optZ);
}

static void htp_ops_loop_binary_worker(void* data, int worker_index) {
    (void)worker_index;
    HtpOpsLoopBinaryFixedTask* task = (HtpOpsLoopBinaryFixedTask*)data;
    for (int iter = task->begin; iter < task->end; ++iter) {
        htp_ops_loop_binary_run_iter(task->state, iter);
    }
    worker_pool_synctoken_jobdone(&(task->state->sync_ctx));
}

static inline bool htp_ops_loop_binary_try_parallel(uint8_t* dst, uint8_t* src0, uint8_t* src1,
                                                    uint8_t* iter0, uint8_t* iter1, uint8_t* iter2,
                                                    const HtpOpsLoopParam* lp, int32_t opType,
                                                    int32_t bytes, bool continuous,
                                                    int optX, int optY, int optZ) {
    if (g_max_num_workers <= 1 || lp->loopNumber < 2 || optX <= 0 || optY <= 0 || optZ <= 0) {
        return false;
    }
    const int workPerIter = optX * optY * optZ;
    if (workPerIter <= 0 || (int64_t)workPerIter * lp->loopNumber < 4096) {
        return false;
    }
    int nTasks = (int)g_max_num_workers;
    if (nTasks > lp->loopNumber) {
        nTasks = lp->loopNumber;
    }
    if (nTasks <= 1) {
        return false;
    }

    HtpOpsLoopBinaryTaskState state = {};
    state.dst = dst;
    state.src0 = src0;
    state.src1 = src1;
    state.iter0 = iter0;
    state.iter1 = iter1;
    state.iter2 = iter2;
    state.lp = lp;
    state.opType = opType;
    state.bytes = bytes;
    state.continuous = continuous;
    state.optX = optX;
    state.optY = optY;
    state.optZ = optZ;

    worker_pool_job_t job;
    job.fptr = htp_ops_loop_binary_worker;
    HtpOpsLoopBinaryFixedTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsLoopBinaryFixedTask, nTasks);
    worker_pool_synctoken_init(&(state.sync_ctx), nTasks);
    const int loopsPerTask = (lp->loopNumber + nTasks - 1) / nTasks;
    for (int i = 0; i < nTasks; ++i) {
        const int begin = i * loopsPerTask;
        int end = begin + loopsPerTask;
        if (end > lp->loopNumber) {
            end = lp->loopNumber;
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

static inline bool htp_ops_loop_check_element(int64_t baseOffset, int64_t extraBytes,
                                              int32_t bytes, int64_t elementSize) {
    if (bytes <= 0 || elementSize <= 0 || extraBytes % bytes != 0) {
        return false;
    }
    int64_t elementOffset = baseOffset + extraBytes / bytes;
    return elementOffset >= 0 && elementOffset < elementSize;
}

static inline float htp_ops_loop_reduce_sum2_f32(HVX_Vector acc0, HVX_Vector acc1) {
    HVX_Vector v = Q6_Vsf_vadd_VsfVsf(acc0, acc1);
    v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64)));
    v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32)));
    v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16)));
    v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8)));
    v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4)));
    union { HVX_Vector v; float f[32]; } u = { .v = v };
    return u.f[0];
}

static inline int htp_ops_loop_up_div(int v, int d) {
    return (v + d - 1) / d;
}

static inline void htp_ops_loop_hmx_pack_activation_k64(__fp16* dst, const uint8_t* src0Base,
                                                       const HtpOpsLoopParam* lp, int eBase,
                                                       int validRows) {
    __fp16* tile0 = dst;
    __fp16* tile1 = dst + 1024;
    if (validRows < 32) {
        memset(tile0, 0, 1024 * sizeof(__fp16));
        memset(tile1, 0, 1024 * sizeof(__fp16));
    }
    int r = 0;
    for (; r <= validRows - 2; r += 2) {
        const uint8_t* src0 = src0Base + (int64_t)(eBase + r) * lp->src0StrideXYZ[0];
        const uint8_t* src1 = src0Base + (int64_t)(eBase + r + 1) * lp->src0StrideXYZ[0];
        HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
        HVX_Vector v1 = vmemu((const HVX_Vector*)src1);
        HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
    if (r < validRows) {
        const uint8_t* src0 = src0Base + (int64_t)(eBase + r) * lp->src0StrideXYZ[0];
        HVX_Vector v0 = vmemu((const HVX_Vector*)src0);
        HVX_Vector v1 = Q6_V_vzero();
        HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
        vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
        vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
}

static inline void htp_ops_loop_hmx_pack_activation_tile(__fp16* dst, const uint8_t* src0Base,
                                                        const HtpOpsLoopParam* lp, int K, int kt,
                                                        int eBase, int validRows) {
    __fp16* tile = dst + (size_t)kt * 1024;
    memset(tile, 0, 1024 * sizeof(__fp16));
    const int kBegin = kt * 32;
    int kRemain = K - kBegin;
    if (kRemain > 32) {
        kRemain = 32;
    }
    for (int r = 0; r < validRows; ++r) {
        const uint8_t* srcRow = src0Base + (int64_t)(eBase + r) * lp->src0StrideXYZ[0];
        for (int k = 0; k < kRemain; ++k) {
            const int dstIndex = (r / 2) * 64 + k * 2 + (r & 1);
            tile[dstIndex] = *(const __fp16*)(srcRow + (int64_t)(kBegin + k) * lp->src0StrideXYZ[1]);
        }
    }
}

static inline void htp_ops_loop_hmx_pack_weight_tile(__fp16* dst, const uint8_t* src1Base,
                                                    const HtpOpsLoopParam* lp, int K, int N, int nt, int kt) {
    __fp16* tile = dst + ((size_t)nt * htp_ops_loop_up_div(K, 32) + kt) * 1024;
    memset(tile, 0, 1024 * sizeof(__fp16));
    int kBegin = kt * 32;
    int kRemain = K - kBegin;
    if (kRemain > 32) {
        kRemain = 32;
    }
    int nBegin = nt * 32;
    int nRemain = N - nBegin;
    if (nRemain > 32) {
        nRemain = 32;
    }
    for (int k = 0; k < kRemain; ++k) {
        const uint8_t* srcRow = src1Base + (int64_t)(kBegin + k) * lp->src1StrideXYZ[1];
        for (int c = 0; c < nRemain; ++c) {
            int dstIndex = (k / 2) * 64 + c * 2 + (k & 1);
            tile[dstIndex] = *(const __fp16*)(srcRow + (int64_t)(nBegin + c) * lp->src1StrideXYZ[2]);
        }
    }
}

static inline void htp_ops_loop_hmx_store_output_tile(uint8_t* dstBase, const __fp16* vtcmOutput,
                                                     const HtpOpsLoopParam* lp, int eBase,
                                                     int validRows, int N, int nt) {
    int nBegin = nt * 32;
    int nRemain = N - nBegin;
    if (nRemain > 32) {
        nRemain = 32;
    }
    if (nBegin == 0 && N <= 16 && lp->dstStrideXYZ[0] == N * 2 && lp->dstStrideXYZ[2] == 2) {
        const uint32_t rowBytes = (uint32_t)(N * (int)sizeof(__fp16));
        const HVX_Vector* src = (const HVX_Vector*)vtcmOutput;
        int r = 0;
        for (; r <= validRows - 2; r += 2) {
            HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
            uint8_t* dst0 = dstBase + (int64_t)(eBase + r) * lp->dstStrideXYZ[0];
            uint8_t* dst1 = dstBase + (int64_t)(eBase + r + 1) * lp->dstStrideXYZ[0];
            vstu_variable(dst0, rowBytes, v);
            vstu_variable(dst1, rowBytes, Q6_V_valign_VVR(v, v, 64));
        }
        if (r < validRows) {
            HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
            uint8_t* dst0 = dstBase + (int64_t)(eBase + r) * lp->dstStrideXYZ[0];
            vstu_variable(dst0, rowBytes, v);
        }
        return;
    }
    for (int r = 0; r < validRows; ++r) {
        uint8_t* dstRow = dstBase + (int64_t)(eBase + r) * lp->dstStrideXYZ[0];
        for (int c = 0; c < nRemain; ++c) {
            int srcIndex = (r / 2) * 64 + c * 2 + (r & 1);
            *(__fp16*)(dstRow + (int64_t)(nBegin + c) * lp->dstStrideXYZ[2]) = vtcmOutput[srcIndex];
        }
    }
}

static inline bool htp_ops_loop_matmul_hmx_general(uint8_t* dstBase, const uint8_t* src0Base,
                                                   const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                   int64_t outOff, int64_t in0Off, int64_t in1Off) {
    const int E = lp->sizeXYZ[0];
    const int K = lp->sizeXYZ[1];
    const int N = lp->sizeXYZ[2];
    if (E <= 0 || K <= 0 || N <= 0) {
        return false;
    }
    if (lp->dstStrideXYZ[2] != 2 || lp->src0StrideXYZ[1] != 2 || lp->src1StrideXYZ[2] != 2) {
        return false;
    }
    if (lp->dstStrideXYZ[0] < 0 || lp->src0StrideXYZ[0] < 0 || lp->src1StrideXYZ[1] < 0) {
        return false;
    }
    const int64_t dstLast = (int64_t)(E - 1) * lp->dstStrideXYZ[0] + (int64_t)(N - 1) * lp->dstStrideXYZ[2];
    const int64_t src0Last = (int64_t)(E - 1) * lp->src0StrideXYZ[0] + (int64_t)(K - 1) * lp->src0StrideXYZ[1];
    const int64_t src1Last = (int64_t)(K - 1) * lp->src1StrideXYZ[1] + (int64_t)(N - 1) * lp->src1StrideXYZ[2];
    if (!htp_ops_loop_check_element(outOff, dstLast, 2, lp->outputElementSize) ||
        !htp_ops_loop_check_element(in0Off, src0Last, 2, lp->input0Size) ||
        !htp_ops_loop_check_element(in1Off, src1Last, 2, lp->input1Size)) {
        return false;
    }
    const int64_t work = (int64_t)E * K * N;
    if (work < 32768) {
        return false;
    }

    const int kp = htp_ops_loop_up_div(K, 32);
    const int np = htp_ops_loop_up_div(N, 32);
    uint8_t* vtcmPtr = (uint8_t*)vtcm_manager_get_vtcm_base();
    __fp16* vtcmActivation = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)kp * 1024 * sizeof(__fp16));
    __fp16* vtcmWeight = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)np * kp * 1024 * sizeof(__fp16));
    __fp16* vtcmOutput = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 1024 * sizeof(__fp16));
    __fp16* vtcmScales = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 256);
    if (vtcmActivation == nullptr || vtcmWeight == nullptr || vtcmOutput == nullptr || vtcmScales == nullptr) {
        return false;
    }

    hmx_manager_enable_execution();
    hmx_unit_acquire();
    hmx_init_column_scales(vtcmScales, Q6_V_vsplat_R(0x3c00));
    hmx_set_output_scales(vtcmScales);

    for (int nt = 0; nt < np; ++nt) {
        for (int kt = 0; kt < kp; ++kt) {
            htp_ops_loop_hmx_pack_weight_tile(vtcmWeight, src1Base, lp, K, N, nt, kt);
        }
    }

    for (int eBase = 0; eBase < E; eBase += 32) {
        int validRows = E - eBase;
        if (validRows > 32) {
            validRows = 32;
        }
        for (int kt = 0; kt < kp; ++kt) {
            htp_ops_loop_hmx_pack_activation_tile(vtcmActivation, src0Base, lp, K, kt, eBase, validRows);
        }
        for (int nt = 0; nt < np; ++nt) {
            hmx_load_tiles_fp16(vtcmActivation, vtcmWeight + (size_t)nt * kp * 1024, kp);
            hmx_consume_accumulator_fp16(vtcmOutput);
            htp_ops_loop_hmx_store_output_tile(dstBase, vtcmOutput, lp, eBase, validRows, N, nt);
        }
    }

    hmx_unit_release();
    hmx_manager_disable_execution();
    return true;
}

static inline bool htp_ops_loop_matmul_hmx_small(uint8_t* dstBase, const uint8_t* src0Base,
                                                 const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                 int64_t outOff, int64_t in0Off, int64_t in1Off) {
    const int E = lp->sizeXYZ[0];
    const int K = lp->sizeXYZ[1];
    const int N = lp->sizeXYZ[2];
    if (E <= 0 || K != 64 || N <= 0 || N > 16) {
        return false;
    }
    if (lp->dstStrideXYZ[2] != 2 || lp->src0StrideXYZ[1] != 2 || lp->src0StrideXYZ[0] != K * 2 ||
        lp->src1StrideXYZ[2] != 2) {
        return false;
    }
    if (lp->dstStrideXYZ[0] < 0 || lp->src0StrideXYZ[0] < 0 || lp->src1StrideXYZ[1] < 0) {
        return false;
    }
    const int64_t dstLast = (int64_t)(E - 1) * lp->dstStrideXYZ[0] + (int64_t)(N - 1) * lp->dstStrideXYZ[2];
    const int64_t src0Last = (int64_t)(E - 1) * lp->src0StrideXYZ[0] + (int64_t)(K - 1) * lp->src0StrideXYZ[1];
    const int64_t src1Last = (int64_t)(K - 1) * lp->src1StrideXYZ[1] + (int64_t)(N - 1) * lp->src1StrideXYZ[2];
    if (!htp_ops_loop_check_element(outOff, dstLast, 2, lp->outputElementSize) ||
        !htp_ops_loop_check_element(in0Off, src0Last, 2, lp->input0Size) ||
        !htp_ops_loop_check_element(in1Off, src1Last, 2, lp->input1Size)) {
        return false;
    }

    const int kp = htp_ops_loop_up_div(K, 32);
    const int np = htp_ops_loop_up_div(N, 32);
    uint8_t* vtcmPtr = (uint8_t*)vtcm_manager_get_vtcm_base();
    __fp16* vtcmActivation = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)kp * 1024 * sizeof(__fp16));
    __fp16* vtcmWeight = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)np * kp * 1024 * sizeof(__fp16));
    __fp16* vtcmOutput = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 1024 * sizeof(__fp16));
    __fp16* vtcmScales = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 256);
    if (vtcmActivation == nullptr || vtcmWeight == nullptr || vtcmOutput == nullptr || vtcmScales == nullptr) {
        return false;
    }

    hmx_manager_enable_execution();
    hmx_unit_acquire();
    hmx_init_column_scales(vtcmScales, Q6_V_vsplat_R(0x3c00));
    hmx_set_output_scales(vtcmScales);

    for (int nt = 0; nt < np; ++nt) {
        for (int kt = 0; kt < kp; ++kt) {
            htp_ops_loop_hmx_pack_weight_tile(vtcmWeight, src1Base, lp, K, N, nt, kt);
        }
    }

    for (int eBase = 0; eBase < E; eBase += 32) {
        int validRows = E - eBase;
        if (validRows > 32) {
            validRows = 32;
        }
        htp_ops_loop_hmx_pack_activation_k64(vtcmActivation, src0Base, lp, eBase, validRows);
        for (int nt = 0; nt < np; ++nt) {
            for (int kt = 0; kt < kp; ++kt) {
                hmx_load_tiles_fp16(vtcmActivation + (size_t)kt * 1024,
                                    vtcmWeight + ((size_t)nt * kp + kt) * 1024, 1);
            }
            hmx_consume_accumulator_fp16(vtcmOutput);
            htp_ops_loop_hmx_store_output_tile(dstBase, vtcmOutput, lp, eBase, validRows, N, nt);
        }
    }

    hmx_unit_release();
    hmx_manager_disable_execution();
    return true;
}

typedef struct {
    uint8_t* dstBase;
    const uint8_t* src0Base;
    const uint8_t* src1Base;
    const HtpOpsLoopParam* lp;
    int L;
    int H;
    int eStart;
    int eEnd;
    worker_synctoken_t* sync_ctx;
} HtpOpsLoopMatmulHContigTask;

static inline void htp_ops_loop_matmul_fast_h_contiguous_range(uint8_t* dstBase, const uint8_t* src0Base,
                                                               const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                               int L, int H, int eStart, int eEnd) {
    const int vecElems = __HVX_LENGTH__ / (int)sizeof(__fp16);
    for (int e = eStart; e < eEnd; ++e) {
        uint8_t* dstRow = dstBase + (int64_t)e * lp->dstStrideXYZ[0];
        const uint8_t* src0Row = src0Base + (int64_t)e * lp->src0StrideXYZ[0];
        int h = 0;
        for (; h + vecElems <= H; h += vecElems) {
            HVX_Vector acc = Q6_V_vzero();
            for (int l = 0; l < L; ++l) {
                const __fp16* aPtr = (const __fp16*)(src0Row + (int64_t)l * lp->src0StrideXYZ[1]);
                uint16_t aBits = *(const uint16_t*)aPtr;
                HVX_Vector aVec = Q6_Vh_vsplat_R(aBits);
                const uint8_t* bPtr = src1Base + (int64_t)l * lp->src1StrideXYZ[1] + (int64_t)h * lp->src1StrideXYZ[2];
                HVX_Vector bVec = vmemu((const HVX_Vector*)bPtr);
                HVX_Vector prod = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(aVec, bVec));
                acc = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(acc, prod));
            }
            vmemu((HVX_Vector*)(dstRow + (int64_t)h * lp->dstStrideXYZ[2])) = acc;
        }
        for (; h < H; ++h) {
            float sum = 0.0f;
            for (int l = 0; l < L; ++l) {
                const int64_t src0Extra = (int64_t)e * lp->src0StrideXYZ[0] + (int64_t)l * lp->src0StrideXYZ[1];
                const int64_t src1Extra = (int64_t)l * lp->src1StrideXYZ[1] + (int64_t)h * lp->src1StrideXYZ[2];
                sum += (float)(*(const __fp16*)(src0Base + src0Extra)) *
                       (float)(*(const __fp16*)(src1Base + src1Extra));
            }
            *(__fp16*)(dstRow + (int64_t)h * lp->dstStrideXYZ[2]) = (__fp16)sum;
        }
    }
}

static void htp_ops_loop_matmul_fast_h_contiguous_worker(void* data, int worker_index) {
    (void)worker_index;
    HtpOpsLoopMatmulHContigTask* task = (HtpOpsLoopMatmulHContigTask*)data;
    htp_ops_loop_matmul_fast_h_contiguous_range(task->dstBase, task->src0Base, task->src1Base,
                                                task->lp, task->L, task->H, task->eStart, task->eEnd);
    worker_pool_synctoken_jobdone(task->sync_ctx);
}

static inline bool htp_ops_loop_matmul_fast_h_contiguous(uint8_t* dstBase, const uint8_t* src0Base,
                                                         const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                         int64_t outOff, int64_t in0Off, int64_t in1Off) {
    const int E = lp->sizeXYZ[0];
    const int L = lp->sizeXYZ[1];
    const int H = lp->sizeXYZ[2];
    if (E <= 0 || L <= 0 || H <= 0 || lp->dstStrideXYZ[2] != 2 || lp->src1StrideXYZ[2] != 2) {
        return false;
    }
    if (lp->dstStrideXYZ[0] < 0 || lp->src0StrideXYZ[0] < 0 || lp->src0StrideXYZ[1] < 0 ||
        lp->src1StrideXYZ[1] < 0) {
        return false;
    }
    const int64_t dstLast = (int64_t)(E - 1) * lp->dstStrideXYZ[0] + (int64_t)(H - 1) * lp->dstStrideXYZ[2];
    const int64_t src0Last = (int64_t)(E - 1) * lp->src0StrideXYZ[0] + (int64_t)(L - 1) * lp->src0StrideXYZ[1];
    const int64_t src1Last = (int64_t)(L - 1) * lp->src1StrideXYZ[1] + (int64_t)(H - 1) * lp->src1StrideXYZ[2];
    if (!htp_ops_loop_check_element(outOff, dstLast, 2, lp->outputElementSize) ||
        !htp_ops_loop_check_element(in0Off, src0Last, 2, lp->input0Size) ||
        !htp_ops_loop_check_element(in1Off, src1Last, 2, lp->input1Size)) {
        return false;
    }

    const int64_t work = (int64_t)E * L * H;
    if (g_max_num_workers > 1 && E >= 8 && work >= 32768) {
        int nTasks = (int)g_max_num_workers;
        if (nTasks > E) {
            nTasks = E;
        }
        HtpOpsLoopMatmulHContigTask* tasks = WORKER_POOL_STACK_ALLOC(HtpOpsLoopMatmulHContigTask, nTasks);
        worker_synctoken_t sync_ctx;
        worker_pool_synctoken_init(&sync_ctx, nTasks);
        worker_pool_job_t job;
        job.fptr = htp_ops_loop_matmul_fast_h_contiguous_worker;
        const int rowsPerTask = (E + nTasks - 1) / nTasks;
        for (int i = 0; i < nTasks; ++i) {
            const int eStart = i * rowsPerTask;
            int eEnd = eStart + rowsPerTask;
            if (eEnd > E) {
                eEnd = E;
            }
            tasks[i].dstBase = dstBase;
            tasks[i].src0Base = src0Base;
            tasks[i].src1Base = src1Base;
            tasks[i].lp = lp;
            tasks[i].L = L;
            tasks[i].H = H;
            tasks[i].eStart = eStart;
            tasks[i].eEnd = eEnd;
            tasks[i].sync_ctx = &sync_ctx;
            job.dptr = tasks + i;
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&sync_ctx);
        return true;
    }
    htp_ops_loop_matmul_fast_h_contiguous_range(dstBase, src0Base, src1Base, lp, L, H, 0, E);
    return true;
}

static inline bool htp_ops_loop_matmul_fast_small_h(uint8_t* dstBase, const uint8_t* src0Base,
                                                    const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                    int64_t outOff, int64_t in0Off, int64_t in1Off) {
    const int E = lp->sizeXYZ[0];
    const int L = lp->sizeXYZ[1];
    const int H = lp->sizeXYZ[2];
    const int vecElems = __HVX_LENGTH__ / (int)sizeof(__fp16);
    if (E <= 0 || L <= 0 || H <= 0 || H > 16 || lp->dstStrideXYZ[2] != 2 ||
        lp->src0StrideXYZ[1] != 2 || lp->src1StrideXYZ[2] != 2) {
        return false;
    }
    if (lp->dstStrideXYZ[0] < 0 || lp->src0StrideXYZ[0] < 0 || lp->src1StrideXYZ[1] < 0) {
        return false;
    }
    const int64_t dstLast = (int64_t)(E - 1) * lp->dstStrideXYZ[0] + (int64_t)(H - 1) * lp->dstStrideXYZ[2];
    const int64_t src0Last = (int64_t)(E - 1) * lp->src0StrideXYZ[0] + (int64_t)(L - 1) * lp->src0StrideXYZ[1];
    const int64_t src1Last = (int64_t)(L - 1) * lp->src1StrideXYZ[1] + (int64_t)(H - 1) * lp->src1StrideXYZ[2];
    if (!htp_ops_loop_check_element(outOff, dstLast, 2, lp->outputElementSize) ||
        !htp_ops_loop_check_element(in0Off, src0Last, 2, lp->input0Size) ||
        !htp_ops_loop_check_element(in1Off, src1Last, 2, lp->input1Size)) {
        return false;
    }

    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPred qH = Q6_Q_vsetq_R(H * (int)sizeof(__fp16));
    const int safeRows = (vecElems + H - 1) / H;
    const int vectorLimit = L >= safeRows ? L - safeRows + 1 : 0;
    for (int e = 0; e < E; ++e) {
        const uint8_t* src0Row = src0Base + (int64_t)e * lp->src0StrideXYZ[0];
        uint8_t* dstRow = dstBase + (int64_t)e * lp->dstStrideXYZ[0];
        HVX_Vector acc0 = Q6_V_vzero();
        HVX_Vector acc1 = Q6_V_vzero();
        int l = 0;
        for (; l < vectorLimit; ++l) {
            const __fp16* aPtr = (const __fp16*)(src0Row + (int64_t)l * lp->src0StrideXYZ[1]);
            uint16_t aBits = *(const uint16_t*)aPtr;
            HVX_Vector aVec = Q6_Vh_vsplat_R(aBits);
            const uint8_t* bPtr = src1Base + (int64_t)l * lp->src1StrideXYZ[1];
            HVX_Vector bVec = Q6_V_vmux_QVV(qH, vmemu((const HVX_Vector*)bPtr), zero);
            HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(aVec, bVec);
            HVX_VectorPair prodF = hvx_my_vqf16_to_wsf(prod);
            acc0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc0, Q6_V_lo_W(prodF)));
            acc1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc1, Q6_V_hi_W(prodF)));
        }
        float tail[16] = {0.0f};
        for (; l < L; ++l) {
            const float a = (float)(*(const __fp16*)(src0Row + (int64_t)l * lp->src0StrideXYZ[1]));
            const __fp16* b = (const __fp16*)(src1Base + (int64_t)l * lp->src1StrideXYZ[1]);
            for (int h = 0; h < H; ++h) {
                tail[h] += a * (float)b[h];
            }
        }
        HVX_Vector out = Q6_Vhf_vcvt_VsfVsf(acc0, acc1);
        __fp16 tmp[64] __attribute__((aligned(128)));
        vmem(tmp) = out;
        for (int h = 0; h < H; ++h) {
            tmp[h] = (__fp16)((float)tmp[h] + tail[h]);
        }
        memcpy(dstRow, tmp, (size_t)H * sizeof(__fp16));
    }
    return true;
}

static inline bool htp_ops_loop_matmul_fast_l_contiguous(uint8_t* dstBase, const uint8_t* src0Base,
                                                         const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                                         int64_t outOff, int64_t in0Off, int64_t in1Off) {
    const int E = lp->sizeXYZ[0];
    const int L = lp->sizeXYZ[1];
    const int H = lp->sizeXYZ[2];
    const int vecElems = __HVX_LENGTH__ / (int)sizeof(__fp16);
    if (E <= 0 || L < vecElems || H <= 0 || lp->dstStrideXYZ[2] != 2 ||
        lp->src0StrideXYZ[1] != 2 || lp->src1StrideXYZ[1] != 2) {
        return false;
    }
    if (lp->dstStrideXYZ[0] < 0 || lp->src0StrideXYZ[0] < 0 || lp->src1StrideXYZ[2] < 0) {
        return false;
    }
    const int64_t dstLast = (int64_t)(E - 1) * lp->dstStrideXYZ[0] + (int64_t)(H - 1) * lp->dstStrideXYZ[2];
    const int64_t src0Last = (int64_t)(E - 1) * lp->src0StrideXYZ[0] + (int64_t)(L - 1) * lp->src0StrideXYZ[1];
    const int64_t src1Last = (int64_t)(H - 1) * lp->src1StrideXYZ[2] + (int64_t)(L - 1) * lp->src1StrideXYZ[1];
    if (!htp_ops_loop_check_element(outOff, dstLast, 2, lp->outputElementSize) ||
        !htp_ops_loop_check_element(in0Off, src0Last, 2, lp->input0Size) ||
        !htp_ops_loop_check_element(in1Off, src1Last, 2, lp->input1Size)) {
        return false;
    }

    for (int e = 0; e < E; ++e) {
        const uint8_t* src0Row = src0Base + (int64_t)e * lp->src0StrideXYZ[0];
        uint8_t* dstRow = dstBase + (int64_t)e * lp->dstStrideXYZ[0];
        for (int h = 0; h < H; ++h) {
            const uint8_t* src1Row = src1Base + (int64_t)h * lp->src1StrideXYZ[2];
            HVX_Vector acc0 = Q6_V_vzero();
            HVX_Vector acc1 = Q6_V_vzero();
            int l = 0;
            for (; l + vecElems <= L; l += vecElems) {
                HVX_Vector aVec = vmemu((const HVX_Vector*)(src0Row + (int64_t)l * lp->src0StrideXYZ[1]));
                HVX_Vector bVec = vmemu((const HVX_Vector*)(src1Row + (int64_t)l * lp->src1StrideXYZ[1]));
                HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(aVec, bVec);
                HVX_VectorPair prodF = hvx_my_vqf16_to_wsf(prod);
                acc0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc0, Q6_V_lo_W(prodF)));
                acc1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc1, Q6_V_hi_W(prodF)));
            }
            float sum = htp_ops_loop_reduce_sum2_f32(acc0, acc1);
            for (; l < L; ++l) {
                sum += (float)(*(const __fp16*)(src0Row + (int64_t)l * lp->src0StrideXYZ[1])) *
                       (float)(*(const __fp16*)(src1Row + (int64_t)l * lp->src1StrideXYZ[1]));
            }
            *(__fp16*)(dstRow + (int64_t)h * lp->dstStrideXYZ[2]) = (__fp16)sum;
        }
    }
    return true;
}

static inline void htp_ops_loop_matmul_region(uint8_t* dstBase, const uint8_t* src0Base,
                                              const uint8_t* src1Base, const HtpOpsLoopParam* lp,
                                              int64_t outOff, int64_t in0Off, int64_t in1Off) {
    if (htp_ops_loop_matmul_hmx_small(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    const bool preferHmxGeneral = ((lp->sizeXYZ[1] | lp->sizeXYZ[2]) & 31) != 0 ||
                                  lp->sizeXYZ[1] == 32 || lp->sizeXYZ[2] == 32;
    if (preferHmxGeneral &&
        htp_ops_loop_matmul_hmx_general(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    if (htp_ops_loop_matmul_fast_small_h(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    if (htp_ops_loop_matmul_fast_h_contiguous(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    if (htp_ops_loop_matmul_fast_l_contiguous(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    if (!preferHmxGeneral &&
        htp_ops_loop_matmul_hmx_general(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off)) {
        return;
    }
    const int E = lp->sizeXYZ[0];
    const int L = lp->sizeXYZ[1];
    const int H = lp->sizeXYZ[2];
    for (int e = 0; e < E; ++e) {
        const int64_t dstEOff = (int64_t)e * lp->dstStrideXYZ[0];
        const int64_t src0EOff = (int64_t)e * lp->src0StrideXYZ[0];
        for (int h = 0; h < H; ++h) {
            const int64_t dstExtra = dstEOff + (int64_t)h * lp->dstStrideXYZ[2];
            if (!htp_ops_loop_check_element(outOff, dstExtra, 2, lp->outputElementSize)) {
                continue;
            }
            float sum = 0.0f;
            for (int l = 0; l < L; ++l) {
                const int64_t src0Extra = src0EOff + (int64_t)l * lp->src0StrideXYZ[1];
                const int64_t src1Extra = (int64_t)l * lp->src1StrideXYZ[1] + (int64_t)h * lp->src1StrideXYZ[2];
                if (!htp_ops_loop_check_element(in0Off, src0Extra, 2, lp->input0Size) ||
                    !htp_ops_loop_check_element(in1Off, src1Extra, 2, lp->input1Size)) {
                    continue;
                }
                sum += (float)(*(const __fp16*)(src0Base + src0Extra)) *
                       (float)(*(const __fp16*)(src1Base + src1Extra));
            }
            *(__fp16*)(dstBase + dstExtra) = (__fp16)sum;
        }
    }
}

AEEResult htp_ops_loop_blit(uint8_t* dst, uint8_t* src0, uint8_t* src1,
                            uint8_t* iter0, uint8_t* iter1, uint8_t* iter2,
                            int32_t cmdKind, int32_t opType,
                            int32_t bytes,
                            uint8_t* param) {
    uint8_t* pParam = param;

    const HtpOpsLoopParam* lp = (HtpOpsLoopParam*)pParam;

    int32_t loopNumber = lp->loopNumber;
    if (loopNumber <= 0) {
        return 0;
    }

    uint8_t* pDst = dst;
    uint8_t* pSrc0 = src0;
    uint8_t* pIter0 = iter0;
    uint8_t* pIter1 = iter1;

    const uint8_t* srcIter0 = pIter0;
    const uint8_t* srcIter1 = pIter1;
    if (cmdKind != 0 && cmdKind != 1 && cmdKind != 2) {
        return AEE_EUNSUPPORTED;
    }
    if (cmdKind == 2 && bytes != 2) {
        return AEE_EUNSUPPORTED;
    }

    int optZ = lp->sizeXYZ[0];
    int optY = lp->sizeXYZ[1];
    int optX = lp->sizeXYZ[2];
    bool continuous = (lp->src0StrideXYZ[2] == bytes && lp->dstStrideXYZ[2] == bytes &&
                       (cmdKind == 0 || lp->src1StrideXYZ[2] == bytes));
    if (continuous) {
        if (lp->src0StrideXYZ[1] == optX * bytes && lp->dstStrideXYZ[1] == optX * bytes &&
            (cmdKind == 0 || lp->src1StrideXYZ[1] == optX * bytes)) {
            optX *= optY;
            optY = 1;
            if (lp->src0StrideXYZ[0] == optX * bytes && lp->dstStrideXYZ[0] == optX * bytes &&
                (cmdKind == 0 || lp->src1StrideXYZ[0] == optX * bytes)) {
                optX *= optZ;
                optZ = 1;
            }
        }
    }
    int optXBytes = optX * bytes;

    if (cmdKind == 1 &&
        htp_ops_loop_binary_try_parallel(pDst, pSrc0, src1, iter0, iter1, iter2, lp, opType, bytes,
                                         continuous, optX, optY, optZ)) {
        return 0;
    }

    for (int iter = 0; iter < loopNumber; ++iter) {
        int32_t it0 = srcIter0 ? loop_read_int32(srcIter0 + iter * sizeof(int32_t)) : iter;
        int32_t it1 = srcIter1 ? loop_read_int32(srcIter1 + iter * sizeof(int32_t)) : iter;

        int32_t outOff = (int32_t)it0 * lp->cmdSteps[0] + lp->cmdViewOffset[0];
        if (outOff < 0 || outOff >= lp->outputElementSize) continue;
        int32_t outByteOffset = (int32_t)outOff * bytes;
        uint8_t* dstBase = pDst + outByteOffset;

        int32_t in0Off = (int32_t)it1 * lp->cmdSteps[1] + lp->cmdViewOffset[1];
        if (in0Off < 0 || in0Off >= lp->input0Size) continue;
        const uint8_t* src0Base = pSrc0 + in0Off * bytes;

        if (cmdKind == 1 || cmdKind == 2) {
            uint8_t* pSrc1 = src1;
            uint8_t* pIter2 = iter2;
            const uint8_t* srcIter2 = pIter2;
            int32_t it2 = srcIter2 ? loop_read_int32(srcIter2 + iter * sizeof(int32_t)) : iter;
            int32_t in1Off = (int32_t)it2 * lp->cmdSteps[2] + lp->cmdViewOffset[2];
            if (in1Off < 0 || in1Off >= lp->input1Size) continue;
            const uint8_t* src1Base = pSrc1 + in1Off * bytes;
            if (cmdKind == 2) {
                htp_ops_loop_matmul_region(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off);
            } else {
                if (loopNumber == 1) {
                    HtpOpsBinaryRegion region;
                    region.src0Offset = 0;
                    region.src1Offset = 0;
                    region.dstOffset = 0;
                    for (int d = 0; d < 3; ++d) {
                        region.size[d] = lp->sizeXYZ[d];
                        region.src0Stride[d] = lp->src0StrideXYZ[d];
                        region.src1Stride[d] = lp->src1StrideXYZ[d];
                        region.dstStride[d] = lp->dstStrideXYZ[d];
                    }
                    htp_ops_binary_blit(dstBase, src0Base, src1Base, (uint8_t*)&region, 1, bytes, opType);
                    continue;
                }
                htp_ops_loop_binary_region(dstBase, src0Base, src1Base, lp, opType, bytes,
                                           continuous, optX, optY, optZ);
            }
        } else if (continuous) {
            for (int z = 0; z < optZ; ++z) {
                const uint8_t* src0Z = src0Base + z * lp->src0StrideXYZ[0];
                uint8_t* dstZ = dstBase + z * lp->dstStrideXYZ[0];
                for (int y = 0; y < optY; ++y) {
                    const uint8_t* src0Y = src0Z + y * lp->src0StrideXYZ[1];
                    uint8_t* dstY = dstZ + y * lp->dstStrideXYZ[1];

                    size_t rowBytes = (size_t)optXBytes;
                    if (rowBytes >= (size_t)__HVX_LENGTH__) {
                        const int vecBytes = __HVX_LENGTH__;
                        const size_t vecCount = rowBytes / (size_t)vecBytes;
                        const size_t tailBytes = rowBytes - vecCount * (size_t)vecBytes;
                        for (size_t iVec = 0; iVec < vecCount; ++iVec) {
                            const uint8_t *sVec = src0Y + iVec * (size_t)vecBytes;
                            uint8_t *dVec = dstY + iVec * (size_t)vecBytes;
                            HVX_Vector v = vmemu((const HVX_Vector *)sVec);
                            vmemu((HVX_Vector *)dVec) = v;
                        }
                        if (tailBytes > 0) {
                            memcpy(dstY + vecCount * (size_t)vecBytes, src0Y + vecCount * (size_t)vecBytes, tailBytes);
                        }
                    } else {
                        memcpy(dstY, src0Y, rowBytes);
                    }
                }
            }
        } else {
            for (int z = 0; z < lp->sizeXYZ[0]; ++z) {
                const uint8_t* src0Z = src0Base + z * lp->src0StrideXYZ[0];
                uint8_t* dstZ = dstBase + z * lp->dstStrideXYZ[0];
                for (int y = 0; y < lp->sizeXYZ[1]; ++y) {
                    const uint8_t* src0Y = src0Z + y * lp->src0StrideXYZ[1];
                    uint8_t* dstY = dstZ + y * lp->dstStrideXYZ[1];
                    for (int x = 0; x < lp->sizeXYZ[2]; ++x) {
                        const uint8_t* s0 = src0Y + x * lp->src0StrideXYZ[2];
                        uint8_t* d = dstY + x * lp->dstStrideXYZ[2];
                        memcpy(d, s0, bytes);
                    }
                }
            }
        }
    }

    return 0;
}

AEEResult htp_ops_batch_matmul(uint8_t* dst, uint8_t* src0, uint8_t* src1,
                               uint8_t* iter0, uint8_t* iter1, uint8_t* iter2,
                               int32_t bytes, uint8_t* param) {
    const HtpOpsLoopParam* lp = (HtpOpsLoopParam*)param;
    if (lp == nullptr || lp->loopNumber <= 0) {
        return 0;
    }
    if (bytes != 2) {
        return AEE_EUNSUPPORTED;
    }
    const uint8_t* srcIter0 = iter0;
    const uint8_t* srcIter1 = iter1;
    const uint8_t* srcIter2 = iter2;
    for (int iter = 0; iter < lp->loopNumber; ++iter) {
        int32_t it0 = srcIter0 ? loop_read_int32(srcIter0 + iter * sizeof(int32_t)) : iter;
        int32_t it1 = srcIter1 ? loop_read_int32(srcIter1 + iter * sizeof(int32_t)) : iter;
        int32_t it2 = srcIter2 ? loop_read_int32(srcIter2 + iter * sizeof(int32_t)) : iter;

        int32_t outOff = (int32_t)it0 * lp->cmdSteps[0] + lp->cmdViewOffset[0];
        if (outOff < 0 || outOff >= lp->outputElementSize) {
            continue;
        }
        int32_t in0Off = (int32_t)it1 * lp->cmdSteps[1] + lp->cmdViewOffset[1];
        if (in0Off < 0 || in0Off >= lp->input0Size) {
            continue;
        }
        int32_t in1Off = (int32_t)it2 * lp->cmdSteps[2] + lp->cmdViewOffset[2];
        if (in1Off < 0 || in1Off >= lp->input1Size) {
            continue;
        }
        uint8_t* dstBase = dst + (int64_t)outOff * bytes;
        const uint8_t* src0Base = src0 + (int64_t)in0Off * bytes;
        const uint8_t* src1Base = src1 + (int64_t)in1Off * bytes;
        htp_ops_loop_matmul_region(dstBase, src0Base, src1Base, lp, outOff, in0Off, in1Off);
    }
    return 0;
}

}  // extern "C"
