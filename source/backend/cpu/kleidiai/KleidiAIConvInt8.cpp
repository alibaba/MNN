//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvInt8.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <algorithm>
#include <limits>
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

// KleidiAI micro-kernel headers (int4 / int8 dynamic-quant matmul + packing).
// The symmetric per-channel int4 path is served by the asymmetric qsi8d32/qai4c32
// kernels below. The asym packer stores signed int4 (v-8), so the dequant is
// w = scale*(v-8) + zero; symmetric weights are exactly this with per-channel zero = 0.
// so no dedicated qai8dxp/qsi4cxp ukernels are needed here.
#include "kai_common.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
#include "kai_lhs_pack_f16pmrx4_f32_neon.h"
#include "kai_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#endif

#define QUANT_INFO_BYTES 4
namespace MNN {

// ===================================================================
// Static classification / gating (moved out of the former KleidiAI class).

KleidiAIConvInt8::KernelType KleidiAIConvInt8::getKernelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes) {
    // Only 4-bit dynamic-quant weights are accelerated today. The variant is picked from
    // symmetry, quant granularity (per-channel when blockSize == 0, else per-block) and the
    // activation precision (f32 when bytes == 4, f16 when bytes == 2). Anything else falls back.
    if (bits != 4 || (blockSize != 0 && blockSize % 32 != 0)) {
        return KernelType::KERNEL_TYPE_ERROR;
    }
    const bool perChannel = (blockSize == 0);
    if (bAsymmetric) {
        if (bytes == 4) {
            return perChannel ? KernelType::QI4_ASYM_PERCHANNEL_F32 : KernelType::QI4_ASYM_PERBLOCK_F32;
        }
        if (bytes == 2) {
            return perChannel ? KernelType::QI4_ASYM_PERCHANNEL_F16 : KernelType::QI4_ASYM_PERBLOCK_F16;
        }
        return KernelType::KERNEL_TYPE_ERROR;
    }
    // Symmetric: only per-channel f32 has a ukernel.
    if (perChannel && bytes == 4) {
        return KernelType::QI4_SYM_PERCHANNEL_F32;
    }
    return KernelType::KERNEL_TYPE_ERROR;
}

// Whether the running CPU provides the ukernels required by this KernelType.
static bool kaiKernelSupport(KleidiAIConvInt8::KernelType type) {
    auto cpu = MNNGetCPUInfo();
    bool hasKernel = cpu->sme2 || (cpu->dot && cpu->i8mm);
    switch (type) {
        case KleidiAIConvInt8::KernelType::QI4_SYM_PERCHANNEL_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F16:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F16:
            return hasKernel;
        default:
            return false;
    }
}

bool KleidiAIConvInt8::isSupported(KernelType type, const Convolution2DCommon* common) {
    if (type == KernelType::KERNEL_TYPE_ERROR) {
        return false;
    }
    if (common->group() != 1 || common->inputCount() <= 0 || common->outputCount() <= 0) {
        return false;
    }
    if (type == KernelType::QI4_ASYM_PERCHANNEL_F32 || type == KernelType::QI4_ASYM_PERCHANNEL_F16
        || type == KernelType::QI8_ASYM_PERCHANNEL || type == KernelType::QI4_SYM_PERCHANNEL_F32) {
        // Symmetric per-channel reuses the asymmetric qsi8d32/qai4c32 kernels, which require
        // the K dimension to be a multiple of 32.
        if (common->inputCount() % 32 != 0) {
            return false;
        }
    }
    if (common->kernelX() == 1 && common->kernelY() == 1
        && common->padX() == 0 && common->padY() == 0
        && common->strideX() == 1 && common->strideY() == 1
        && common->dilateX() == 1 && common->dilateY() == 1) {
        return kaiKernelSupport(type);
    }
    return false;
}

size_t KleidiAIConvInt8::getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep) {
    return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep);
}

// ===================================================================
// Per-instance kernel parameter resolution and ukernel dispatch.

// ===================================================================
// Uniform-signature adapters over the concrete KleidiAI micro-kernels.
// Each adapter matches one KleidiAIConvInt8::Ukernel slot; `bl` is ignored by the channel-quant
// (qsi4cx / qai8dx) kernels that do not take it. All are bound once in configKernel().
namespace {

// The rhs/lhs "size" and "offset" getters are pure forwarders that differ only by the concrete
// kai function and whether the trailing granularity arg is sr (channel-quant) or bl (block-quant).
// Generate them from a single pattern to avoid a wall of near-identical one-liners.
//   DEFINE_RHS_INFO      : rhs size/offset, shape (idx, k, nr, kr, <sr|bl>).
//   DEFINE_LHS_INFO_CHNL : lhs size/offset for channel-quant kernels that take no bl.
//   DEFINE_LHS_INFO_BLK  : lhs size/offset for block-quant kernels that take bl (3rd arg).
#define DEFINE_RHS_INFO(NAME, KAIFN, LAST) \
    size_t NAME(size_t idx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) { \
        (void)sr; (void)bl; \
        return KAIFN(idx, k, nr, kr, LAST); \
    }
#define DEFINE_LHS_INFO_CHNL(NAME, KAIFN) \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        (void)bl; \
        return KAIFN(idx, k, mr, kr, sr); \
    }
#define DEFINE_LHS_INFO_BLK(NAME, KAIFN) \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        return KAIFN(idx, k, bl, mr, kr, sr); \
    }

// ---- rhs packed size ----
DEFINE_RHS_INFO(rhsSizeAsymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsSizeAsymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,      bl)

// ---- rhs packed offset ----
DEFINE_RHS_INFO(rhsOffAsymSme2,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsOffAsymNeon,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,    bl)

// ---- rhs pack ----
void rhsPackAsymSme2(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                     const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, nr, kr, sr, bl,
        (const uint8_t*)rhs, zeroPoint, bias, scale, rhsPacked, 0, &params);
}
void rhsPackAsymNeon(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                     const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, nr, kr, sr, bl,
        (const uint8_t*)rhs, zeroPoint, bias, scale, rhsPacked, 0, &params);
}

// ---- lhs quanted packed size ----
DEFINE_LHS_INFO_BLK(lhsSizeAsymF32,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF16,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon)
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
size_t lhsSizeDirectF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    if (mr == 1) {
        return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, 1, kr, sr);
    }
    return kai_get_lhs_packed_size_lhs_pack_f16pmrx4_f32_neon(m, k, bl, mr, kr, sr);
}
#endif

// ---- lhs quanted packed offset ----
DEFINE_LHS_INFO_BLK(lhsOffAsymF32,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffAsymF16,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon)
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
DEFINE_LHS_INFO_BLK(lhsOffDirectF32,  kai_get_lhs_packed_offset_lhs_pack_f16pmrx4_f32_neon)
#endif

// ---- lhs quant + pack ----
void lhsPackAsymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, kr, sr, 0, (const __fp16*)lhs, k * sizeof(__fp16), out);
}
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
void lhsPackDirectF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    // A GEMM packing worker can own a one-row tail. Keep the layout selected
    // for the full operation (mr), rather than treating that tail as GEMV.
    if (mr == 1) {
        lhsPackAsymF32(m, k, bl, 1, kr, sr, lhs, out);
        return;
    }
    kai_run_lhs_pack_f16pmrx4_f32_neon(m, k, bl, mr, kr, sr, 0, lhs, k * sizeof(float), out);
}
#endif

// ---- matmul (GEMV when m == 1, GEMM otherwise) ----
void matmulAsymF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF32Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF16Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF16Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
void matmulDirectF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                         size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(
            m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
        return;
    }
    kai_run_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
        m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
}
#endif

#undef DEFINE_RHS_INFO
#undef DEFINE_LHS_INFO_CHNL
#undef DEFINE_LHS_INFO_BLK

} // namespace

// ===================================================================
// Per-instance kernel parameter resolution and ukernel dispatch.

void KleidiAIConvInt8::configKernel() {
    auto cpu = MNNGetCPUInfo();
    mSme2 = cpu->sme2;
    mDot  = cpu->dot;
    mI8mm = cpu->i8mm;
    mHybrid = false;
    mChnlQuant = (mKernelType == KernelType::QI4_SYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F16);

    // Slot fillers. Each binds one (KernelParam, Ukernel) pair to a concrete kernel family so that
    // both the primary (SME) and, when hybrid, the secondary (NEON) slot are configured identically.
    auto fillSmeF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF32;
        u.lhsPackedOffset = lhsOffAsymF32;
        u.runLhsQuantPack = lhsPackAsymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulAsymF32Sme2;
    };
    auto fillNeonF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF32;
        u.lhsPackedOffset = lhsOffAsymF32;
        u.runLhsQuantPack = lhsPackAsymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = 8;
        p.mKaiNStep     = 4;
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = 4;
        p.mKaiNr        = 4;
        p.mKaiKr        = 16;
        p.mKaiSr        = 2;
        u.rhsPackedSize   = rhsSizeAsymNeon;
        u.rhsPackedOffset = rhsOffAsymNeon;
        u.runRhsPack      = rhsPackAsymNeon;
        u.matmul          = matmulAsymF32Neon;
    };
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
    auto fillSmeDirectF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeDirectF32;
        u.lhsPackedOffset = lhsOffDirectF32;
        u.runLhsQuantPack = lhsPackDirectF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulDirectF32Sme2;
    };
#endif
    auto fillSmeF16 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF16;
        u.lhsPackedOffset = lhsOffAsymF16;
        u.runLhsQuantPack = lhsPackAsymF16;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulAsymF16Sme2;
    };
    auto fillNeonF16 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF16;
        u.lhsPackedOffset = lhsOffAsymF16;
        u.runLhsQuantPack = lhsPackAsymF16;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = 8;
        p.mKaiNStep     = 4;
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = 4;
        p.mKaiNr        = 4;
        p.mKaiKr        = 16;
        p.mKaiSr        = 2;
        u.rhsPackedSize   = rhsSizeAsymNeon;
        u.rhsPackedOffset = rhsOffAsymNeon;
        u.runRhsPack      = rhsPackAsymNeon;
        u.matmul          = matmulAsymF16Neon;
    };

    switch (mKernelType) {
        // Symmetric per-channel int4 is served by the asymmetric qsi8d32/qai4c32 kernels:
        // the asym packer stores signed int4 (v-8), so w = scale*(v-8) + zero; symmetric is
        // exactly this with per-channel zero = 0. The symmetric scale/zero are synthesized in
        // the constructor.
        case KernelType::QI4_SYM_PERCHANNEL_F32:
        case KernelType::QI4_ASYM_PERCHANNEL_F32:
            if (mSme2) {
                fillSmeF32(mParam, mUkernel);
                if (mDot && mI8mm) {
                    // Also configure the NEON slot so the two can run concurrently (SME + NEON).
                    fillNeonF32(mParamNeon, mUkernelNeon);
                    mHybrid = true;
                }
            } else if (mDot && mI8mm) {
                fillNeonF32(mParam, mUkernel);
            }
            break;
        case KernelType::QI4_ASYM_PERBLOCK_F32:
            if (mSme2) {
#ifdef MNN_KLEIDIAI_F16_PACKED_INT4
                // The direct kernel converts FP32 activations to packed FP16 and accumulates in
                // FP32 ZA, avoiding dynamic INT8 quantization and per-block requantization.
                fillSmeDirectF32(mParam, mUkernel);
#else
                fillSmeF32(mParam, mUkernel);
#endif
            } else if (mDot && mI8mm) {
                fillNeonF32(mParam, mUkernel);
            }
            break;
        case KernelType::QI4_ASYM_PERCHANNEL_F16:
        case KernelType::QI4_ASYM_PERBLOCK_F16:
            if (mSme2) {
                fillSmeF16(mParam, mUkernel);
                if (mDot && mI8mm) {
                    fillNeonF16(mParamNeon, mUkernelNeon);
                    mHybrid = true;
                }
            } else if (mDot && mI8mm) {
                fillNeonF16(mParam, mUkernel);
            }
            break;
        default:
            break;
    }
}

size_t KleidiAIConvInt8::getRhsPackedSize(const Ukernel& u, const KernelParam& p, size_t n, size_t k, size_t bl) const {
    return u.rhsPackedSize(n, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl);
}

size_t KleidiAIConvInt8::getRhsPackedOffset(const Ukernel& u, const KernelParam& p, size_t nIdx, size_t k, size_t bl) const {
    if (nIdx == 0) {
        return 0;
    }
    return u.rhsPackedOffset(nIdx, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl);
}

void KleidiAIConvInt8::runRhsPack(const Ukernel& u, const KernelParam& p, size_t numGroups, size_t n, size_t k, size_t bl,
                                  const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                                  void* rhsPacked) const {
    u.runRhsPack(numGroups, n, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl,
                 rhs, scale, zeroPoint, bias, rhsPacked);
}

size_t KleidiAIConvInt8::getLhsQuantedPackedSize(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl) const {
    return u.lhsPackedSize(m, k, mChnlQuant ? k : bl, getMr(p, m), getKr(p), getSr(p));
}

size_t KleidiAIConvInt8::getLhsQuantedPackedOffset(const Ukernel& u, const KernelParam& p, size_t m, size_t mIdx, size_t k, size_t bl) const {
    if (mIdx == 0) {
        return 0;
    }
    return u.lhsPackedOffset(mIdx, k, mChnlQuant ? k : bl, getMr(p, m), getKr(p), getSr(p));
}

void KleidiAIConvInt8::runLhsQuantPack(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl, size_t mr,
                                       const void* lhs, void* lhsQuantedPacked) const {
    u.runLhsQuantPack(m, k, mChnlQuant ? k : bl, mr, getKr(p), getSr(p), lhs, lhsQuantedPacked);
}

void KleidiAIConvInt8::runMatmul(const Ukernel& u, const KernelParam& p, size_t m, size_t n, size_t k, size_t bl,
                                 const void* lhsPacked, const void* rhsPacked, void* dst,
                                 size_t dstStrideRow, size_t dstStrideCol,
                                 const float scalarMax, const float scalarMin) const {
    (void)p;
    u.matmul(m, n, k, mChnlQuant ? k : bl, lhsPacked, rhsPacked, dst,
             dstStrideRow, dstStrideCol, scalarMin, scalarMax);
}

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant,
    KernelType kernelType, int32_t blockNum)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), mKernelType(kernelType), mBlockNum(blockNum) {
    // Publish validity only after all persistent resources have been packed.
    mValid = false;
    // Resolve CPU features and kernel packing parameters for this KernelType.
    configKernel();

    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();
    if (ic <= 0 || oc <= 0 || mBlockNum <= 0 || ic % mBlockNum != 0 || (ic / mBlockNum) % 32 != 0) {
        return;
    }

    // backend info
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int pack = core->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int scaleSize = ocUp4 * mBlockNum;

    // kleidia info
    bool bFP16 = core->bytes == 2 ? true : false;
    bool bAsym = quanCommon->asymmetric;
    size_t blkSize = mBlockNum == 1 ? 0 : ic / mBlockNum;

    AutoStorage<int8_t> reorderedQuantInfo;
    reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
    if (reorderedQuantInfo.get() == nullptr) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    ::memset(reorderedQuantInfo.get(), 0, reorderedQuantInfo.size());

    // Prepare bias (needed by every path) and, for the symmetric path, scale/zero.
    // The asymmetric path fills scale/zero below in the ukernel-specific linear layout,
    // so we intentionally skip them here to avoid computing them twice with different layouts.
    {
        int outputCount = convOp->common()->outputCount();
        auto quanInfoPtr = quanCommon->alpha.get();
        auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
        auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
        if (!quanCommon->asymmetric) {
            // Symmetric weights routed through the asymmetric ukernel: the packer stores signed
            // int4 (v-8), so w = scale*(v-8) + zero. Symmetric is exactly scale*(v-8), i.e. zero = 0.
            for (int i = 0; i < blockNum; ++i) {
                auto dstScale = scalePtr + i * ocUp4;
                auto dstZero  = zeroPtr + i * ocUp4;
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstScale[j] = quanInfoPtr[scaleIndex];
                    dstZero[j] = 0.f;
                }
            }
        }
        if (convOp->bias() != nullptr && convOp->bias()->size() != 0) {
            if (convOp->bias()->size() != static_cast<size_t>(oc)) {
                return;
            }
            ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
        }
    }

    int n = oc;
    int k = ic;
    int packedWeightSize = getRhsPackedSize(n, k, blkSize);

    //Alloc packed weight tensor.
    mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);

    if (!success) {
        MNN_ERROR("Out of static memory!\n");
        return;
    }

    size_t paraNum = scaleSize;
    float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
    float *biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
    //Reload some parameters to fit ukernels' layout.
    auto quanInfoPtr = quanCommon->alpha.get();
    auto alphaSize = quanCommon->alpha.size();
    if(bAsym) {
        for(int i = 0; i < paraNum; i++) {
            if(i*2 >= alphaSize){
                zeroPtr[i] = 0;
                scalePtr[i] = 0;
            }
            else{
                zeroPtr[i] = quanInfoPtr[i * 2];
                scalePtr[i] = quanInfoPtr[i * 2 + 1];
            }
        }
    } else {
        if(blkSize != 0) {
            memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
        }
    }

    //Run rhs pack.
    auto weightPackedData = mWeightInt8->host<uint8_t>();
    runRhsPack(1, n, k, blkSize,
               (uint8_t*)quanCommon->weight.get(),
               (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
               weightPackedData);

    if (mHybrid) {
        // Pack a second copy of the weights in the NEON slot layout so the NEON kernels can run
        // concurrently with the SME kernel on the remaining threads. Same scale/zero/bias, but a
        // different packed layout, hence a separate static buffer (~2x weight memory).
        int packedWeightSizeNeon = getRhsPackedSize(mUkernelNeon, mParamNeon, n, k, blkSize);
        mWeightInt8Neon.reset(Tensor::createDevice<uint8_t>({packedWeightSizeNeon}));
        bool successNeon = backend->onAcquireBuffer(mWeightInt8Neon.get(), Backend::STATIC);
        if (!successNeon) {
            MNN_ERROR("Out of static memory!\n");
            return;
        }
        runRhsPack(mUkernelNeon, mParamNeon, 1, n, k, blkSize,
                   (uint8_t*)quanCommon->weight.get(),
                   (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                   mWeightInt8Neon->host<uint8_t>());
    }
    mValid = true;
    return;
}


KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend),
    mWeightInt8(exe.mWeightInt8),
    mWeightInt8Neon(exe.mWeightInt8Neon),
    mKernelType(exe.mKernelType), mBlockNum(exe.mBlockNum) {
    configKernel();
}

KleidiAIConvInt8::~KleidiAIConvInt8() {
    // Do nothing
}

bool KleidiAIConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!valid()) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvInt8(bn, op, *this);
    if (!exe->valid()) {
        delete exe;
        return false;
    }
    *dst = exe;
    return true;
}

// need
ErrorCode KleidiAIConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core =static_cast<CPUBackend*>(backend())->functions();
    auto b = backend();

    const size_t m = inputs[0]->batch() * inputs[0]->width() * inputs[0]->height(); //lhs vector number.
    const size_t n = outputs[0]->channel(); //rhs vector number.
    const size_t k = inputs[0]->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    configSchedule(m, n, k, blkSize, static_cast<CPUBackend*>(backend())->threadNumber());

    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mInputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{input->batch(), input->height(), input->width(), input->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mOutputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{output->batch(), output->height(), output->width(), output->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    int packedSize = getLhsQuantedPackedSize(m, k, blkSize);
    int elementSize = core->bytes;

    //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
    //Part0: Lhs_packed.
    //Part1: Lhs/Dst before transfer.
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        MNN_ERROR("Out of dynamic memory!\n");
        return OUT_OF_MEMORY;
    }

    if (mHybrid) {
        // The NEON slot packs lhs with a different mr, so it needs its own packed buffer.
        int packedSizeNeon = getLhsQuantedPackedSize(mUkernelNeon, mParamNeon, m, k, blkSize);
        mTempIm2ColBufferNeon.reset(Tensor::createDevice<int8_t>({packedSizeNeon}));
        bool successNeon = backend()->onAcquireBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
        if (!successNeon) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (mHybrid) {
        backend()->onReleaseBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
    }

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

namespace {

// File-local scheduling policy: balance SME/NEON relative work costs rather
// than predict device latency. Independent of tensors and kernel state.
class KleidiAIHybridScheduler {
public:
    struct Plan {
        size_t smeColumns = 0;
        size_t neonColumns = 0;
        size_t neonColumnsPerThread = 0;
        int neonThreads = 0;

        bool enabled() const { return smeColumns > 0 && neonColumns > 0; }
    };

    // One SME worker processes the leading columns; the remaining workers use NEON.
    // blockSize == 0 denotes per-channel quantization.
    static Plan selectPlan(bool isF16, size_t m, size_t n, size_t k, size_t blockSize,
                           size_t smeNStep, size_t neonNStep, int threadCount) {
        Plan plan;
        if (m == 0 || k == 0 || threadCount <= 1 || smeNStep == 0 || neonNStep == 0 || n <= smeNStep
            || smeNStep % neonNStep != 0) {
            return plan;
        }
        // configKernel() uses a single SME slot for FP32 per-block,
        // so only FP32 per-channel and the two FP16 profiles need hybrid fits.
        if (!isF16 && blockSize != 0) {
            return plan;
        }
        const int neonThreads = threadCount - 1;
        const auto& sme = smeModel(isF16, blockSize != 0);
        const auto& neon = neonModel(isF16, blockSize != 0);
        double bestBottleneckCost = std::numeric_limits<double>::max();
        for (size_t candidate = smeNStep; candidate < n; candidate += smeNStep) {
            const size_t neonColumns = n - candidate;
            // Match the executor's aligned partition and price its busiest worker,
            // not the unaligned average. Keep the first candidate on ties.
            const size_t average = (neonColumns + neonThreads - 1) / neonThreads;
            const size_t perThread = ((average + neonNStep - 1) / neonNStep) * neonNStep;
            const size_t busiest = std::min(perThread, neonColumns);
            // Concurrent engines are limited by the more expensive share.
            const double bottleneckCost = std::max(relativeCost(sme, true, m, candidate, k, blockSize),
                                                   relativeCost(neon, false, m, busiest, k, blockSize));
            if (bottleneckCost < bestBottleneckCost) {
                bestBottleneckCost = bottleneckCost;
                plan.smeColumns = candidate;
                plan.neonColumnsPerThread = perThread;
            }
        }
        plan.neonColumns = n - plan.smeColumns;
        plan.neonThreads = static_cast<int>((plan.neonColumns + plan.neonColumnsPerThread - 1)
                                            / plan.neonColumnsPerThread);
        return plan;
    }

private:
    // Decimal feature scales preserve the existing three-significant-digit fit
    // without another rounding step. Both engines use a shared cost scale.
    static constexpr double kKScale = 1000.0;
    static constexpr double kMacScale = 1000000.0;

    struct GemvCoefficients {
        double fixedCost;
        double panel;
        double panelKiloK; // Per panel, per 1000 K elements.
        double panelBlock;
    };

    struct GemmCoefficients {
        double fixedCost;
        double tile;
        double computeScaled; // SME: per tile per 1000 K elements; NEON: per million MACs.
        double tileBlock;
        double narrowPanel;
    };

    struct Model {
        size_t mTile;
        size_t nPanel;
        GemvCoefficients gemv;
        GemmCoefficients gemm;
    };

    // Empirical relative matmul costs, kept to three significant digits.
    // For equal work, a lower score represents higher
    // efficiency; these scores are not a device-latency API.
    // Only comparisons matter: a common positive scale factor applied to BOTH
    // complete profiles preserves the split; normalizing each engine independently
    // would destroy the relative-efficiency information.
    // Relative efficiency can also vary by CPU. These profiles are empirical, not
    // universal hardware constants. Packing and thread-pool overhead are excluded.
    // Profiles match the hybrid-enabled branches in configKernel().
    // GEMV: fixedCost + panels * (panel + panelKiloK*(K/1000) + panelBlock*K/blockSize).
    // GEMM: fixedCost + tile*T + computeScaled*work + tileBlock*T*K/blockSize
    //       + narrowPanel*singleNarrow, with T = ceil(M/MR)*ceil(N/NR).
    // Normalized work: SME T*(Kpad/1000), NEON (M*N*K)/1000000.
    // All terms contribute to the same shared cost scale.
    static const Model& smeModel(bool isF16, bool perBlock) {
        static const Model fp32Channel = {16, 64, {0.0, 0.0472, 0.331, 0.0},
                                                 {0.272, 0.259, 0.521, 0.0, 0.559}};
        static const Model fp16Channel = {16, 64, {0.0, 0.0732, 0.254, 0.0},
                                                 {0.380, 0.291, 0.484, 0.0, 0.438}};
        static const Model fp16Block = {16, 64, {0.0839, 0.00565, 0.275, 0.00773},
                                               {0.469, 0.0941, 0.368, 0.200, 0.0}};
        return isF16 ? (perBlock ? fp16Block : fp16Channel) : fp32Channel;
    }

    static const Model& neonModel(bool isF16, bool perBlock) {
        static const Model fp32Channel = {8, 4, {0.0381, -0.00303, 0.0647, 0.0},
                                               {0.0228, 8.07e-4, 5.52, 0.0, 0.0}};
        static const Model fp16Channel = {8, 4, {0.0369, -0.00296, 0.0648, 0.0},
                                               {0.0174, 0.00133, 5.54, 0.0, 0.0}};
        static const Model fp16Block = {8, 4, {0.0301, 3.04e-4, 0.0438, -1.02e-4},
                                             {-0.00574, 0.00306, 4.99, 7.97e-4, 0.0}};
        return isF16 ? (perBlock ? fp16Block : fp16Channel) : fp32Channel;
    }

    static double relativeCost(const Model& model, bool isSme, size_t m, size_t n, size_t k, size_t blockSize) {
        if (m == 0 || n == 0) {
            return 0.0;
        }
        const size_t panels = (n + model.nPanel - 1) / model.nPanel;
        if (m == 1) {
            const auto& c = model.gemv;
            const double kiloK = static_cast<double>(k) / kKScale;
            double panelCost = c.panel + c.panelKiloK * kiloK;
            if (blockSize != 0) {
                panelCost += c.panelBlock * static_cast<double>(k) / static_cast<double>(blockSize);
            }
            return c.fixedCost + static_cast<double>(panels) * panelCost;
        }

        const size_t mTiles = (m + model.mTile - 1) / model.mTile;
        const double tiles = static_cast<double>(mTiles) * static_cast<double>(panels);
        const auto& c = model.gemm;
        double cost = c.fixedCost + c.tile * tiles;
        if (isSme) {
            // MOPA work is T*Kpad, Kpad = roundup(K, 32).
            const double kiloKPadded = static_cast<double>(((k + 31) / 32) * 32) / kKScale;
            cost += c.computeScaled * tiles * kiloKPadded;
        } else {
            const double megaMac = static_cast<double>(m) * static_cast<double>(n)
                                   * static_cast<double>(k) / kMacScale;
            cost += c.computeScaled * megaMac;
        }
        if (blockSize != 0) {
            cost += c.tileBlock * tiles * static_cast<double>(k) / static_cast<double>(blockSize);
        }
        // Zero for SME N-steps >= 64, but smaller runtime vector lengths can
        // produce candidates below 64. Retain their single-panel correction.
        if (isSme && n < model.nPanel) {
            const double singleNarrow = static_cast<double>(mTiles) * static_cast<double>(model.nPanel - n)
                                        / static_cast<double>(model.nPanel);
            cost += c.narrowPanel * singleNarrow;
        }
        return cost;
    }
};

} // namespace

void KleidiAIConvInt8::configSchedule(size_t m, size_t n, size_t k, size_t blockSize, int threadCount) {
    mSmeColumns = 0;
    mNeonColumnsPerThread = 0;
    mHybridThreadCount = 0;
    if (!mHybrid) {
        return;
    }
    const bool isF16 = mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F16
                       || mKernelType == KernelType::QI4_ASYM_PERBLOCK_F16;
    const auto plan = KleidiAIHybridScheduler::selectPlan(isF16, m, n, k, blockSize,
                                                        getNStep(mParam), getNStep(mParamNeon), threadCount);
    if (plan.enabled()) {
        mSmeColumns = plan.smeColumns;
        mNeonColumnsPerThread = plan.neonColumnsPerThread;
        mHybridThreadCount = 1 + plan.neonThreads;
    }
}

ErrorCode KleidiAIConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

    // Initialize for convert
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    auto b = backend();
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    const size_t m = input->batch() * input->width() * input->height(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    size_t elementSize = core->bytes;

    auto lhs = input->host<uint8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    if(inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // Convert input to NHWC format.
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(input, mInputConvertBuffer.get(), core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        lhs = mInputConvertBuffer->host<uint8_t>();
    }

    // Dynamic-quant + pack lhs into `out` using the given kernel slot. Splits the M dimension over
    // the thread pool (single call for the GEMV m == 1 case).
    auto packLhs = [&](const Ukernel& u, const KernelParam& p, int8_t* out) {
        if (m == 1) {
            runLhsQuantPack(u, p, 1, k, blkSize, getMr(p, m), lhs, out);
            return;
        }
        size_t mr = getMr(p, m);
        int vecPer = getVecNumPerThread(m, threadNum, mr);
        int need = m % vecPer == 0 ? m / vecPer : (m / vecPer + 1);
        size_t srcStride = (size_t)vecPer * k * elementSize;
        MNN_CONCURRENCY_BEGIN(tId, need) {
            int t = (int)tId;
            auto threadSrc = lhs + (size_t)t * srcStride;
            auto threadDst = out + getLhsQuantedPackedOffset(u, p, m, (size_t)t * vecPer, k, blkSize);
            int vecNum = (t == need - 1) ? (m - vecPer * t) : vecPer; //Last threadN may less than vecPer.
            runLhsQuantPack(u, p, vecNum, k, blkSize, mr, threadSrc, threadDst);
        }
        MNN_CONCURRENCY_END();
    };

    //Run matmul.
    auto dst = output->host<uint8_t>();
    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        //store matmul result to convert buffer.
        dst = mOutputConvertBuffer->host<uint8_t>();
    }
    auto postPtr = getPostParameters();

    // The shape-dependent split was resolved during onResize().
    if (mSmeColumns == 0) {
        // Single-slot path: SME-only on one thread (SME prefers a single thread for better
        // performance/power ratio) or NEON-only spread across all threads.
        auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
        auto rhsPacked = mWeightInt8->host<uint8_t>();
        packLhs(mUkernel, mParam, lhsPacked);
        int matThreadNum = bSupportSme2() ? 1 : threadNum;
        int vecPerThread = getVecNumPerThread(n, matThreadNum, getNStep());
        int threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            int t = (int)tId;
            auto threadRhsPacked = rhsPacked + getRhsPackedOffset(t * vecPerThread, k, blkSize);
            auto threadDst = dst + getDstOffset(0, t * vecPerThread, n, elementSize);
            int vecNum = (t == threadNeed - 1) ? (n - vecPerThread * t) : vecPerThread; //Last threadN may less than vecPerThread.
            runMatmul(m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst,
                      n * elementSize, elementSize, postPtr[3], postPtr[2]);
        }
        MNN_CONCURRENCY_END();
    } else {
        // Hybrid path: pack lhs once per slot (different mr => different packed layout), then run the
        // SME kernel on thread 0 over columns [0, nSme) concurrently with NEON kernels on the
        // remaining threads over columns [nSme, n).
        const size_t nSme = mSmeColumns;
        const size_t nNeon = n - nSme;
        auto lhsPackedSme  = mTempIm2ColBuffer->host<int8_t>();
        auto lhsPackedNeon = mTempIm2ColBufferNeon->host<int8_t>();
        packLhs(mUkernel, mParam, lhsPackedSme);
        packLhs(mUkernelNeon, mParamNeon, lhsPackedNeon);
        auto rhsPackedSme  = mWeightInt8->host<uint8_t>();
        auto rhsPackedNeon = mWeightInt8Neon->host<uint8_t>();
        const size_t vecPerNeon = mNeonColumnsPerThread;
        MNN_CONCURRENCY_BEGIN(tId, mHybridThreadCount) {
            int t = (int)tId;
            if (t == 0) {
                // SME slot: columns [0, nSme).
                runMatmul(mUkernel, mParam, m, nSme, k, blkSize, lhsPackedSme, rhsPackedSme,
                          dst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
            } else {
                // NEON slot: columns [nSme, n) split among neonThreads.
                const size_t localStart = static_cast<size_t>(t - 1) * vecPerNeon;
                if (localStart < nNeon) {
                    const size_t vecNum = std::min(vecPerNeon, nNeon - localStart);
                    size_t globalStart = nSme + localStart;
                    auto threadRhsPacked = rhsPackedNeon + getRhsPackedOffset(mUkernelNeon, mParamNeon, globalStart, k, blkSize);
                    auto threadDst = dst + getDstOffset(0, globalStart, n, elementSize);
                    runMatmul(mUkernelNeon, mParamNeon, m, vecNum, k, blkSize, lhsPackedNeon, threadRhsPacked,
                              threadDst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // Convert output from NHWC format to original format.
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(mOutputConvertBuffer.get(), output, core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
#endif //MNN_KLEIDIAI_ENABLED
