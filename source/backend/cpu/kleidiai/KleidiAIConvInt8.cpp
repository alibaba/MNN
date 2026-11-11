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
#include "kai_matmul.h"
#include "kai_matmul_pack_lhs.h"
#include "kai_lhs_pack_f16pmrx2_f32_neon.h"

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
    // QSI4 has no affine zero term; never route asymmetric weights through it.
    if (!perChannel && (bytes == 4 || bytes == 2)) {
        return bytes == 4 ? KernelType::QI4_SYM_PERBLOCK_F32 : KernelType::QI4_SYM_PERBLOCK_F16;
    }
    // Keep the existing symmetric per-channel route unchanged.
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
        case KleidiAIConvInt8::KernelType::QI4_SYM_PERBLOCK_F32:
        case KleidiAIConvInt8::KernelType::QI4_SYM_PERBLOCK_F16:
            return cpu->sme2;
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

// ---- lhs quanted packed offset ----
DEFINE_LHS_INFO_BLK(lhsOffAsymF32,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffAsymF16,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon)

// ---- lhs quant + pack ----
void lhsPackAsymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, kr, sr, 0, (const __fp16*)lhs, k * sizeof(__fp16), out);
}

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

// QSI4C32P stores each NR panel as all K/4 groups of four offset-binary
// nibbles, followed by [K/BL, NR] FP16 scales. Unlike QAI4 there is no bias
// or zero-point field. Pack directly from MNN's high-nibble-first q+8 codes;
// this also avoids the official intermediate RHS packer's BL <= 256 limit.
size_t rhsSizeQsi4(size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    return kai_roundup(n, nr) * (k / 2 + (k / bl) * sizeof(__fp16));
}
size_t rhsOffQsi4(size_t nIdx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    MNN_ASSERT(nIdx % nr == 0);
    return nIdx * (k / 2 + (k / bl) * sizeof(__fp16));
}
void rhsPackQsi4(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                 const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* out) {
    MNN_ASSERT(numGroups == 1 && bl != 0 && bl % 32 == 0 && k % bl == 0);
    const auto src = static_cast<const uint8_t*>(rhs);
    const auto scales = static_cast<const float*>(scale);
    const size_t blocks = k / bl;
    const size_t panelBytes = nr * (k / 2 + blocks * sizeof(__fp16));
    ::memset(out, 0, rhsSizeQsi4(n, k, nr, kr, sr, bl));
    for (size_t start = 0; start < n; start += nr) {
        auto panel = static_cast<uint8_t*>(out) + (start / nr) * panelBytes;
        auto panelScales = reinterpret_cast<__fp16*>(panel + nr * k / 2);
        for (size_t c = 0; c < std::min(nr, n - start); ++c) {
            const auto row = src + (start + c) * k / 2;
            for (size_t group = 0; group < k / 4; ++group) {
                for (size_t pair = 0; pair < 2; ++pair) {
                    const uint8_t v = row[2 * group + pair];
                    panel[2 * (group * nr + c) + pair] = (v >> 4) | (v << 4);
                }
            }
            for (size_t block = 0; block < blocks; ++block) {
                panelScales[block * nr + c] = static_cast<__fp16>(scales[(start + c) * blocks + block]);
            }
        }
    }
}
DEFINE_LHS_INFO_BLK(lhsSizeF16Mopa, kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffF16Mopa, kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon)
void lhsPackF16MopaF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                    const void* lhs, void* out) {
    // Even M=1 uses the MOPA MR layout, including one-row parallel tails.
    kai_run_lhs_pack_f16pmrx2_f32_neon(m, k, bl, mr, kr, sr, 0, lhs, k * sizeof(float), out);
}
void lhsPackF16MopaF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                    const void* lhs, void* out) {
    const auto api = kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2();
    kai_matmul_pack_lhs_uker_config config = {};
    kai_matmul_pack_lhs_uker_args args = {};
    args.shape.m = m;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = k * sizeof(__fp16);
    args.operand.lhs_packed.ptr = out;
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {m, k};
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &shape);
    api.run(&config, &args);
}
void matmulQsi4F16Mopa(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                      size_t sr, size_t sc, float mn, float mx) {
    MNN_ASSERT(sc == sizeof(float));
    const auto api = kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_f16p_4vsx16vs_sme2_mopa();
    kai_matmul_uker_config config = {};
    config.format.bl = bl;
    kai_matmul_uker_args args = {};
    args.shape.m = m;
    args.shape.n = n;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.rhs.ptr = rhs;
    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = sr;
    // Default LUT decodes q+8. No activation clamp here: bias must be added
    // first. The executor performs bias + clamp and only then converts to F16.
    api.run(&config, &args);
}
// Each NR panel contains K/32 blocks of [16*NR code bytes, NR half offsets,
// NR half scales]. Unlike QSI4, metadata separates every 32 depths, and its
// channel order is 0, NR/2, 1, NR/2+1, ... . Original BL stays separate from
// the compute BL=32: subdivide blocks by repeating metadata, not requantizing.
size_t rhsSizeQai4(size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    return kai_roundup(n, nr) * (k / 32) * 20;
}
size_t rhsOffQai4(size_t nIdx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    MNN_ASSERT(nIdx % nr == 0);
    return nIdx * (k / 32) * 20;
}
void rhsPackQai4(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                 const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* out) {
    MNN_ASSERT(numGroups == 1 && n > 0 && bl != 0 && bl % 32 == 0 && k % bl == 0);
    const auto src = static_cast<const uint8_t*>(rhs);
    const auto scales = static_cast<const float*>(scale);
    const auto zeros = static_cast<const float*>(zeroPoint);
    const size_t blocks = k / bl;
    const size_t panelBytes = nr * (k / 32) * 20;
    for (size_t start = 0; start < n; start += nr) {
        auto panel = static_cast<uint8_t*>(out) + (start / nr) * panelBytes;
        for (size_t block = 0; block < k / 32; ++block) {
            auto data = panel + block * 20 * nr;
            auto offsets = reinterpret_cast<__fp16*>(data + 16 * nr);
            auto packedScales = reinterpret_cast<__fp16*>(data + 18 * nr);
            for (size_t c = 0; c < nr; ++c) {
                // Match the official packer's last-channel replication for N tails.
                const size_t channel = std::min(start + c, n - 1);
                const auto row = src + channel * k / 2 + block * 16;
                for (size_t group = 0; group < 8; ++group) {
                    for (size_t pair = 0; pair < 2; ++pair) {
                        const uint8_t v = row[2 * group + pair];
                        data[2 * (group * nr + c) + pair] = (v >> 4) | (v << 4);
                    }
                }
                const size_t idx = channel * blocks + (block * 32) / bl;
                const size_t meta = 2 * (c % (nr / 2)) + c / (nr / 2);
                // MNN w=(u-8)*s+z; official kernel dequantizes FMA16(u,s,z-8*s).
                offsets[meta] = static_cast<__fp16>(zeros[idx] - 8.0f * scales[idx]);
                packedScales[meta] = static_cast<__fp16>(scales[idx]);
            }
        }
    }
}
void matmulQai4F16Mopa(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                      size_t sr, size_t sc, float mn, float mx) {
    MNN_ASSERT(sc == sizeof(float));
    const auto api = kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa();
    kai_matmul_uker_config config = {};
    config.format.bl = 32;
    kai_matmul_uker_args args = {};
    args.shape.m = m;
    args.shape.n = n;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.rhs.ptr = rhs;
    args.operand.dst.ptr = dst;
    const kai_matmul_uker_lhs_dim_args lhsShape = {m, k};
    const kai_matmul_uker_rhs_dim_args rhsShape = {n, k};
    // QAI4's wrapper consumes both strides verbatim; zero only works for one tile.
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhsShape);
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhsShape);
    args.operand.dst.stride.m = sr;
    // Bias and clamp belong after FP32 accumulation, before any FP16 output cast.
    api.run(&config, &args);
}
#undef DEFINE_RHS_INFO
size_t lhsSizeQai4Dot(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    const auto api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    kai_matmul_pack_lhs_uker_config config = {};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {m, k};
    const auto stride = api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_size(&config, &shape, &stride);
}
size_t lhsOffQai4Dot(size_t mIdx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    return mIdx * lhsSizeQai4Dot(1, k, bl, mr, kr, sr);
}
void lhsPackQai4Dot(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                    const void* lhs, void* out) {
    const auto api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    kai_matmul_pack_lhs_uker_config config = {};
    kai_matmul_pack_lhs_uker_args args = {};
    args.shape.m = m;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = k * sizeof(float);
    args.operand.lhs_packed.ptr = out;
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {m, k};
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &shape);
    api.run(&config, &args);
}
void matmulQai4Dot(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                  size_t sr, size_t sc, float mn, float mx) {
    MNN_ASSERT(m == 1 && sc == sizeof(float));
    const auto api = kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot();
    kai_matmul_uker_config config = {};
    config.format.bl = 32;
    kai_matmul_uker_args args = {};
    args.shape.m = m;
    args.shape.n = n;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.rhs.ptr = rhs;
    args.operand.dst.ptr = dst;
    const kai_matmul_uker_lhs_dim_args lhsShape = {m, k};
    const kai_matmul_uker_rhs_dim_args rhsShape = {n, k};
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhsShape);
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhsShape);
    args.operand.dst.stride.m = sr;
    // Like MOPA, DOT has no fused bias; clamp only after the executor adds it.
    api.run(&config, &args);
}
#undef DEFINE_LHS_INFO_CHNL
#undef DEFINE_LHS_INFO_BLK

} // namespace

// ===================================================================
// Per-instance kernel parameter resolution and ukernel dispatch.

void KleidiAIConvInt8::configKernel(bool allowQai4F16Mopa) {
    auto cpu = MNNGetCPUInfo();
    mSme2 = cpu->sme2;
    mDot  = cpu->dot;
    mI8mm = cpu->i8mm;
    mHybrid = false;
    mF16Mopa = false;
    mQai4F16Mopa = false;
    mParamGemv = KernelParam{};
    mUkernelGemv = Ukernel{};
    mParam = KernelParam{};
    mParamNeon = KernelParam{};
    mUkernel = Ukernel{};
    mUkernelNeon = Ukernel{};
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

    // Select the RHS layout before packing, using the model's original block
    // length, not the fixed BL32 used internally by the official FP16 kernel.
    // Larger blocks retain the QSI8 path and its matching GEMM/GEMV packers.
    const int blockSize = mBlockNum > 1 ? mCommon->inputCount() / mBlockNum : 0;
    const bool preferQai4F16 = blockSize > 0 && blockSize < 64;
    if (allowQai4F16Mopa && preferQai4F16 && mSme2
        && (mKernelType == KernelType::QI4_ASYM_PERBLOCK_F32
            || mKernelType == KernelType::QI4_ASYM_PERBLOCK_F16)) {
        const auto api = kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa();
        kai_matmul_uker_config config = {};
        config.format.bl = 32;
        const auto step = api.get_step(&config);
        mParam.mKaiMstepGemv = mParam.mKaiMstepGemm = step.m;
        mParam.mKaiMrGemv = mParam.mKaiMrGemm = step.m;
        mParam.mKaiNStep = mParam.mKaiNr = step.n;
        mParam.mKaiKr = 4;
        mParam.mKaiSr = 2;
        mUkernel.rhsPackedSize = rhsSizeQai4;
        mUkernel.rhsPackedOffset = rhsOffQai4;
        mUkernel.runRhsPack = rhsPackQai4;
        mUkernel.lhsPackedSize = lhsSizeF16Mopa;
        mUkernel.lhsPackedOffset = lhsOffF16Mopa;
        mUkernel.runLhsQuantPack = mKernelType == KernelType::QI4_ASYM_PERBLOCK_F32
            ? lhsPackF16MopaF32 : lhsPackF16MopaF16;
        mUkernel.matmul = matmulQai4F16Mopa;
        const auto gemvApi = kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot();
        const auto gemvStep = gemvApi.get_step(&config);
        MNN_ASSERT(gemvStep.n == step.n && gemvStep.m == 1);
        mParamGemv = mParam;
        mParamGemv.mKaiMrGemv = mParamGemv.mKaiMrGemm = gemvStep.m;
        mParamGemv.mKaiMstepGemv = mParamGemv.mKaiMstepGemm = gemvStep.m;
        mUkernelGemv = mUkernel;
        mUkernelGemv.lhsPackedSize = lhsSizeQai4Dot;
        mUkernelGemv.lhsPackedOffset = lhsOffQai4Dot;
        mUkernelGemv.runLhsQuantPack = lhsPackQai4Dot;
        mUkernelGemv.matmul = matmulQai4Dot;
        mF16Mopa = mQai4F16Mopa = true;
        // No hybrid policy has been calibrated for this packed FP16 route.
        return;
    }
    switch (mKernelType) {
        case KernelType::QI4_SYM_PERBLOCK_F32:
        case KernelType::QI4_SYM_PERBLOCK_F16: {
            if (!mSme2) {
                break;
            }
            const auto api = kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_f16p_4vsx16vs_sme2_mopa();
            kai_matmul_uker_config config = {};
            const auto step = api.get_step(&config);
            mParam.mKaiMstepGemv = mParam.mKaiMstepGemm = step.m;
            mParam.mKaiMrGemv = mParam.mKaiMrGemm = step.m;
            mParam.mKaiNStep = mParam.mKaiNr = step.n;
            mParam.mKaiKr = 4;
            mParam.mKaiSr = 2;
            mUkernel.rhsPackedSize = rhsSizeQsi4;
            mUkernel.rhsPackedOffset = rhsOffQsi4;
            mUkernel.runRhsPack = rhsPackQsi4;
            mUkernel.lhsPackedSize = lhsSizeF16Mopa;
            mUkernel.lhsPackedOffset = lhsOffF16Mopa;
            mUkernel.runLhsQuantPack = mKernelType == KernelType::QI4_SYM_PERBLOCK_F32
                ? lhsPackF16MopaF32 : lhsPackF16MopaF16;
            mUkernel.matmul = matmulQsi4F16Mopa;
            mF16Mopa = true;
            break;
        }
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
                fillSmeF32(mParam, mUkernel);
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

    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();
    if (ic <= 0 || oc <= 0 || mBlockNum <= 0 || ic % mBlockNum != 0 || (ic / mBlockNum) % 32 != 0) {
        return;
    }
    // Resolve the block-size policy only after validating the source shape.
    configKernel();
    if (mUkernel.matmul == nullptr) {
        return;
    }
    if (mQai4F16Mopa) {
        if (!quanCommon->asymmetric || mBlockNum <= 1 || quanCommon->alpha.size() < 2 * oc * mBlockNum) {
            return;
        }
        // Reject unsafe FP16 metadata before packing, so the QSI8 route keeps
        // its original FP32 scale/zero ABI. Its selection must survive cloning.
        for (int i = 0; i < oc * mBlockNum; ++i) {
            const float zero = quanCommon->alpha.get()[2 * i];
            const float scale = quanCommon->alpha.get()[2 * i + 1];
            const float offset = zero - 8.0f * scale;
            const float hs = static_cast<float>(static_cast<__fp16>(scale));
            const float ho = static_cast<float>(static_cast<__fp16>(offset));
            // Fused FP16 dequantization is affine over unsigned codes 0..15.
            // Checking both endpoints also bounds every intermediate code.
            if (!std::isfinite(scale) || !std::isfinite(zero) || !std::isfinite(hs)
                || !std::isfinite(ho) || std::fabs(ho + 15.0f * hs) > 65504.0f
                || (scale != 0.0f && hs == 0.0f) || (offset != 0.0f && ho == 0.0f)) {
                configKernel(false);
                break;
            }
        }
    }
    if (mF16Mopa && !mQai4F16Mopa) {
        if (quanCommon->asymmetric || mBlockNum <= 1 || quanCommon->alpha.size() < oc * mBlockNum) {
            return;
        }
        // QSI4 dequantizes in FP16. Reject scales whose conversion (or INT4
        // multiplication) would overflow, or erase an entire nonzero block.
        for (int i = 0; i < oc * mBlockNum; ++i) {
            const float scale = quanCommon->alpha.get()[i];
            const float rounded = static_cast<float>(static_cast<__fp16>(scale));
            if (!std::isfinite(rounded) || std::fabs(rounded) > 65504.0f / 8.0f
                || (scale != 0.0f && rounded == 0.0f)) {
                return;
            }
        }
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
        auto quanInfoPtr = quanCommon->getAlphaFloat();
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
    auto quanInfoPtr = quanCommon->getAlphaFloat();
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
            // alpha has only OC * blockNum entries, not the padded OC count.
            memcpy(scalePtr, (uint8_t*)quanInfoPtr, oc * mBlockNum * sizeof(float));
        }
    }

    if (mF16Mopa) {
        // Retain original F32 metadata for exceptional inputs. Shared DOT also
        // needs it with FP16 storage: a block sum can overflow even for finite
        // FP16 inputs. Neither fallback needs a second copy of INT4 codes.
        const bool retainMetadata = !bFP16 || mQai4F16Mopa;
        const size_t entries = static_cast<size_t>(oc)
            * (1 + (retainMetadata ? mBlockNum * (mQai4F16Mopa ? 2 : 1) : 0));
        if (entries > static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float)) {
            return;
        }
        mF16MopaBias.reset(Tensor::createDevice<uint8_t>({static_cast<int>(entries * sizeof(float))}));
        if (!backend->onAcquireBuffer(mF16MopaBias.get(), Backend::STATIC)) {
            return;
        }
        ::memcpy(mF16MopaBias->host<float>(), biasPtr, oc * sizeof(float));
        if (retainMetadata) {
            ::memcpy(mF16MopaBias->host<float>() + oc, scalePtr, oc * mBlockNum * sizeof(float));
            if (mQai4F16Mopa) {
                ::memcpy(mF16MopaBias->host<float>() + oc + oc * mBlockNum, zeroPtr,
                         oc * mBlockNum * sizeof(float));
            }
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
    mF16MopaBias(exe.mF16MopaBias),
    mKernelType(exe.mKernelType), mBlockNum(exe.mBlockNum) {
    configKernel(exe.mQai4F16Mopa);
}

KleidiAIConvInt8::~KleidiAIConvInt8() {
    // Do nothing
}

bool KleidiAIConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!valid()) {
        return false;
    }
    if (nullptr == dst) {
        // StaticModule probes clone support with both bn and dst null.
        return true;
    }
    // KernelType fixes both input/output storage precision. Sharing weights
    // across different backend precisions must rebuild, not reuse this slot.
    if (bn == nullptr || bn->type() != backend()->type()
        || static_cast<CPUBackend*>(bn)->functions()->bytes
        != static_cast<CPUBackend*>(backend())->functions()->bytes) {
        return false;
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

    mUseQai4Gemv = mQai4F16Mopa && m == 1;

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

    int packedSize = mUseQai4Gemv ? getLhsQuantedPackedSize(mUkernelGemv, mParamGemv, m, k, blkSize)
                                : getLhsQuantedPackedSize(m, k, blkSize);
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

    mGemvInputF32.reset();
    if (mUseQai4Gemv && core->bytes == 2) {
        if (k > static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float)) {
            return OUT_OF_MEMORY;
        }
        mGemvInputF32.reset(Tensor::createDevice<uint8_t>({static_cast<int>(k * sizeof(float))}));
        if (!b->onAcquireBuffer(mGemvInputF32.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
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

    if (mF16Mopa && core->bytes == 2) {
        const size_t outputBytes = m * n * sizeof(float);
        if (outputBytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return OUT_OF_MEMORY;
        }
        mF16MopaOutput.reset(Tensor::createDevice<uint8_t>({static_cast<int>(outputBytes)}));
        if (!backend()->onAcquireBuffer(mF16MopaOutput.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (mGemvInputF32) {
        b->onReleaseBuffer(mGemvInputF32.get(), Backend::DYNAMIC);
    }
    if (mHybrid) {
        backend()->onReleaseBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
    }
    if (mF16MopaOutput) {
        backend()->onReleaseBuffer(mF16MopaOutput.get(), Backend::DYNAMIC);
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

    if (mUseQai4Gemv && elementSize == 2) {
        core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(lhs), mGemvInputF32->host<float>(), k);
        lhs = mGemvInputF32->host<uint8_t>();
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

    // Official FP16 MOPA accumulates/stores FP32, regardless of backend storage
    // precision. Keep activation packing strides in the original element size.
    auto matmulDst = mF16MopaOutput ? mF16MopaOutput->host<uint8_t>() : dst;
    const size_t matmulElementSize = mF16Mopa ? sizeof(float) : elementSize;

    // Normal-precision callers may supply finite values outside FP16 range.
    // Never silently turn those into Inf during LHS packing. This rare path
    // evaluates original weights/metadata in F32 from the same packed codes.
    bool f32Fallback = false;
    if (mF16Mopa && (elementSize == sizeof(float) || mUseQai4Gemv)) {
        const auto inputF32 = reinterpret_cast<const float*>(lhs);
        for (size_t i = 0; i < m * k; ++i) {
            if (!std::isfinite(inputF32[i]) || std::fabs(inputF32[i]) > 65504.0f
                || (mUseQai4Gemv && inputF32[i] != 0.0f
                    && std::fabs(inputF32[i]) < 127.0f * std::numeric_limits<float>::min())) {
                f32Fallback = true;
                break;
            }
        }
    }

    if (mUseQai4Gemv && !f32Fallback) {
        auto packed = mTempIm2ColBuffer->host<int8_t>();
        packLhs(mUkernelGemv, mParamGemv, packed);
        // DOT LHS blocks contain [32 int8, half(sum(q)*scale), half(scale)].
        // The sum can overflow even when each source value fits in FP16.
        const auto src = reinterpret_cast<const float*>(lhs);
        for (size_t block = 0; block < k / 32; ++block) {
            const auto data = packed + block * 36;
            const auto meta = reinterpret_cast<const __fp16*>(data + 32);
            int sum = 0;
            bool nonzero = false;
            for (size_t i = 0; i < 32; ++i) {
                sum += data[i];
                nonzero = nonzero || src[block * 32 + i] != 0.0f;
            }
            if (!std::isfinite(static_cast<float>(meta[0])) || !std::isfinite(static_cast<float>(meta[1]))
                || (nonzero && meta[1] == 0) || (sum != 0 && meta[0] == 0)) {
                f32Fallback = true;
                break;
            }
        }
    }

    // The shape-dependent split was resolved during onResize().
    if (f32Fallback) {
        const size_t nr = getNr();
        const size_t blocks = k / blkSize;
        const size_t panelBytes = mQai4F16Mopa ? nr * (k / 32) * 20
                                             : nr * (k / 2 + blocks * sizeof(__fp16));
        const auto rhs = mWeightInt8->host<uint8_t>();
        const auto scales = mF16MopaBias->host<float>() + n;
        const auto zeros = mQai4F16Mopa ? scales + n * blocks : nullptr;
        const auto inputF32 = reinterpret_cast<const float*>(lhs);
        auto outputF32 = reinterpret_cast<float*>(matmulDst);
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            for (size_t row = tId; row < m; row += threadNum) {
                for (size_t col = 0; col < n; ++col) {
                    const auto panel = rhs + (col / nr) * panelBytes;
                    float sum = 0.0f;
                    for (size_t depth = 0; depth < k; ++depth) {
                        const size_t codeOffset = mQai4F16Mopa
                            ? (depth / 32) * 20 * nr + 2 * (((depth % 32) / 4) * nr + col % nr)
                            : 2 * ((depth / 4) * nr + col % nr);
                        const uint8_t pair = panel[codeOffset + (depth % 4) / 2];
                        const int q = ((pair >> (4 * (depth % 2))) & 15) - 8;
                        const size_t idx = col * blocks + depth / blkSize;
                        sum += inputF32[row * k + depth] * (q * scales[idx] + (zeros ? zeros[idx] : 0.0f));
                    }
                    outputF32[row * n + col] = sum;
                }
            }
        }
        MNN_CONCURRENCY_END();
    } else if (mSmeColumns == 0) {
        // The paired DOT changes only the LHS contract and shares the primary RHS.
        const auto& ukernel = mUseQai4Gemv ? mUkernelGemv : mUkernel;
        const auto& param = mUseQai4Gemv ? mParamGemv : mParam;
        auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
        auto rhsPacked = mWeightInt8->host<uint8_t>();
        if (!mUseQai4Gemv) {
            packLhs(ukernel, param, lhsPacked);
        }
        int matThreadNum = bSupportSme2() ? 1 : threadNum;
        int vecPerThread = getVecNumPerThread(n, matThreadNum, getNStep(param));
        int threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            int t = (int)tId;
            auto threadRhsPacked = rhsPacked + getRhsPackedOffset(ukernel, param, t * vecPerThread, k, blkSize);
            auto threadDst = matmulDst + getDstOffset(0, t * vecPerThread, n, matmulElementSize);
            int vecNum = (t == threadNeed - 1) ? (n - vecPerThread * t) : vecPerThread; //Last threadN may less than vecPerThread.
            runMatmul(ukernel, param, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst,
                      n * matmulElementSize, matmulElementSize, postPtr[3], postPtr[2]);
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

    if (mF16Mopa) {
        const auto values = reinterpret_cast<const float*>(matmulDst);
        const auto bias = mF16MopaBias->host<float>();
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            for (size_t row = tId; row < m; row += threadNum) {
                for (size_t col = 0; col < n; ++col) {
                    const size_t idx = row * n + col;
                    const float value = std::min(postPtr[3], std::max(postPtr[2], values[idx] + bias[col]));
                    if (elementSize == sizeof(float)) {
                        reinterpret_cast<float*>(dst)[idx] = value;
                    } else {
                        reinterpret_cast<__fp16*>(dst)[idx] = static_cast<__fp16>(value);
                    }
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
