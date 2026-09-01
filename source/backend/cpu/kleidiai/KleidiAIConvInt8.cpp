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
#include <arm_neon.h>
#include <string.h>
#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

// KleidiAI micro-kernel headers (int4 dynamic-quant matmul + packing).
// Symmetric per-channel INT4 uses KleidiAI's dedicated qai8dxp/qsi4cxp pack and
// matmul family; it does not reuse the asymmetric qai4c32 layout.
#include "kai_common.h"
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"

#define QUANT_INFO_BYTES 4
namespace MNN {

// ===================================================================
// Static classification / gating (moved out of the former KleidiAI class).

KleidiAIConvInt8::KernelType KleidiAIConvInt8::getKernelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes) {
    // Only 4-bit dynamic-quant weights are accelerated today. The variant is picked from
    // symmetry, quant granularity (per-channel when blockSize == 0, else per-block), and
    // activation precision (f32 when bytes == 4, f16 when bytes == 2). Anything else falls back.
    if (bits != 4) {
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
    if (common->group() != 1) {
        return false;
    }
    if (type == KernelType::QI4_ASYM_PERCHANNEL_F32 || type == KernelType::QI4_ASYM_PERCHANNEL_F16
        || type == KernelType::QI8_ASYM_PERCHANNEL) {
        if (common->inputCount() % 32 != 0) {
            return false;
        }
    }
    if (type == KernelType::QI4_SYM_PERCHANNEL_F32 && (common->inputCount() % 2 != 0)) {
        return false;
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

constexpr size_t kKaiNumBytesAdderRhs = 4;
constexpr size_t kKaiNumBytesMultiplierRhs = sizeof(float);
constexpr size_t kKaiNumBytesBias = sizeof(float);

inline size_t kaiKRoundedUpCompat(size_t k, size_t kr, size_t sr) {
    const size_t krSrRoundedUp4 = kai_roundup(kr * sr, 4);
    return kai_roundup(k, krSrRoundedUp4);
}

void rhsPackSymNeonCompat(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
                          const uint8_t* rhs, const float* bias, const float* scale, void* rhsPacked,
                          size_t extraBytes) {
    KAI_ASSERT(numGroups == 1);
    KAI_ASSERT(extraBytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != nullptr);
    KAI_ASSERT(scale != nullptr);
    KAI_ASSERT(rhsPacked != nullptr);

    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    const size_t rhsZeroPoint = params.rhs_zero_point;
    const size_t rhsPackedStride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(k, nr, kr, sr);
    const size_t kInternal = kaiKRoundedUpCompat(k, kr, sr);
    const size_t dstNumRows = kai_roundup(n, nr) / nr;
    const size_t dstNumBytesPerRow = nr * (kInternal / 2);
    const size_t blockLengthInBytes = kr / sr;
    const size_t kInterleavedV = 16U;
    const size_t rhsStride = kai_roundup(k, 2) / 2;

    for (size_t dstRowIdx = 0; dstRowIdx < dstNumRows; ++dstRowIdx) {
        uint8_t* dstRow = reinterpret_cast<uint8_t*>(rhsPacked) + dstRowIdx * rhsPackedStride;
        int32_t* sums = reinterpret_cast<int32_t*>(dstRow + nr * (kInternal / 2));
        memset(sums, 0, nr * sizeof(int32_t));

        for (size_t dstByteIdx = 0; dstByteIdx < dstNumBytesPerRow; ++dstByteIdx) {
            const size_t blockIdx = dstByteIdx / blockLengthInBytes;
            const size_t blockByteIdx = dstByteIdx % blockLengthInBytes;
            const size_t superBlockIdx = blockIdx / nr;
            const size_t nrIdx = blockIdx % nr;

            const size_t kAdjustment =
                ((blockByteIdx + superBlockIdx * blockLengthInBytes) / kInterleavedV) * kInterleavedV;
            const size_t k0Idx = blockByteIdx + superBlockIdx * blockLengthInBytes + kAdjustment;
            const size_t k1Idx = k0Idx + kInterleavedV;
            const size_t n0Idx = dstRowIdx * nr + nrIdx;
            const size_t n0ValidIdx = KAI_MIN(n0Idx, n - 1);

            const size_t srcAddrByte0 = (k0Idx / 2) + n0ValidIdx * rhsStride;
            const size_t srcAddrByte1 = (k1Idx / 2) + n0ValidIdx * rhsStride;

            uint8_t byte0 = rhsZeroPoint | (rhsZeroPoint << 4);
            uint8_t byte1 = rhsZeroPoint | (rhsZeroPoint << 4);

            if (k0Idx < k) {
                byte0 = rhs[srcAddrByte0];
            }
            if (k1Idx < k) {
                byte1 = rhs[srcAddrByte1];
            }

            const size_t shiftRightX0 = ((k0Idx + 1) % 2) * 4;
            const size_t shiftRightX1 = ((k1Idx + 1) % 2) * 4;
            const uint8_t srcX0Lo = (byte0 >> shiftRightX0) & 0x0F;
            const uint8_t srcX0Hi = (byte1 >> shiftRightX1) & 0x0F;

            sums[nrIdx] += (int32_t)srcX0Lo + (int32_t)srcX0Hi - 2 * (int32_t)rhsZeroPoint;
            *dstRow = (srcX0Lo | (srcX0Hi << 4)) ^ 0x88;
            dstRow += sizeof(uint8_t);
        }

        for (size_t i = 0; i < nr; ++i) {
            sums[i] = sums[i] * 16;
            dstRow += sizeof(int32_t);
        }

        for (size_t i = 0; i < nr; ++i) {
            const size_t srcRowIdx = KAI_MIN(dstRowIdx * nr + i, n - 1);
            *reinterpret_cast<float*>(dstRow) = scale[srcRowIdx] * 0.0625F;
            dstRow += sizeof(float);
        }

        if (bias == nullptr) {
            memset(dstRow, 0, nr * sizeof(float));
        } else {
            for (size_t i = 0; i < nr; ++i) {
                const size_t srcRowIdx = KAI_MIN(dstRowIdx * nr + i, n - 1);
                reinterpret_cast<float*>(dstRow)[i] = bias[srcRowIdx];
            }
        }
    }
}

void rhsPackSymSme2Compat(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
                          const uint8_t* rhs, const float* bias, const float* scale, void* rhsPacked,
                          size_t extraBytes) {
    const size_t kInternal = kaiKRoundedUpCompat(k, 16, 2);

    KAI_ASSERT((kInternal % kr) == 0);
    KAI_ASSERT(numGroups == 1);
    KAI_ASSERT(extraBytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != nullptr);
    KAI_ASSERT(scale != nullptr);
    KAI_ASSERT(rhsPacked != nullptr);

    struct kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    const int32_t rhsZeroPoint = params.rhs_zero_point;
    const size_t rhsStride = kai_roundup(k, 2) / 2;
    const size_t rhsPackedStride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(k, nr, kr, sr);
    const size_t dstNrBlockSize = nr * kr * sizeof(uint8_t) / 2;

    for (size_t rowIdx = 0; rowIdx < n; rowIdx += nr) {
        int8_t* const dstRow = reinterpret_cast<int8_t*>(rhsPacked) + ((rowIdx / nr) * rhsPackedStride);
        int32_t* const sums = reinterpret_cast<int32_t*>(dstRow + (nr * (kInternal / 2)));
        float* const scalingFactors =
            reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sums) + (nr * kKaiNumBytesAdderRhs));
        float* const biases =
            reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalingFactors) + (nr * kKaiNumBytesMultiplierRhs));

        memset(sums, 0, nr * kKaiNumBytesAdderRhs);

        size_t rowsLeft = n - rowIdx;
        if (rowsLeft >= nr) {
            memcpy(scalingFactors, &scale[rowIdx], nr * kKaiNumBytesMultiplierRhs);
            if (bias != nullptr) {
                memcpy(biases, &bias[rowIdx], nr * kKaiNumBytesBias);
            } else {
                memset(biases, 0, nr * kKaiNumBytesBias);
            }
        } else {
            memcpy(scalingFactors, &scale[rowIdx], rowsLeft * kKaiNumBytesMultiplierRhs);
            memset(&scalingFactors[rowsLeft], 0, (nr - rowsLeft) * kKaiNumBytesMultiplierRhs);
            if (bias != nullptr) {
                memcpy(biases, &bias[rowIdx], rowsLeft * kKaiNumBytesBias);
                memset(&biases[rowsLeft], 0, (nr - rowsLeft) * kKaiNumBytesBias);
            } else {
                memset(biases, 0, nr * kKaiNumBytesBias);
            }
        }

        for (size_t nrBlockIdx = 0; nrBlockIdx < nr; ++nrBlockIdx) {
            const uint8_t* const srcRow = rhs + ((rowIdx + nrBlockIdx) * rhsStride);
            int8_t* dstKrBlock = dstRow + (nrBlockIdx * kr / 2);

            int32_t sum = 0;
            for (size_t colIdx = 0; colIdx < kInternal; colIdx += kr) {
                for (size_t krBlockIdx = 0; krBlockIdx < kr; krBlockIdx += 2) {
                    if (rowIdx + nrBlockIdx >= n || colIdx + krBlockIdx >= k) {
                        dstKrBlock[krBlockIdx / 2] = 0;
                        continue;
                    }

                    const uint8_t dstByte = srcRow[(colIdx + krBlockIdx) / 2];
                    const int32_t secondValue = (dstByte & 0xF) - rhsZeroPoint;
                    const int32_t firstValue = colIdx + krBlockIdx + 1 >= k ? 0 : (dstByte >> 4) - rhsZeroPoint;
                    sum += firstValue + secondValue;

                    dstKrBlock[krBlockIdx / 2] = static_cast<int8_t>((secondValue << 4) | (firstValue & 0xF));
                }
                dstKrBlock += dstNrBlockSize;
            }

            sums[nrBlockIdx] = sum;
        }
    }
}

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
DEFINE_RHS_INFO(rhsSizeSymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, sr)
DEFINE_RHS_INFO(rhsSizeSymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, sr)
DEFINE_RHS_INFO(rhsSizeAsymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsSizeAsymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,      bl)

// ---- rhs packed offset ----
DEFINE_RHS_INFO(rhsOffSymSme2, kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, sr)
DEFINE_RHS_INFO(rhsOffSymNeon, kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, sr)
DEFINE_RHS_INFO(rhsOffAsymSme2,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsOffAsymNeon,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,    bl)

// ---- rhs pack ----
void rhsPackSymSme2(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                    const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    (void)bl;
    (void)zeroPoint;
    rhsPackSymSme2Compat(numGroups, n, k, nr, kr, sr, (const uint8_t*)rhs, (const float*)bias,
                         (const float*)scale, rhsPacked, 0);
}
void rhsPackSymNeon(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                    const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    (void)bl;
    (void)zeroPoint;
    rhsPackSymNeonCompat(numGroups, n, k, nr, kr, sr, (const uint8_t*)rhs, (const float*)bias,
                         (const float*)scale, rhsPacked, 0);
}
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
DEFINE_LHS_INFO_CHNL(lhsSizeSymF32, kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF32,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF16,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon)

// ---- lhs quanted packed offset ----
DEFINE_LHS_INFO_CHNL(lhsOffSymF32, kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32)
DEFINE_LHS_INFO_BLK(lhsOffAsymF32,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffAsymF16,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon)

// ---- lhs quant + pack ----
void lhsPackSymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    (void)bl;
    kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, kr, sr, 0, (const __fp16*)lhs, k * sizeof(__fp16), out);
}

// ---- matmul (GEMV when m == 1, GEMM otherwise) ----
void matmulSymF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                      size_t sr, size_t sc, float mn, float mx) {
    (void)bl;
    if (m == 1) {
        kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
            m, n, k, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
            m, n, k, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulSymF32Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                      size_t sr, size_t sc, float mn, float mx) {
    (void)bl;
    if (m == 1) {
        kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(
            m, n, k, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(
            m, n, k, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
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
    mPrimaryIsNeon = false;
    mChnlQuant = (mKernelType == KernelType::QI4_SYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F16);

    // Slot fillers. Each binds one (KernelParam, Ukernel) pair to a concrete kernel family so that
    // both the primary (SME) and, when hybrid, the secondary (NEON) slot are configured identically.
    auto fillSmeSymF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeSymF32;
        u.lhsPackedOffset = lhsOffSymF32;
        u.runLhsQuantPack = lhsPackSymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeSymSme2;
        u.rhsPackedOffset = rhsOffSymSme2;
        u.runRhsPack      = rhsPackSymSme2;
        u.matmul          = matmulSymF32Sme2;
    };
    auto fillNeonSymF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeSymF32;
        u.lhsPackedOffset = lhsOffSymF32;
        u.runLhsQuantPack = lhsPackSymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = 8;
        p.mKaiNStep     = 4;
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = 4;
        p.mKaiNr        = 4;
        p.mKaiKr        = 16;
        p.mKaiSr        = 2;
        u.rhsPackedSize   = rhsSizeSymNeon;
        u.rhsPackedOffset = rhsOffSymNeon;
        u.runRhsPack      = rhsPackSymNeon;
        u.matmul          = matmulSymF32Neon;
    };
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

    switch (mKernelType) {
        case KernelType::QI4_SYM_PERCHANNEL_F32:
            // Keep upstream's dedicated symmetric pack and one-engine selection. The hybrid
            // scheduler is calibrated only for the distinct asymmetric qai4c32 kernel family.
            if (mSme2) {
                fillSmeSymF32(mParam, mUkernel);
            } else if (mDot && mI8mm) {
                fillNeonSymF32(mParam, mUkernel);
                mPrimaryIsNeon = true;
            }
            break;
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
                mPrimaryIsNeon = true;
            }
            break;
        case KernelType::QI4_ASYM_PERBLOCK_F32:
            if (mSme2) {
                fillSmeF32(mParam, mUkernel);
            } else if (mDot && mI8mm) {
                fillNeonF32(mParam, mUkernel);
                mPrimaryIsNeon = true;
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
                mPrimaryIsNeon = true;
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
    // Resolve CPU features and kernel packing parameters for this KernelType.
    configKernel();

    // Convolution metadata.
    auto convOp = op->main_as_Convolution2D();
    const int oc = convOp->common()->outputCount();
    const int ic = convOp->common()->inputCount();

    // Backend metadata.
    auto core = static_cast<CPUBackend*>(backend)->functions();
    const int pack = core->pack;

    const int ocUp4 = ROUND_UP(oc, pack);
    const int scaleSize = ocUp4 * mBlockNum;

    const bool bAsym = quanCommon->asymmetric;
    const size_t blkSize = mBlockNum == 1 ? 0 : ic / mBlockNum;

    AutoStorage<int8_t> reorderedQuantInfo;
    reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
    if (reorderedQuantInfo.get() == nullptr) {
        MNN_ERROR("Memory not enough\n");
        return;
    }

    // Symmetric per-channel weights use the upstream qsi4cxp packer, whose scale array is
    // indexed directly by output channel. Asymmetric kernels repack their [zero, scale] pairs
    // below into their own layout.
    {
        int outputCount = convOp->common()->outputCount();
        auto quanInfoPtr = quanCommon->alpha.get();
        auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
        auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
        if (!bAsym) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstScale = scalePtr + i * ocUp4;
                auto dstZero = zeroPtr + i * ocUp4;
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstScale[j] = quanInfoPtr[scaleIndex];
                    dstZero[j] = 0.f;
                }
            }
        }
        ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
    }

    const int n = oc;
    const int k = ic;
    const int packedWeightSize = getRhsPackedSize(n, k, blkSize);

    // Allocate the packed weight tensor.
    mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
    const bool success = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);

    if (!success) {
        MNN_ERROR("Out of static memory!\n");
        return;
    }

    const size_t paraNum = scaleSize;
    float* scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    float* zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
    float* biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
    // Reload asymmetric quantization parameters in the ukernels' linear layout.
    auto quanInfoPtr = quanCommon->alpha.get();
    auto alphaSize = quanCommon->alpha.size();
    if (bAsym) {
        for (int i = 0; i < paraNum; i++) {
            if (i * 2 >= alphaSize) {
                zeroPtr[i] = 0;
                scalePtr[i] = 0;
            } else {
                zeroPtr[i] = quanInfoPtr[i * 2];
                scalePtr[i] = quanInfoPtr[i * 2 + 1];
            }
        }
    } else if (blkSize != 0) {
        memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
    }

    // Pack the weights.
    auto weightPackedData = mWeightInt8->host<uint8_t>();
    runRhsPack(1, n, k, blkSize,
               (uint8_t*)quanCommon->weight.get(),
               (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
               weightPackedData);

    if (mHybrid) {
        // Pack a second copy of the weights in the NEON slot layout so the NEON kernels can run
        // concurrently with the SME kernel on the remaining threads. Same scale/zero/bias, but a
        // different packed layout, hence a separate static buffer (~2x weight memory).
        const int packedWeightSizeNeon = getRhsPackedSize(mUkernelNeon, mParamNeon, n, k, blkSize);
        mWeightInt8Neon.reset(Tensor::createDevice<uint8_t>({packedWeightSizeNeon}));
        const bool successNeon = backend->onAcquireBuffer(mWeightInt8Neon.get(), Backend::STATIC);
        if (!successNeon) {
            MNN_ERROR("Out of static memory!\n");
            return;
        }
        runRhsPack(mUkernelNeon, mParamNeon, 1, n, k, blkSize,
                   (uint8_t*)quanCommon->weight.get(),
                   (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                   mWeightInt8Neon->host<uint8_t>());
    }
    return;
}

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend),
    mWeightInt8(exe.mWeightInt8), mTempIm2ColBuffer(exe.mTempIm2ColBuffer),
    mWeightInt8Neon(exe.mWeightInt8Neon),
    mKernelType(exe.mKernelType), mBlockNum(exe.mBlockNum) {
    configKernel();
}

KleidiAIConvInt8::~KleidiAIConvInt8() {
    // Do nothing
}

bool KleidiAIConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvInt8(bn, op, *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

ErrorCode KleidiAIConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto b = backend();

    const size_t m = inputs[0]->batch() * inputs[0]->width() * inputs[0]->height();
    const size_t k = inputs[0]->channel();
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    if (inputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        mInputConvertBuffer.reset(Tensor::createDevice(
            std::vector<int>{input->batch(), input->height(), input->width(), input->channel()}, dataType,
            Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        mOutputConvertBuffer.reset(Tensor::createDevice(
            std::vector<int>{output->batch(), output->height(), output->width(), output->channel()}, dataType,
            Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    const int packedSize = getLhsQuantedPackedSize(m, k, blkSize);

    // Allocate the primary packed-LHS buffer.
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
    const bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        MNN_ERROR("Out of dynamic memory!\n");
        return OUT_OF_MEMORY;
    }

    if (mHybrid) {
        // The NEON slot packs lhs with a different mr, so it needs its own packed buffer.
        const int packedSizeNeon = getLhsQuantedPackedSize(mUkernelNeon, mParamNeon, m, k, blkSize);
        mTempIm2ColBufferNeon.reset(Tensor::createDevice<int8_t>({packedSizeNeon}));
        const bool successNeon = backend()->onAcquireBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
        if (!successNeon) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (mHybrid) {
        backend()->onReleaseBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
    }

    if (inputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

// ---------------------------------------------------------------------------
// Hybrid matmul latency cost model.
//
// Each table entry estimates the matmul latency in microseconds for one worker
// on a scheduling profile, execution engine, kernel variant, and workload. The
// estimate is used only to compare candidate SME/NEON N-dimension splits; it is
// a scheduling heuristic rather than an absolute performance model.
//
// All entries use the same linear equation:
//   t = constant + linear * units + work * workItems + block * units * blocks
//       + narrow * narrowPanel.
// `units` and `workItems` are engine- and workload-specific measures of tiled
// work. `blocks` captures per-block quantization overhead, and `narrowPanel`
// captures the penalty of an incomplete N panel when applicable.
//
// The evaluator defines these terms centrally, so supporting another profile
// or kernel only requires adding fitted data instead of a scheduling branch.
// ---------------------------------------------------------------------------
enum class KaiExecutionEngine {
    Sme,
    Neon,
};

enum class KaiKernelVariant {
    F32PerChannel,
    F16PerChannel,
    F16PerBlock,
    Unknown,
};

enum class KaiWorkload {
    Gemv,
    Gemm,
};

struct KaiCostParameters {
    double constant;
    double linear;
    double work;
    double block;
    double narrow;
};

struct KaiCostModel {
    KaiExecutionEngine engine;
    KaiKernelVariant kernel;
    KaiWorkload workload;
    KaiCostParameters parameters;
};

// macOS uses a separately calibrated profile. Engine and workload select the
// common equation's units and work-items basis.
static const KaiCostModel kKaiMacCostModels[] = {
    // macOS: SME GEMV.
    {KaiExecutionEngine::Sme, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemv, {0.0, 0.047204, 3.31259e-4, 0.0, 0.0}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemv, {0.0, 0.0732, 2.539e-4, 0.0, 0.0}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemv, {0.083925, 0.00565402, 2.7488e-4, 0.00773111, 0.0}},

    // macOS: SME GEMM.
    {KaiExecutionEngine::Sme, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemm, {0.272156, 0.258691, 5.210733e-4, 0.0, 0.558753}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemm, {0.3797, 0.29089, 4.836073e-4, 0.0, 0.43818}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemm, {0.4685, 0.09406, 3.682232e-4, 0.19966, 0.0}},

    // macOS: NEON GEMV.
    {KaiExecutionEngine::Neon, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemv, {0.0381022, -0.00302672, 6.469778e-5, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemv, {0.0369377, -0.0029569, 6.484345e-5, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemv, {0.0300534, 3.04287e-4, 4.384333e-5, -1.019584e-4, 0.0}},

    // macOS: NEON GEMM.
    {KaiExecutionEngine::Neon, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemm, {0.022823, 8.06611e-4, 5.522018e-6, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemm, {0.0173908, 0.00132875, 5.538539e-6, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemm, {-0.00574413, 0.00306348, 4.992718e-6, 7.96544e-4, 0.0}},
};

// The default profile uses the Android-calibrated coefficients for every
// non-macOS target. It intentionally has no CPU, SoC, or device-model matching.
static const KaiCostModel kKaiDefaultCostModels[] = {
    // Per-channel rows use blockSize == 0, so their block coefficients are zero.
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemv, {15.589006, 0.056143701, 0.00067438572, 0.0, 0.0}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemm, {11.059853, 0.1388165, 0.0048144657, 0.0, 0.59507372}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemv, {1.8502469, 0.0, 5.5104502e-5, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerBlock,
     KaiWorkload::Gemm, {10.462774, 0.0079093951, 1.1897356e-5, 0.0, 0.0}},

    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemv, {15.471472, 0.065109692, 0.00058154598, 0.0, 0.0}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemm, {10.737841, 0.40824941, 0.0013750797, 0.0, 0.20343534}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemv, {1.9456034, 0.0, 5.6246484e-5, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F16PerChannel,
     KaiWorkload::Gemm, {3.8408735, 0.0064410004, 3.093288e-6, 0.0, 0.0}},

    {KaiExecutionEngine::Sme, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemv, {15.307163, 0.05562534, 0.00059049123, 0.0, 0.0}},
    {KaiExecutionEngine::Sme, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemm, {10.421552, 0.28919573, 0.0013874438, 0.0, 0.033247118}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemv, {1.9155689, 0.0, 5.5334761e-5, 0.0, 0.0}},
    {KaiExecutionEngine::Neon, KaiKernelVariant::F32PerChannel,
     KaiWorkload::Gemm, {3.4662585, 0.0057725285, 3.0581624e-6, 0.0, 0.0}},
};

static KaiKernelVariant kaiGetKernelVariant(KleidiAIConvInt8::KernelType type) {
    switch (type) {
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F32:
            return KaiKernelVariant::F32PerChannel;
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F16:
            return KaiKernelVariant::F16PerChannel;
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F16:
            return KaiKernelVariant::F16PerBlock;
        default:
            return KaiKernelVariant::Unknown;
    }
}

static const KaiCostModel* kaiFindCostModelInTable(const KaiCostModel* models, size_t modelCount,
                                                   KaiExecutionEngine engine, KaiKernelVariant kernel,
                                                   KaiWorkload workload) {
    for (size_t i = 0; i < modelCount; ++i) {
        const auto& model = models[i];
        if (model.engine == engine && model.kernel == kernel && model.workload == workload) {
            return &model;
        }
    }
    return nullptr;
}

static const KaiCostModel* kaiFindCostModel(KaiExecutionEngine engine, KaiKernelVariant kernel,
                                            KaiWorkload workload) {
#if defined(__APPLE__) && TARGET_OS_OSX
    return kaiFindCostModelInTable(kKaiMacCostModels,
                                   sizeof(kKaiMacCostModels) / sizeof(kKaiMacCostModels[0]),
                                   engine, kernel, workload);
#else
    return kaiFindCostModelInTable(kKaiDefaultCostModels,
                                   sizeof(kKaiDefaultCostModels) / sizeof(kKaiDefaultCostModels[0]),
                                   engine, kernel, workload);
#endif
}

static double kaiEvaluateCost(const KaiCostModel& model, size_t m, size_t nCols, size_t k, size_t blkSize) {
    const size_t nr = model.engine == KaiExecutionEngine::Sme ? 64 : 4;
    const size_t mr = model.engine == KaiExecutionEngine::Sme ? 16 : 8;
    const size_t nPanel = (nCols + nr - 1) / nr;
    const double blocks = blkSize == 0 ? 0.0 : (double)k / (double)blkSize;
    const KaiCostParameters& p = model.parameters;
    double units = (double)nPanel;
    double workItems = units * (double)k;
    double narrowPanel = 0.0;
    if (model.workload == KaiWorkload::Gemm) {
        const size_t mTile = (m + mr - 1) / mr;
        units = (double)mTile * (double)nPanel;
        if (model.engine == KaiExecutionEngine::Sme) {
            const double kpad = (double)(((k + 31) / 32) * 32);
            workItems = units * kpad;
            if (nPanel == 1 && nCols < nr) {
                narrowPanel = (double)mTile * (double)(nr - nCols) / (double)nr;
            }
        } else {
            workItems = (double)m * (double)nCols * (double)k;
        }
    }
    return p.constant + p.linear * units + p.work * workItems + p.block * units * blocks
           + p.narrow * narrowPanel;
}

static double kaiEstimateUs(const KaiCostModel& model, size_t m, size_t nCols, size_t k, size_t blkSize) {
    if (m == 0 || nCols == 0) {
        return 0.0;
    }
    // A fitted linear model can otherwise produce an impossible negative cost
    // when a future shape falls outside its calibration range.
    const double cost = kaiEvaluateCost(model, m, nCols, k, blkSize);
    return cost > 0.0 ? cost : 0.0;
}

ErrorCode KleidiAIConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

    // Initialize for convert
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto outputDes = TensorUtils::getDescribe(outputs[0]);

    const size_t m = input->batch() * input->width() * input->height(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    const size_t elementSize = core->bytes;

    auto lhs = input->host<uint8_t>();
    const int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    if (inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
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
    if (outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        //store matmul result to convert buffer.
        dst = mOutputConvertBuffer->host<uint8_t>();
    }
    auto postPtr = getPostParameters();

    // Decide whether to use the primary slot, split N between the SME and NEON
    // slots, or run the secondary NEON slot across all workers.
    enum class MatmulDispatch {
        Primary,
        Hybrid,
        NeonOnly,
    };
    MatmulDispatch dispatch = mHybrid && threadNum > 1 ? MatmulDispatch::Hybrid : MatmulDispatch::Primary;
    size_t nSme = 0, nNeon = 0;
    int neonThreads = 0;
    if (dispatch == MatmulDispatch::Hybrid) {
        const KaiKernelVariant kernel = kaiGetKernelVariant(mKernelType);
        const KaiWorkload workload = m == 1 ? KaiWorkload::Gemv : KaiWorkload::Gemm;
        const KaiCostModel* smeModel =
            kaiFindCostModel(KaiExecutionEngine::Sme, kernel, workload);
        const KaiCostModel* neonModel =
            kaiFindCostModel(KaiExecutionEngine::Neon, kernel, workload);
        if (smeModel == nullptr || neonModel == nullptr) {
            dispatch = MatmulDispatch::Primary;
        } else {
            neonThreads = threadNum - 1;
            const auto estimateNeonFinish = [&](size_t nCols, size_t workerCount) {
                const size_t nPerWorker = std::min(
                    nCols, getVecNumPerThread(nCols, workerCount, getNStep(mParamNeon)));
                return kaiEstimateUs(*neonModel, m, nPerWorker, k, blkSize);
            };
            // Include an all-NEON candidate. It uses every worker, whereas mixed
            // execution reserves thread 0 for SME and has only `neonThreads` NEON workers.
            const double neonOnlyFinish = estimateNeonFinish(n, threadNum);
            // Balance the N-split with the calibrated cost-model table: the SME slot runs
            // columns [0, nSme) on one thread concurrently with the NEON slot running the remaining
            // columns spread over neonThreads. Pick the SME column count (aligned to a whole number of
            // SME N-steps) that minimises the concurrent finish time max(t_sme, t_neon_per_thread).
            const size_t nStepSme = getNStep(mParam);
            size_t bestNSme = nStepSme;
            double bestFinish = 1e300;
            bool haveSplitCandidate = false;
            for (size_t candidate = nStepSme; candidate < n; candidate += nStepSme) {
                const size_t nNeonCandidate = n - candidate;
                haveSplitCandidate = true;
                const double tSme = kaiEstimateUs(*smeModel, m, candidate, k, blkSize);
                const double tNeon = estimateNeonFinish(nNeonCandidate, neonThreads);
                const double finish = std::max(tSme, tNeon);
                if (finish < bestFinish) {
                    bestFinish = finish;
                    bestNSme = candidate;
                }
            }
            if (!haveSplitCandidate) {
                // There is no non-empty SME/NEON split for N <= one SME panel. Keep the
                // old single-SME fallback unless all-worker NEON is predicted to finish first.
                const double smeOnlyFinish = kaiEstimateUs(*smeModel, m, n, k, blkSize);
                dispatch = neonOnlyFinish < smeOnlyFinish ? MatmulDispatch::NeonOnly
                                                           : MatmulDispatch::Primary;
            } else if (neonOnlyFinish < bestFinish) {
                // Pure NEON wins the same max-worker finish-time objective, so do not
                // assign a mandatory minimum SME panel just because SME is available.
                dispatch = MatmulDispatch::NeonOnly;
            } else {
                nSme = bestNSme;
                nNeon = n - nSme;
            }
        }
    }

    if (dispatch != MatmulDispatch::Hybrid) {
        // The normal SME fallback executes on one thread. A selected all-NEON candidate
        // instead uses the secondary NEON slot and all available threads.
        const bool useNeonOnly = dispatch == MatmulDispatch::NeonOnly;
        const Ukernel& activeUkernel = useNeonOnly ? mUkernelNeon : mUkernel;
        const KernelParam& activeParam = useNeonOnly ? mParamNeon : mParam;
        auto lhsPacked = (useNeonOnly ? mTempIm2ColBufferNeon : mTempIm2ColBuffer)->host<int8_t>();
        auto rhsPacked = (useNeonOnly ? mWeightInt8Neon : mWeightInt8)->host<uint8_t>();
        packLhs(activeUkernel, activeParam, lhsPacked);

        const int matThreadNum = (mPrimaryIsNeon || useNeonOnly) ? threadNum : 1;
        const int vecPerThread = getVecNumPerThread(n, matThreadNum, getNStep(activeParam));
        const int threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            const int t = (int)tId;
            auto threadRhsPacked = rhsPacked + getRhsPackedOffset(activeUkernel, activeParam, t * vecPerThread, k, blkSize);
            auto threadDst = dst + getDstOffset(0, t * vecPerThread, n, elementSize);
            const int vecNum = (t == threadNeed - 1) ? (n - vecPerThread * t) : vecPerThread;
            runMatmul(activeUkernel, activeParam, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst,
                      n * elementSize, elementSize, postPtr[3], postPtr[2]);
        }
        MNN_CONCURRENCY_END();
    } else {
        // Hybrid path: pack lhs once per slot (different mr => different packed layout), then run the
        // SME kernel on thread 0 over columns [0, nSme) concurrently with NEON kernels on the
        // remaining threads over columns [nSme, n).
        auto lhsPackedSme  = mTempIm2ColBuffer->host<int8_t>();
        auto lhsPackedNeon = mTempIm2ColBufferNeon->host<int8_t>();
        packLhs(mUkernel, mParam, lhsPackedSme);
        packLhs(mUkernelNeon, mParamNeon, lhsPackedNeon);
        auto rhsPackedSme  = mWeightInt8->host<uint8_t>();
        auto rhsPackedNeon = mWeightInt8Neon->host<uint8_t>();
        size_t nStepNeon = getNStep(mParamNeon);
        int vecPerNeon = getVecNumPerThread(nNeon, neonThreads, nStepNeon);
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            int t = (int)tId;
            if (t == 0) {
                // SME slot: columns [0, nSme).
                runMatmul(mUkernel, mParam, m, nSme, k, blkSize, lhsPackedSme, rhsPackedSme,
                          dst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
            } else {
                // NEON slot: columns [nSme, n) split among neonThreads.
                int neonId = t - 1;
                int localStart = neonId * vecPerNeon;
                if (localStart < (int)nNeon) {
                    int vecNum = (localStart + vecPerNeon > (int)nNeon) ? ((int)nNeon - localStart) : vecPerNeon;
                    size_t globalStart = nSme + (size_t)localStart;
                    auto threadRhsPacked = rhsPackedNeon + getRhsPackedOffset(mUkernelNeon, mParamNeon, globalStart, k, blkSize);
                    auto threadDst = dst + getDstOffset(0, globalStart, n, elementSize);
                    runMatmul(mUkernelNeon, mParamNeon, m, vecNum, k, blkSize, lhsPackedNeon,
                              threadRhsPacked, threadDst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    if (outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
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
