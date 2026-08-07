//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvInt8.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

// KleidiAI micro-kernel headers (int4 / int8 dynamic-quant matmul + packing).
// Keep a dedicated symmetric per-channel int4 path (qai8dxp/qsi4cxp) to preserve
// legacy behavior and support K values that are only 2-aligned (for example K=16/48).
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

KleidiAIConvInt8::KernelType KleidiAIConvInt8::getKernelType(size_t bits, bool bAsymmetric, size_t blockSize,
                                                             size_t bytes) {
    // Only 4-bit dynamic-quant weights are accelerated today. The variant is picked from
    // symmetry, quant granularity (per-channel when blockSize == 0, else per-block) and the
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
    if (common->group() != 1) {
        return false;
    }
    if (type == KernelType::QI4_ASYM_PERCHANNEL_F32 || type == KernelType::QI4_ASYM_PERCHANNEL_F16 ||
        type == KernelType::QI8_ASYM_PERCHANNEL) {
        if (common->inputCount() % 32 != 0) {
            return false;
        }
    }
    if (type == KernelType::QI4_SYM_PERCHANNEL_F32 && (common->inputCount() % 2 != 0)) {
        return false;
    }
    if (common->kernelX() == 1 && common->kernelY() == 1 && common->padX() == 0 && common->padY() == 0 &&
        common->strideX() == 1 && common->strideY() == 1 && common->dilateX() == 1 && common->dilateY() == 1) {
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

// Legacy-compatible qsi4cxp rhs packing wrappers. Keep these transformations to
// preserve historical symmetric int4 numerics for IC values accepted by the old path.
void rhsPackSymNeonCompat(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs,
                          const float* bias, const float* scale, void* rhsPacked, size_t extraBytes) {
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

            const uint8_t dstQs0 = srcX0Lo | (srcX0Hi << 4);
            *dstRow = dstQs0 ^ 0x88;
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

void rhsPackSymSme2Compat(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs,
                          const float* bias, const float* scale, void* rhsPacked, size_t extraBytes) {
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
#define DEFINE_RHS_INFO(NAME, KAIFN, LAST)                                          \
    size_t NAME(size_t idx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) { \
        (void)sr;                                                                   \
        (void)bl;                                                                   \
        return KAIFN(idx, k, nr, kr, LAST);                                         \
    }
#define DEFINE_LHS_INFO_CHNL(NAME, KAIFN)                                           \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        (void)bl;                                                                   \
        return KAIFN(idx, k, mr, kr, sr);                                           \
    }
#define DEFINE_LHS_INFO_BLK(NAME, KAIFN)                                            \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        return KAIFN(idx, k, bl, mr, kr, sr);                                       \
    }

// ---- rhs packed size ----
DEFINE_RHS_INFO(rhsSizeSymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, sr)
DEFINE_RHS_INFO(rhsSizeSymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, sr)
DEFINE_RHS_INFO(rhsSizeAsymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsSizeAsymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, bl)

// ---- rhs packed offset ----
DEFINE_RHS_INFO(rhsOffSymSme2, kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon, sr)
DEFINE_RHS_INFO(rhsOffSymNeon, kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, sr)
DEFINE_RHS_INFO(rhsOffAsymSme2, kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon,
                bl)
DEFINE_RHS_INFO(rhsOffAsymNeon, kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon, bl)

// ---- rhs pack ----
void rhsPackSymSme2(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs,
                    const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    (void)bl;
    (void)zeroPoint;
    rhsPackSymSme2Compat(numGroups, n, k, nr, kr, sr, (const uint8_t*)rhs, (const float*)bias, (const float*)scale,
                         rhsPacked, 0);
}
void rhsPackSymNeon(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs,
                    const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    (void)bl;
    (void)zeroPoint;
    rhsPackSymNeonCompat(numGroups, n, k, nr, kr, sr, (const uint8_t*)rhs, (const float*)bias, (const float*)scale,
                         rhsPacked, 0);
}
void rhsPackAsymSme2(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs,
                     const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
        numGroups, n, k, nr, kr, sr, bl, (const uint8_t*)rhs, zeroPoint, bias, scale, rhsPacked, 0, &params);
}
void rhsPackAsymNeon(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs,
                     const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, nr, kr, sr, bl, (const uint8_t*)rhs,
                                                               zeroPoint, bias, scale, rhsPacked, 0, &params);
}

// ---- lhs quanted packed size ----
DEFINE_LHS_INFO_CHNL(lhsSizeSymF32, kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF32, kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF16, kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon)

// ---- lhs quanted packed offset ----
DEFINE_LHS_INFO_CHNL(lhsOffSymF32, kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32)
DEFINE_LHS_INFO_BLK(lhsOffAsymF32, kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffAsymF16, kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon)

// ---- lhs quant + pack ----
void lhsPackSymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    (void)bl;
    kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float),
                                                     out);
}
void lhsPackAsymF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, kr, sr, 0, (const __fp16*)lhs, k * sizeof(__fp16),
                                                     out);
}

// ---- matmul (GEMV when m == 1, GEMM otherwise) ----
void matmulSymF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                      size_t sc, float mn, float mx) {
    (void)bl;
    if (m == 1) {
        kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(m, n, k, lhs, rhs, (float*)dst, sr, sc, mn,
                                                                         mx);
    } else {
        kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(m, n, k, lhs, rhs, (float*)dst, sr, sc, mn,
                                                                             mx);
    }
}
void matmulSymF32Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                      size_t sc, float mn, float mx) {
    (void)bl;
    if (m == 1) {
        kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(m, n, k, lhs, rhs, (float*)dst, sr, sc, mn,
                                                                           mx);
    } else {
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(m, n, k, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                       size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc,
                                                                          mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr,
                                                                               sc, mn, mx);
    }
}
void matmulAsymF32Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                       size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc,
                                                                          mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn,
                                                                       mx);
    }
}
void matmulAsymF16Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                       size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc,
                                                                          mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr,
                                                                               sc, mn, mx);
    }
}
void matmulAsymF16Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst, size_t sr,
                       size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc,
                                                                          mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn,
                                                                       mx);
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
    mDot = cpu->dot;
    mI8mm = cpu->i8mm;
    mChnlQuant =
        (mKernelType == KernelType::QI4_SYM_PERCHANNEL_F32 || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F32 ||
         mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F16);

    KernelParam& p = mParam;
    Ukernel& u = mUkernel;
    switch (mKernelType) {
        case KernelType::QI4_SYM_PERCHANNEL_F32:
            u.lhsPackedSize = lhsSizeSymF32;
            u.lhsPackedOffset = lhsOffSymF32;
            u.runLhsQuantPack = lhsPackSymF32;
            if (mSme2) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                p.mKaiNStep = kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                p.mKaiNr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                p.mKaiKr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                p.mKaiSr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                u.rhsPackedSize = rhsSizeSymSme2;
                u.rhsPackedOffset = rhsOffSymSme2;
                u.runRhsPack = rhsPackSymSme2;
                u.matmul = matmulSymF32Sme2;
            } else if (mDot && mI8mm) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = 8;
                p.mKaiNStep = 4;
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = 4;
                p.mKaiNr = 4;
                p.mKaiKr = 16;
                p.mKaiSr = 2;
                u.rhsPackedSize = rhsSizeSymNeon;
                u.rhsPackedOffset = rhsOffSymNeon;
                u.runRhsPack = rhsPackSymNeon;
                u.matmul = matmulSymF32Neon;
            }
            break;
        case KernelType::QI4_ASYM_PERCHANNEL_F32:
        case KernelType::QI4_ASYM_PERBLOCK_F32:
            u.lhsPackedSize = lhsSizeAsymF32;
            u.lhsPackedOffset = lhsOffAsymF32;
            u.runLhsQuantPack = lhsPackAsymF32;
            if (mSme2) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiNStep = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiNr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiKr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiSr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                u.rhsPackedSize = rhsSizeAsymSme2;
                u.rhsPackedOffset = rhsOffAsymSme2;
                u.runRhsPack = rhsPackAsymSme2;
                u.matmul = matmulAsymF32Sme2;
            } else if (mDot && mI8mm) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = 8;
                p.mKaiNStep = 4;
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = 4;
                p.mKaiNr = 4;
                p.mKaiKr = 16;
                p.mKaiSr = 2;
                u.rhsPackedSize = rhsSizeAsymNeon;
                u.rhsPackedOffset = rhsOffAsymNeon;
                u.runRhsPack = rhsPackAsymNeon;
                u.matmul = matmulAsymF32Neon;
            }
            break;
        case KernelType::QI4_ASYM_PERCHANNEL_F16:
        case KernelType::QI4_ASYM_PERBLOCK_F16:
            u.lhsPackedSize = lhsSizeAsymF16;
            u.lhsPackedOffset = lhsOffAsymF16;
            u.runLhsQuantPack = lhsPackAsymF16;
            if (mSme2) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiNStep = kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiNr = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiKr = kai_get_kr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                p.mKaiSr = kai_get_sr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
                u.rhsPackedSize = rhsSizeAsymSme2;
                u.rhsPackedOffset = rhsOffAsymSme2;
                u.runRhsPack = rhsPackAsymSme2;
                u.matmul = matmulAsymF16Sme2;
            } else if (mDot && mI8mm) {
                p.mKaiMstepGemv = 1;
                p.mKaiMstepGemm = 8;
                p.mKaiNStep = 4;
                p.mKaiMrGemv = 1;
                p.mKaiMrGemm = 4;
                p.mKaiNr = 4;
                p.mKaiKr = 16;
                p.mKaiSr = 2;
                u.rhsPackedSize = rhsSizeAsymNeon;
                u.rhsPackedOffset = rhsOffAsymNeon;
                u.runRhsPack = rhsPackAsymNeon;
                u.matmul = matmulAsymF16Neon;
            }
            break;
        default:
            break;
    }
}

size_t KleidiAIConvInt8::getRhsPackedSize(size_t n, size_t k, size_t bl) const {
    return mUkernel.rhsPackedSize(n, k, getNr(), getKr(), getSr(), mChnlQuant ? k : bl);
}

size_t KleidiAIConvInt8::getRhsPackedOffset(size_t nIdx, size_t k, size_t bl) const {
    if (nIdx == 0) {
        return 0;
    }
    return mUkernel.rhsPackedOffset(nIdx, k, getNr(), getKr(), getSr(), mChnlQuant ? k : bl);
}

void KleidiAIConvInt8::runRhsPack(size_t numGroups, size_t n, size_t k, size_t bl, const void* rhs, const void* scale,
                                  const void* zeroPoint, const void* bias, void* rhsPacked) const {
    mUkernel.runRhsPack(numGroups, n, k, getNr(), getKr(), getSr(), mChnlQuant ? k : bl, rhs, scale, zeroPoint, bias,
                        rhsPacked);
}

size_t KleidiAIConvInt8::getLhsQuantedPackedSize(size_t m, size_t k, size_t bl) const {
    return mUkernel.lhsPackedSize(m, k, mChnlQuant ? k : bl, getMr(m), getKr(), getSr());
}

size_t KleidiAIConvInt8::getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k, size_t bl) const {
    if (mIdx == 0) {
        return 0;
    }
    return mUkernel.lhsPackedOffset(mIdx, k, mChnlQuant ? k : bl, getMr(m), getKr(), getSr());
}

void KleidiAIConvInt8::runLhsQuantPack(size_t m, size_t k, size_t bl, size_t mr, const void* lhs,
                                       void* lhsQuantedPacked) const {
    mUkernel.runLhsQuantPack(m, k, mChnlQuant ? k : bl, mr, getKr(), getSr(), lhs, lhsQuantedPacked);
}

void KleidiAIConvInt8::runMatmul(size_t m, size_t n, size_t k, size_t bl, const void* lhsPacked, const void* rhsPacked,
                                 void* dst, size_t dstStrideRow, size_t dstStrideCol, const float scalarMax,
                                 const float scalarMin) const {
    mUkernel.matmul(m, n, k, mChnlQuant ? k : bl, lhsPacked, rhsPacked, dst, dstStrideRow, dstStrideCol, scalarMin,
                    scalarMax);
}

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op,
                                   std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant,
                                   KernelType kernelType, int32_t blockNum)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), mKernelType(kernelType), mBlockNum(blockNum) {
    // Resolve CPU features and kernel packing parameters for this KernelType.
    configKernel();

    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();

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

    // Prepare bias (needed by every path) and per-channel symmetric scale/zero.
    // Asymmetric paths repack their scale/zero values in a ukernel-specific layout below.
    {
        int outputCount = convOp->common()->outputCount();
        auto quanInfoPtr = quanCommon->alpha.get();
        auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
        auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
        if (!quanCommon->asymmetric) {
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

    int n = oc;
    int k = ic;
    int packedWeightSize = getRhsPackedSize(n, k, blkSize);

    // Alloc packed weight tensor.
    mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);

    if (!success) {
        MNN_ERROR("Out of static memory!\n");
        return;
    }

    size_t paraNum = scaleSize;
    float* scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    float* zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
    float* biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
    // Reload some parameters to fit ukernels' layout.
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
    } else {
        if (blkSize != 0) {
            memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
        }
    }

    // Run rhs pack.
    auto weightPackedData = mWeightInt8->host<uint8_t>();
    runRhsPack(1, n, k, blkSize, (uint8_t*)quanCommon->weight.get(), (const void*)scalePtr, (const void*)zeroPtr,
               (const void*)biasPtr, weightPackedData);
    return;
}

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend),
      mWeightInt8(exe.mWeightInt8),
      mTempIm2ColBuffer(exe.mTempIm2ColBuffer),
      mKernelType(exe.mKernelType),
      mBlockNum(exe.mBlockNum) {
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

// need
ErrorCode KleidiAIConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    auto input = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto b = backend();

    const size_t m = inputs[0]->batch() * inputs[0]->width() * inputs[0]->height(); // lhs vector number.
    const size_t n = outputs[0]->channel();                                         // rhs vector number.
    const size_t k = inputs[0]->channel();                                          // vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    if (inputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        mInputConvertBuffer.reset(
            Tensor::createDevice(std::vector<int>{input->batch(), input->height(), input->width(), input->channel()},
                                 dataType, Tensor::DimensionType::TENSORFLOW));
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

    int packedSize = getLhsQuantedPackedSize(m, k, blkSize);
    int elementSize = core->bytes;

    // Split mTempIm2ColBuffer as two parts for linear/tile transfer:
    // Part0: Lhs_packed.
    // Part1: Lhs/Dst before transfer.
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        MNN_ERROR("Out of dynamic memory!\n");
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);

    if (inputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC) {
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode KleidiAIConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

    // Initialize for convert
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    auto b = backend();
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    const size_t m = input->batch() * input->width() * input->height(); // lhs vector number.
    const size_t n = output->channel();                                 // rhs vector number.
    const size_t k = input->channel();                                  // vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    size_t elementSize = core->bytes;
    size_t lhsPackedSize = getLhsQuantedPackedSize(m, k, blkSize);

    auto lhs = input->host<uint8_t>();
    auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
    auto rhsPacked = mWeightInt8->host<uint8_t>();

    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    int threadNeed, vecPerThread;

    if (inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // Convert input to NHWC format.
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(input, mInputConvertBuffer.get(), core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        lhs = mInputConvertBuffer->host<uint8_t>();
    }

    // Dynamic quant pack lhs.
    if (m == 1) {
        runLhsQuantPack(1, k, blkSize, 1, lhs, lhsPacked);
    } else {
        vecPerThread = getVecNumPerThread(m, threadNum, getMr(m));
        threadNeed = m % vecPerThread == 0 ? m / vecPerThread : (m / vecPerThread + 1);
        size_t srcStride = vecPerThread * k * elementSize;

        auto BatchDynamicQuant = [=](int tId) {
            auto threadSrc = lhs + tId * srcStride;
            auto threadDst = lhsPacked + getLhsQuantedPackedOffset(m, tId * vecPerThread, k, blkSize);
            int vecNum = (tId == threadNeed - 1) ? (m - vecPerThread * tId)
                                                 : vecPerThread; // Last threadN may less than vecPerThread.
            runLhsQuantPack(vecNum, k, blkSize, getMr(m), threadSrc, threadDst);
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            BatchDynamicQuant((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    // Run matmul.
    auto dst = output->host<uint8_t>();
    if (outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // store matmul result to convert buffer.
        dst = mOutputConvertBuffer->host<uint8_t>();
    }

    if (bSupportSme2()) {
        // SME prefer running on single thread to obtain better performance/power consumption ratio.
        threadNum = 1;
    }

    vecPerThread = getVecNumPerThread(n, threadNum, getNStep());
    threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);
    auto postPtr = getPostParameters();

    auto ThreadFunction = [=](int tId) {
        auto threadRhsPacked = rhsPacked + getRhsPackedOffset(tId * vecPerThread, k, blkSize);
        auto threadDst = dst + getDstOffset(0, tId * vecPerThread, n, elementSize);
        int vecNum = (tId == threadNeed - 1) ? (n - vecPerThread * tId)
                                             : vecPerThread; // Last threadN may less than vecPerThread.
        runMatmul(m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst, n * elementSize, elementSize,
                  postPtr[3], postPtr[2]);
    };

    MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
        ThreadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();

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
#endif // MNN_KLEIDIAI_ENABLED
