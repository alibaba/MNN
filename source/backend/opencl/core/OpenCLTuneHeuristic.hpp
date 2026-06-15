//
//  OpenCLTuneHeuristic.hpp
//  MNN
//
//  Created by MNN on 2025/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  Heuristic rules for OpenCL kernel local size and Xgemm parameter selection,
//  derived from tuning data across multiple devices.
//  Use these when tuning cache is unavailable.
//

#ifndef MNN_OPENCL_TUNE_HEURISTIC_HPP
#define MNN_OPENCL_TUNE_HEURISTIC_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {

// Reuse GpuType { MALI=0, ADRENO=1, RADEON=2, INTEL=3, OTHER=4 }
//   and GpuLevel { UNDEFINED=0, TOP=1, MEDIUM=2, LOW=3 }
// from OpenCLRuntime.hpp directly.

// ============================================================================
// Part 1: Kernel Local Size Heuristic
// ============================================================================

/**
 * Get heuristic local work size for a given kernel and global size.
 * Matches by kernelName (the key field in tuning cache), not program name.
 *
 * @param kernelName   OpenCL kernel name (e.g. "gemv_conv_c8_buf1024_1024", "transpose_bias")
 * @param globalSize   global work size (1D/2D/3D)
 * @param gpuType      GPU type from OpenCLRuntime (ADRENO, MALI, etc.)
 * @param gpuLevel     GPU performance level from OpenCLRuntime (TOP, MEDIUM, LOW, UNDEFINED)
 * @return             recommended local work size
 */
inline std::vector<uint32_t> getHeuristicLocalSize(const std::string& kernelName,
                                                   const std::vector<uint32_t>& globalSize, GpuType gpuType = ADRENO,
                                                   GpuLevel gpuLevel = MEDIUM) {
    // Skip heuristic for unknown devices or empty globalSize
    if (gpuLevel == UNDEFINED || (gpuType != ADRENO && gpuType != MALI) || globalSize.empty()) {
        return {0, 0, 0, 0};
    }

    uint64_t totalGS = 1;
    for (auto g : globalSize) {
        if (g > 0 && totalGS > UINT64_MAX / g) {
            // Overflow detected, skip heuristic
            return {0, 0, 0, 0};
        }
        totalGS *= g;
    }

    // ---- gemm_b4_c8_int4_buf* / gemm_b4_c4_int4_buf* (2D) ----
    if (kernelName.find("gemm_b4_c8_int4_buf") == 0 || kernelName.find("gemm_b4_c4_int4_buf") == 0) {
        if (globalSize.size() < 2) {
            return {0, 0, 0, 0};
        }
        uint32_t gs0 = globalSize[0], gs1 = globalSize[1];
        if (gpuType == MALI) {
            // Mali tuning data across G76/G77/G715/G1-Ultra/G925:
            // - G925: gs1<=128→{2,16}, gs1<=256→{4,8}, gs1>256→{4,4}
            // - G715: small gs0→{8,2}/{16,1}; large gs0→varies {2,16}/{1,64}
            // - G1-Ultra: mostly {4,4} or small variations
            // - G77: tends toward larger lws[1] (16-64)
            // - G76: small gs0→{4,32}/{4,64}; large gs0→{2,4}
            if (gpuLevel == TOP) {
                // G925/G715/G1-Ultra: workgroup size 16-32
                if (gs0 <= 16)
                    return {8, 2};
                if (gs1 <= 128)
                    return {2, 16};
                if (gs1 <= 256)
                    return {4, 8};
                return {4, 4};
            } else if (gpuLevel == MEDIUM) {
                // G77: tends toward larger lws[1]
                if (gs0 <= 16)
                    return {4, 64};
                if (gs0 <= 64)
                    return {2, 32};
                return {2, 16};
            } else {
                // LOW (G76): small gs0→large lws[1]; large gs0→small lws[1]
                if (gs0 <= 16)
                    return {4, 32};
                if (gs0 <= 64)
                    return {2, 8};
                return {2, 4};
            }
        }
        // Adreno path (8gen3 Adreno 750 + 8gen5 Adreno 830/840 tuning data)
        if (gs0 <= 8) {
            return {1, 64};
        }
        if (gs0 <= 16) {
            return {4, 64};
        }
        // gs0 > 16: behavior depends on gs1 range
        if (gs1 >= 256) {
            // Large gs1: prefer parallelism in gs1 dimension
            // 8gen5: gs0=128/256,gs1=608 → {2,128}; gs0=134,gs1=608 → {1,128}
            return {2, 128};
        }
        if (gs1 <= 16) {
            // Small gs1: prefer parallelism in gs0 dimension
            // 8gen3/8gen5: gs0=134,gs1=16 → {16,4}; gs0=64,gs1=16 → {4,16}
            if (gs0 >= 128)
                return {16, 4};
            return {4, 16};
        }
        // Medium gs1 (17-255):
        // gs0<256: {4,32} consistent across 8gen3/8gen5 for gs0=32..134
        // gs0>=256: {4,64} consistent across 8gen3/8gen5 for gs0=256..512
        if (gs0 < 256) {
            return {4, 32};
        }
        return {4, 64};
    }

    // ---- gemm_b4_c8_int8_buf* (2D) ----
    if (kernelName.find("gemm_b4_c8_int8_buf") == 0) {
        if (globalSize.size() < 2) {
            return {0, 0, 0, 0};
        }
        uint32_t gs0 = globalSize[0], gs1 = globalSize[1];
        if (gpuType == MALI) {
            if (gs1 <= 128)
                return {2, 16};
            return {4, 4};
        }
        // Adreno: similar to int4
        if (gs0 <= 8)
            return {1, 64};
        if (gs0 <= 16)
            return {4, 64};
        if (gs0 <= 128)
            return {4, 32};
        return {4, 64};
    }

    // Note: matmul_qk_div_mask_prefill and matmul_qkv_prefill use XgemmBatched kernel
    // in prefill phase, whose local size = {MDIMC, NDIMC} is determined by
    // getHeuristicXgemmParams(), not by this function.

    return {0, 0, 0, 0}; // Use OpenCL runtime default
}

// ============================================================================
// Part 2: Xgemm Parameter Heuristic
// ============================================================================

/**
 * Get heuristic Xgemm parameters without tuning.
 *
 * @param M          Matrix M dimension (must be %16==0)
 * @param N          Matrix N dimension (must be %16==0)
 * @param K          Matrix K dimension (must be %4==0)
 * @param batch      Batch count (1 for Xgemm, >1 for XgemmBatched)
 * @param gpuType    GPU type from OpenCLRuntime
 * @param gpuLevel   GPU performance level from OpenCLRuntime
 * @return           14-element param_info vector, or empty if no recommendation
 */
inline std::vector<uint32_t> getHeuristicXgemmParams(uint32_t M, uint32_t N, uint32_t K, uint32_t batch = 1,
                                                     GpuType gpuType = ADRENO, GpuLevel gpuLevel = MEDIUM) {
    // Skip heuristic for unknown devices — return empty to signal no recommendation
    if (gpuLevel == UNDEFINED || (gpuType != ADRENO && gpuType != MALI)) {
        return {};
    }

    const uint32_t KWG = 16, KWI = 2, SA = 0, SB = 0;

    auto findValidTile = [](uint32_t dim, uint32_t preferred) -> uint32_t {
        if (dim % preferred == 0)
            return preferred;
        for (uint32_t t : {64u, 32u, 16u}) {
            if (dim % t == 0)
                return t;
        }
        return 16;
    };

    uint32_t MDIMA, MDIMC, MWG, NDIMB, NDIMC, NWG, STRM, STRN, VWM, VWN;

    // ============================================================
    // Mali path
    // ============================================================
    if (gpuType == MALI) {
        bool isLarge = (M >= 256 && N >= 896);

        if (gpuLevel == TOP) {
            if (isLarge) {
                MWG = findValidTile(M, 128);
                MDIMA = MDIMC = 16;
                VWM = (MWG >= 128) ? 8 : 4;
                // G925: N>=3072 prefers NWG=128/NDIMC=16; N=2048 still uses NWG=64
                if (N >= 3072) {
                    NWG = findValidTile(N, 128);
                    NDIMB = NDIMC = (NWG >= 128) ? 16 : 8;
                } else {
                    NWG = findValidTile(N, 64);
                    NDIMB = NDIMC = 8;
                }
                VWN = (NWG >= 64) ? 8 : 4;
            } else if (N <= 128 && M >= 256) {
                MWG = findValidTile(M, 32);
                NWG = findValidTile(N, 128);
                MDIMA = MDIMC = 4;
                NDIMB = NDIMC = 8;
                VWM = 8;
                VWN = NWG / NDIMC;
            } else {
                MWG = findValidTile(M, 32);
                NWG = findValidTile(N, 32);
                MDIMA = MDIMC = 4;
                NDIMB = NDIMC = 8;
                VWM = MWG / MDIMC;
                VWN = NWG / NDIMC;
            }
        } else if (gpuLevel == MEDIUM) {
            if (isLarge) {
                MWG = findValidTile(M, 128);
                NWG = findValidTile(N, 64);
                MDIMA = MDIMC = 16;
                NDIMB = NDIMC = 8;
                VWM = 2;
                VWN = 8;
            } else {
                MWG = 16;
                NWG = findValidTile(N, 64);
                MDIMA = MDIMC = (M >= 128) ? 8 : 4;
                NDIMB = NDIMC = 8;
                VWM = 2;
                VWN = NWG / NDIMC;
            }
        } else {
            if (M >= 256 && N >= 896) {
                MWG = findValidTile(M, 64);
                NWG = findValidTile(N, 32);
                MDIMA = MDIMC = 16;
                NDIMB = NDIMC = 8;
                VWM = MWG / MDIMC;
                VWN = NWG / NDIMC;
            } else {
                MWG = 16;
                NWG = findValidTile(N, 64);
                MDIMA = MDIMC = (M <= 32) ? 8 : 4;
                NDIMB = NDIMC = 8;
                VWM = 2;
                VWN = NWG / NDIMC;
            }
        }
        STRM = 0;
        STRN = 0;

        if (batch > 1) {
            // XgemmBatched: used by attention matmul_qk_div_mask_prefill / matmul_qkv_prefill
            // Consistent pattern across G76/G77/G715/G1-Ultra/G925:
            // - QK^T (K=head_dim small, M=N=seq_len large): small tiles {4,4,32, 8,8,32, VWM=8, VWN=2}
            // - QKV  (N=head_dim small, K=seq_len large): large tiles {16,16,128, 8,8,64, VWM=8, VWN=8}
            // - Small M/N (M<=32, N<=64): {8,8,16, 8,8,32, VWM=2, VWN=4}
            if (M <= 32 && N <= 64) {
                MWG = 16;
                NWG = findValidTile(N, 32);
                MDIMA = MDIMC = 8;
                NDIMB = NDIMC = 8;
                VWM = 2;
                VWN = NWG / NDIMC;
            } else if (K <= 128 && M >= 256 && N >= 256) {
                // Attention QK^T: K=head_dim (64/128), M=N=seq_len
                // All Mali GPUs consistently use small tiles for this case
                MWG = 32;
                NWG = 32;
                MDIMA = MDIMC = 4;
                NDIMB = NDIMC = 8;
                VWM = 8;
                VWN = 2;
            } else if (M >= 256) {
                // Attention QKV or large M: use large tiles
                MWG = findValidTile(M, 128);
                NWG = findValidTile(N, 64);
                MDIMA = MDIMC = 16;
                NDIMB = NDIMC = 8;
                VWM = 8;
                VWN = 8;
            } else {
                // Medium M (64-255)
                MWG = findValidTile(M, 64);
                NWG = findValidTile(N, 32);
                MDIMA = MDIMC = 16;
                NDIMB = NDIMC = 8;
                VWM = std::min(MWG / MDIMC, 4u);
                VWN = 4;
            }
        }

        if (M % MWG != 0)
            MWG = findValidTile(M, 16);
        if (N % NWG != 0)
            NWG = findValidTile(N, 16);
        VWM = std::min(VWM, MWG / MDIMC);
        VWN = std::min(VWN, NWG / NDIMC);
        while (MWG % (MDIMC * VWM) != 0 && VWM > 1)
            VWM /= 2;
        while (NWG % (NDIMC * VWN) != 0 && VWN > 1)
            VWN /= 2;

        return {KWG, KWI, MDIMA, MDIMC, MWG, NDIMB, NDIMC, NWG, SA, SB, STRM, STRN, VWM, VWN};
    }

    // ============================================================
    // Adreno path
    // ============================================================
    if (batch == 1) {
        if (M <= 64) {
            MWG = 16;
            NWG = (N >= 4096) ? findValidTile(N, 128) : findValidTile(N, 64);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            VWM = 2;
            VWN = std::min(NWG / NDIMC, 8u);
            STRM = 0;
            STRN = 0;
        } else if (N <= 128) {
            MWG = 16;
            NWG = findValidTile(N, 128);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            VWM = 2;
            VWN = std::min(NWG / NDIMC, 8u);
            STRM = 0;
            STRN = 0;
            if (M >= 2048) {
                MWG = findValidTile(M, 64);
                MDIMA = MDIMC = 16;
                VWM = std::min(MWG / MDIMC, 4u);
            }
        } else if (M >= 1024 && N >= 896) {
            MWG = findValidTile(M, 128);
            NWG = findValidTile(N, 64);
            MDIMA = MDIMC = 16;
            NDIMB = NDIMC = 8;
            VWM = std::min(MWG / MDIMC, 8u);
            VWN = std::min(NWG / NDIMC, 8u);
            if (gpuLevel == LOW)
                VWM = std::min(VWM, 4u);
            if (gpuLevel == MEDIUM && M >= 1024 && N >= 1024) {
                uint32_t nwg128 = findValidTile(N, 128);
                if (nwg128 == 128) {
                    NWG = 128;
                    NDIMB = NDIMC = 16;
                    VWN = std::min(NWG / NDIMC, 8u);
                }
            }
            STRM = 1;
            STRN = 0;
        } else if (M >= 256 && N >= 896) {
            MWG = findValidTile(M, 64);
            MDIMA = MDIMC = 16;
            NDIMB = NDIMC = 8;
            if (gpuLevel == MEDIUM) {
                // Adreno 730 (8gen1): consistently prefers NWG=128 for N>=896
                NWG = findValidTile(N, 128);
                VWM = std::min(MWG / MDIMC, 4u);
                VWN = std::min(NWG / NDIMC, 8u);
            } else if (K >= 2048) {
                MWG = findValidTile(M, 128);
                NWG = findValidTile(N, 64);
                VWM = std::min(MWG / MDIMC, 8u);
                VWN = std::min(NWG / NDIMC, 8u);
                if (gpuLevel == LOW)
                    VWM = std::min(VWM, 4u);
            } else {
                NWG = findValidTile(N, 128);
                VWM = std::min(MWG / MDIMC, 4u);
                VWN = std::min(NWG / NDIMC, 8u);
            }
            STRM = 0;
            STRN = 0;
        } else if (M >= 128 && N >= 896) {
            MWG = 16;
            NWG = findValidTile(N, 128);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            VWM = 2;
            VWN = std::min(NWG / NDIMC, 8u);
            STRM = 0;
            STRN = 0;
        } else {
            MWG = 16;
            NWG = findValidTile(N, 64);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            VWM = 2;
            VWN = std::min(NWG / NDIMC, 8u);
            STRM = 0;
            STRN = 0;
        }
    } else {
        // ============================================================
        // XgemmBatched path (batch > 1)
        // Used by matmul_qk_div_mask_prefill / matmul_qkv_prefill in attention.
        // Tuning data shows consistent patterns across Adreno 750/730/840.
        // ============================================================
        MDIMA = MDIMC = 16;
        STRM = 0;
        STRN = 0;

        if (M <= 32 && N <= 64) {
            // Very small: use smaller tiles
            MWG = findValidTile(M, 32);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            NWG = findValidTile(N, 32);
            VWM = 2;
            VWN = std::min(NWG / NDIMC, 4u);
        } else if (K <= 128 && M >= 128 && N >= 128) {
            // Attention QK^T: K=head_dim (64/128), M=N=seq_len
            // 8gen5 [544,544,64,14] → {8,8,32, 8,8,32, VWM=2, VWN=4, STRM=1}
            // 8gen3 [544,544,64,14] → {4,4,32, 8,8,32, VWM=8, VWN=2}
            // Both use MWG=32, NWG=32; use 8gen5 pattern (larger workgroup=64)
            MWG = findValidTile(M, 32);
            NWG = findValidTile(N, 32);
            MDIMA = MDIMC = 8;
            NDIMB = NDIMC = 8;
            VWM = 2;
            VWN = 4;
            STRM = 1;
        } else if (M >= 128) {
            // Main XgemmBatched path: M >= 128
            MWG = findValidTile(M, 128);

            if (N % 64 != 0 && N % 32 == 0) {
                // N not divisible by 64 (e.g. N=96): use NWG=32
                NWG = 32;
                NDIMB = NDIMC = 8;
                VWN = 4;
            } else if (N <= 64) {
                // Small N: NWG=64
                NWG = findValidTile(N, 64);
                NDIMB = NDIMC = 8;
                VWN = 8;
            } else {
                // N >= 128: NWG=64 on TOP, NWG=128 on MEDIUM
                if (gpuLevel == TOP) {
                    NWG = findValidTile(N, 64);
                    NDIMB = NDIMC = 8;
                    VWN = 8;
                } else {
                    NWG = findValidTile(N, 128);
                    NDIMB = NDIMC = (NWG >= 128) ? 16 : 8;
                    VWN = 8;
                }
            }

            // VWM: 4 on TOP tier (750/840), 8 on MEDIUM tier (730)
            if (gpuLevel == TOP) {
                VWM = std::min(MWG / MDIMC, 4u);
            } else {
                VWM = std::min(MWG / MDIMC, 8u);
            }
            STRM = 1;

            // Small M=128 with small K and small batch: use smaller tiles
            if (M <= 128 && K <= 80 && batch <= 14) {
                MWG = findValidTile(M, 64);
                NWG = findValidTile(N, 32);
                NDIMB = NDIMC = 8;
                VWM = std::min(MWG / MDIMC, 4u);
                VWN = 4;
                STRM = 0;
            }
        } else {
            // M = 64: intermediate
            MWG = findValidTile(M, 64);
            NWG = findValidTile(N, 64);
            NDIMB = NDIMC = 8;
            VWM = std::min(MWG / MDIMC, 4u);
            VWN = std::min(NWG / NDIMC, 8u);
        }
    }

    if (M % MWG != 0)
        MWG = findValidTile(M, 16);
    if (N % NWG != 0)
        NWG = findValidTile(N, 16);
    while (MWG % (MDIMC * VWM) != 0 && VWM > 1)
        VWM /= 2;
    while (NWG % (NDIMC * VWN) != 0 && VWN > 1)
        VWN /= 2;
    while (MWG % (MDIMA * VWM) != 0 && VWM > 1)
        VWM /= 2;
    while (NWG % (NDIMB * VWN) != 0 && VWN > 1)
        VWN /= 2;

    return {KWG, KWI, MDIMA, MDIMC, MWG, NDIMB, NDIMC, NWG, SA, SB, STRM, STRN, VWM, VWN};
}

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_TUNE_HEURISTIC_HPP