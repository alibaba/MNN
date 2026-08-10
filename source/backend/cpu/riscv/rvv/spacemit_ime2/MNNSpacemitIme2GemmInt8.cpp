//
//  MNNSpacemitIme2GemmInt8.cpp
//  MNN
//
//  Created by MNN on 2026/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <riscv_vector.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include "backend/cpu/compute/Int8FunctionsOpt.h"

#if defined(MNN_USE_SPACEMIT_IME2)
extern "C" size_t MNNSpacemitIme2GemmI8I4Local(size_t blkLen, const uint8_t* a, const uint8_t* b, const uint8_t* bZp,
                                               float* c, size_t countM, size_t countN, size_t kBlocks, size_t ldc);
extern "C" size_t MNNSpacemitIme2GemmI8I4HpM1NativeLocal(size_t blkLen, const uint8_t* a, const uint8_t* b,
                                                         const uint8_t* bZp, float* c, size_t countM, size_t countN,
                                                         size_t kBlocks, size_t ldc);
extern "C" size_t MNNSpacemitIme2GemmI8I4HpM4DirectC4Local(size_t blkLen, const uint8_t* a, const uint8_t* b,
                                                           const uint8_t* bZp, int8_t* dst, size_t dstStep,
                                                           size_t countM, size_t countN, size_t kBlocks,
                                                           const float* inputScale, const float* bias, float fp32Min,
                                                           float fp32Max, int needClamp);
#endif

static bool MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPost_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                                 size_t srcDepthQuad, size_t dst_step,
                                                                 size_t dst_depth_quad,
                                                                 const QuanPostTreatParameters* post);
static bool MNNGemmInt8AddBiasScale_16x4_w4_BatchS4FastPost_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                                size_t srcDepthQuad, size_t dst_step,
                                                                size_t dst_depth_quad,
                                                                const QuanPostTreatParameters* post, size_t realCount);
extern "C" int MNNSpacemitIme2PackFloatAHpStridedRowsWithSum(uint8_t* dst, float* srcKernelSum, const float* inputScale,
                                                             const float* quantScale, const float* src,
                                                             size_t srcDepthQuad, size_t blockNum, size_t srcRows,
                                                             size_t rowBegin, size_t realCount);

namespace {

using MNNSpacemitIme2GemmI8I4 = size_t (*)(size_t, const uint8_t*, const uint8_t*, const uint8_t*, float*, size_t,
                                           size_t, size_t, size_t);
static constexpr size_t MNN_IME2_Q8_BLOCK_BYTES = sizeof(float) + sizeof(int16_t) + 32;

static thread_local bool gMNNSpacemitIme2BlockInputScale = false;
static thread_local bool gMNNSpacemitIme2PackedAInput = false;

static constexpr bool MNNSpacemitIme2Enabled() {
#if defined(MNN_USE_SPACEMIT_IME2)
    return true;
#else
    return false;
#endif
}

static constexpr bool MNNSpacemitIme2HpM1Enabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2HpM1CenteredEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2FoldBlockScaleEnabled() {
    return true;
}

static inline bool MNNSpacemitIme2BlockInputScaleEnabled() {
    return gMNNSpacemitIme2BlockInputScale;
}

static inline bool MNNSpacemitIme2PackedAInputEnabled() {
    return gMNNSpacemitIme2PackedAInput;
}

static constexpr bool MNNSpacemitIme2TiledAEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2ExecutorPackAEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2HpTiledPackEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2DecodeTailEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2FusedPostEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2FusedPostMEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2FusedScaleRowEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2PostMRowNEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2SkipResidualEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2ZeroResidualSkipEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2PostMSeg4Enabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2W4A4Enabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2W4A4DynamicEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2FixedAScaleEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2SymW4Enabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2FuseResidualEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2ACacheEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2ACacheFastEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2PrepackAEnabled() {
    return false;
}

static inline bool MNNSpacemitIme2UseACacheEnabled() {
    return MNNSpacemitIme2ACacheEnabled() || MNNSpacemitIme2PrepackAEnabled();
}

static constexpr bool MNNSpacemitIme2TcmEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2TcmDecodePipelineEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2LinearSpinEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2DirectC4EpilogueEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2TcmTaskSpinEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2HpM1AsymPairEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2OuterTileSingleWorkerEnabled() {
    return false;
}

static thread_local bool gMNNSpacemitIme2OuterTileParallel = false;

extern "C" void MNNSpacemitIme2SetOuterTileParallel(int enabled) {
    gMNNSpacemitIme2OuterTileParallel = enabled != 0;
}

extern "C" void MNNSpacemitIme2SetPackedAInput(int enabled) {
    gMNNSpacemitIme2PackedAInput = enabled != 0;
}

static inline bool MNNSpacemitIme2OuterTileParallel() {
    return gMNNSpacemitIme2OuterTileParallel;
}

static inline bool MNNSpacemitIme2IsHpBlkLen(size_t blkLen) {
    return blkLen == 256 || blkLen == 257 || blkLen == 258 || blkLen == 259 || blkLen == 260 || blkLen == 261;
}

static constexpr bool MNNSpacemitIme2BindTidEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2AffinityEnabled() {
    return true;
}

static constexpr size_t MNNSpacemitIme2TileRows() {
    return 8;
}

static constexpr bool MNNSpacemitIme2ZpEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2HpEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2HpM4Enabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2WorkerPostEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2DecodeWorkerPostEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2DecodeDirectOutputEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2DecodeBiasPostEnabled() {
    return true;
}

static inline bool MNNSpacemitIme2UseDecodeBiasPost(const QuanPostTreatParameters* post) {
    // DenseConvInt8TiledExecutor uses an indices self pointer as a private marker. inputBias must stay null:
    // the regular inputBias layout is [block, row], while this decode correction is one scalar for a
    // weightKernelSum that is already reduced across all blocks.
    return post->indices == reinterpret_cast<const int32_t*>(post) && post->scale != nullptr &&
           post->inputBias == nullptr && post->inputScale != nullptr && post->weightKernelSum != nullptr &&
           post->biasFloat != nullptr && MNNSpacemitIme2DecodeBiasPostEnabled();
}

static constexpr bool MNNSpacemitIme2WorkerSplitMEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2Worker2DEnabled() {
    return false;
}

static constexpr int MNNSpacemitIme2DirectOuterMode() {
    return 0;
}

static constexpr bool MNNSpacemitIme2DirectOuterBindEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2SpinEnabled() {
    return true;
}

static constexpr bool MNNSpacemitIme2SerialDispatchEnabled() {
    return false;
}

static constexpr bool MNNRvvW4DecodeDz2Enabled() {
    return true;
}

static constexpr bool MNNRvvW4DecodePrepackEnabled() {
    return false;
}

static constexpr bool MNNRvvW4CenteredPostEnabled() {
    return false;
}

static constexpr bool MNNRvvW4BatchDz2Enabled() {
    return false;
}

static inline bool MNNRvvFp32MinMaxIsFullRange(float minValue, float maxValue);

static inline MNNSpacemitIme2GemmI8I4 MNNSpacemitIme2Gemm() {
#if defined(MNN_USE_SPACEMIT_IME2)
    return MNNSpacemitIme2GemmI8I4Local;
#else
    return nullptr;
#endif
}

static inline void MNNSpacemitSetCurrentThreadAI(bool forceTid = false) {
    FILE* file = std::fopen("/proc/set_ai_thread", "w");
    if (file == nullptr) {
        return;
    }
    if (forceTid || MNNSpacemitIme2BindTidEnabled()) {
        std::fprintf(file, "%ld", static_cast<long>(syscall(SYS_gettid)));
    } else {
        std::fprintf(file, "0");
    }
    std::fclose(file);
}

static inline void MNNSpacemitIme2BindWorker(size_t index) {
    if (MNNSpacemitIme2BindTidEnabled()) {
        MNNSpacemitSetCurrentThreadAI(true);
    } else {
        MNNSpacemitSetCurrentThreadAI();
    }
    if (MNNSpacemitIme2AffinityEnabled()) {
        constexpr size_t cpuBase = 8;
        constexpr size_t cpuCount = 8;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(cpuBase + index % std::max<size_t>(cpuCount, 1)), &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
}

static inline void MNNSpacemitIme2BindCurrentThreadOnce() {
    static std::atomic<size_t> nextIndex(0);
    thread_local bool bound = false;
    if (bound) {
        return;
    }
    const size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
    MNNSpacemitIme2BindWorker(index);
    bound = true;
}

static inline void MNNSpacemitIme2ClearWorker(size_t index) {
    (void)index;
}

static inline void MNNSpacemitIme2Relax() {
#if defined(__riscv)
    asm volatile("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

static inline std::mutex& MNNSpacemitIme2DispatchMutex() {
    static std::mutex mutex;
    return mutex;
}

static inline void MNNSpacemitIme2PostChunkFused(int8_t* dst, size_t dstStep, const float* c, size_t countN,
                                                 size_t dzStart, size_t fullCountN, const float* residual,
                                                 const QuanPostTreatParameters* post, const float* biasPtr,
                                                 float fp32min, float fp32max, bool scaleCByInput, bool skipResidual) {
    const size_t nOffset = dzStart * GEMM_INT8_UNIT;
    const bool decodeBiasPost = MNNSpacemitIme2UseDecodeBiasPost(post);
    const float decodeBiasCorrection = decodeBiasPost ? post->scale[0] : 0.0f;
    const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const float inputScale = scaleCByInput && post->inputScale != nullptr ? post->inputScale[0] : 1.0f;
    if (dstStep == GEMM_INT8_UNIT * sizeof(float)) {
        float* dstBase = reinterpret_cast<float*>(dst + dzStart * dstStep);
        size_t x = 0;
        while (x < countN) {
            const size_t vl = __riscv_vsetvl_e32m4(countN - x);
            vfloat32m4_t value = __riscv_vle32_v_f32m4(c + x, vl);
            if (scaleCByInput && post->inputScale != nullptr) {
                value = __riscv_vfmul_vf_f32m4(value, inputScale, vl);
            }
            if (!skipResidual) {
                for (size_t bk = 0; bk < post->blockNum; ++bk) {
                    const float srcSum = post->srcKernelSum[bk];
                    const float* residualBk = residual + bk * fullCountN + nOffset;
                    value = __riscv_vfmacc_vf_f32m4(value, srcSum, __riscv_vle32_v_f32m4(residualBk + x, vl), vl);
                }
            }
            if (biasPtr != nullptr) {
                vfloat32m4_t fusedBias = __riscv_vle32_v_f32m4(biasPtr + nOffset + x, vl);
                if (decodeBiasPost) {
                    fusedBias =
                        __riscv_vfmacc_vf_f32m4(fusedBias, decodeBiasCorrection,
                                                __riscv_vle32_v_f32m4(post->weightKernelSum + nOffset + x, vl), vl);
                }
                value = __riscv_vfadd_vv_f32m4(value, fusedBias, vl);
            }
            if (needClamp) {
                value = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(value, fp32min, vl), fp32max, vl);
            }
            __riscv_vse32_v_f32m4(dstBase + x, value, vl);
            x += vl;
        }
        return;
    }

    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const size_t localDstDepthQuad = countN / GEMM_INT8_UNIT;
    for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
        const size_t dz = dzStart + dzLocal;
        vfloat32m1_t value = __riscv_vle32_v_f32m1(c + dzLocal * GEMM_INT8_UNIT, vl4);
        if (scaleCByInput && post->inputScale != nullptr) {
            value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
        }
        if (!skipResidual) {
            for (size_t bk = 0; bk < post->blockNum; ++bk) {
                const float srcSum = post->srcKernelSum[bk];
                value = __riscv_vfmacc_vf_f32m1(
                    value, srcSum, __riscv_vle32_v_f32m1(residual + bk * fullCountN + dz * GEMM_INT8_UNIT, vl4), vl4);
            }
        }
        if (biasPtr != nullptr) {
            vfloat32m1_t fusedBias = __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4);
            if (decodeBiasPost) {
                fusedBias = __riscv_vfmacc_vf_f32m1(
                    fusedBias, decodeBiasCorrection,
                    __riscv_vle32_v_f32m1(post->weightKernelSum + dz * GEMM_INT8_UNIT, vl4), vl4);
            }
            value = __riscv_vfadd_vv_f32m1(value, fusedBias, vl4);
        }
        if (needClamp) {
            value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
        }
        float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep);
        __riscv_vse32_v_f32m1(dstRow, value, vl4);
    }
}

static inline void MNNSpacemitIme2PostChunk(int8_t* dst, size_t dstStep, float* c, size_t countN, size_t dzStart,
                                            size_t fullCountN, const float* residual,
                                            const QuanPostTreatParameters* post, const float* biasPtr, float fp32min,
                                            float fp32max, bool scaleCByInput = false,
                                            bool skipResidualOverride = false) {
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const size_t localDstDepthQuad = countN / GEMM_INT8_UNIT;
    const size_t nOffset = dzStart * GEMM_INT8_UNIT;
    const bool decodeBiasPost = MNNSpacemitIme2UseDecodeBiasPost(post);
    const float decodeBiasCorrection = decodeBiasPost ? post->scale[0] : 0.0f;
    const float inputScale = scaleCByInput && post->inputScale != nullptr ? post->inputScale[0] : 1.0f;
    if (post->inputBias == nullptr) {
        const bool skipResidual = skipResidualOverride || MNNSpacemitIme2SkipResidualEnabled();
        if (MNNSpacemitIme2FusedPostEnabled()) {
            MNNSpacemitIme2PostChunkFused(dst, dstStep, c, countN, dzStart, fullCountN, residual, post, biasPtr,
                                          fp32min, fp32max, scaleCByInput, skipResidual);
            return;
        }
        if (scaleCByInput && post->inputScale != nullptr) {
            size_t x = 0;
            while (x < countN) {
                const size_t vl = __riscv_vsetvl_e32m4(countN - x);
                vfloat32m4_t value = __riscv_vle32_v_f32m4(c + x, vl);
                value = __riscv_vfmul_vf_f32m4(value, inputScale, vl);
                __riscv_vse32_v_f32m4(c + x, value, vl);
                x += vl;
            }
        }
        if (!skipResidual) {
            for (size_t bk = 0; bk < post->blockNum; ++bk) {
                const float srcSum = post->srcKernelSum[bk];
                const float* residualBk = residual + bk * fullCountN + nOffset;
                size_t x = 0;
                while (x < countN) {
                    const size_t vl = __riscv_vsetvl_e32m4(countN - x);
                    vfloat32m4_t value = __riscv_vle32_v_f32m4(c + x, vl);
                    value = __riscv_vfmacc_vf_f32m4(value, srcSum, __riscv_vle32_v_f32m4(residualBk + x, vl), vl);
                    __riscv_vse32_v_f32m4(c + x, value, vl);
                    x += vl;
                }
            }
        }
        const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
        size_t x = 0;
        while (x < countN) {
            const size_t vl = __riscv_vsetvl_e32m4(countN - x);
            vfloat32m4_t value = __riscv_vle32_v_f32m4(c + x, vl);
            if (biasPtr != nullptr) {
                vfloat32m4_t fusedBias = __riscv_vle32_v_f32m4(biasPtr + nOffset + x, vl);
                if (decodeBiasPost) {
                    fusedBias =
                        __riscv_vfmacc_vf_f32m4(fusedBias, decodeBiasCorrection,
                                                __riscv_vle32_v_f32m4(post->weightKernelSum + nOffset + x, vl), vl);
                }
                value = __riscv_vfadd_vv_f32m4(value, fusedBias, vl);
            }
            if (needClamp) {
                value = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(value, fp32min, vl), fp32max, vl);
            }
            __riscv_vse32_v_f32m4(c + x, value, vl);
            x += vl;
        }
        if (dstStep == GEMM_INT8_UNIT * sizeof(float)) {
            std::memcpy(dst + dzStart * dstStep, c, countN * sizeof(float));
            return;
        }
        for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
            const size_t dz = dzStart + dzLocal;
            float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep);
            __riscv_vse32_v_f32m1(dstRow, __riscv_vle32_v_f32m1(c + dzLocal * GEMM_INT8_UNIT, vl4), vl4);
        }
        return;
    }
    for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
        const size_t dz = dzStart + dzLocal;
        vfloat32m1_t value = __riscv_vle32_v_f32m1(c + dzLocal * GEMM_INT8_UNIT, vl4);
        if (scaleCByInput) {
            value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
        }
        for (size_t bk = 0; bk < post->blockNum; ++bk) {
            value = __riscv_vfmacc_vf_f32m1(
                value, post->srcKernelSum[bk],
                __riscv_vle32_v_f32m1(residual + bk * fullCountN + dz * GEMM_INT8_UNIT, vl4), vl4);
            if (post->inputBias != nullptr && post->weightKernelSum != nullptr) {
                const float* weightKernelSum =
                    post->weightKernelSum + dz * (post->blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;
                value = __riscv_vfmacc_vf_f32m1(value, post->inputBias[bk], __riscv_vle32_v_f32m1(weightKernelSum, vl4),
                                                vl4);
            }
        }
        if (biasPtr != nullptr) {
            value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
        }
        if (post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max)) {
            value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
        }
        float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep);
        __riscv_vse32_v_f32m1(dstRow, value, vl4);
    }
}

static inline float MNNSpacemitIme2ReadSrcKernelSumM(const QuanPostTreatParameters* post, size_t bk, size_t row,
                                                     size_t countM, bool tiledA, size_t tileRows) {
    if (!tiledA) {
        return post->srcKernelSum[bk * countM + row];
    }
    const size_t tile = row / tileRows;
    const size_t localRow = row - tile * tileRows;
    const size_t rowsInTile = std::min(tileRows, countM - tile * tileRows);
    return post->srcKernelSum[tile * tileRows * post->blockNum + bk * rowsInTile + localRow];
}

static inline float MNNSpacemitIme2ReadSrcKernelSumMDirect(const QuanPostTreatParameters* post, size_t bk, size_t row,
                                                           size_t countM, bool tiledA, size_t tileRows,
                                                           const float* directSrcKernelSum,
                                                           size_t directSrcKernelSumStride, size_t directSrcRowOffset) {
    if (directSrcKernelSum != nullptr) {
        return directSrcKernelSum[bk * directSrcKernelSumStride + directSrcRowOffset + row];
    }
    return MNNSpacemitIme2ReadSrcKernelSumM(post, bk, row, countM, tiledA, tileRows);
}

static inline void MNNSpacemitIme2PostChunkMStoreSeg4(int8_t* dst, size_t dstStep, const float* c, size_t countM,
                                                      size_t countN, size_t dzStart, size_t fullCountN,
                                                      const float* biasPtr, float fp32min, float fp32max,
                                                      bool needClamp) {
    const size_t localDstDepthQuad = countN / GEMM_INT8_UNIT;
    const long cStrideBytes = static_cast<long>(fullCountN * sizeof(float));
    for (size_t rowBase = 0; rowBase < countM;) {
        const size_t vl = __riscv_vsetvl_e32m1(countM - rowBase);
        for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
            const size_t dz = dzStart + dzLocal;
            const float* cBase = c + rowBase * fullCountN + dzLocal * GEMM_INT8_UNIT;
            vfloat32m1_t v0 = __riscv_vlse32_v_f32m1(cBase + 0, cStrideBytes, vl);
            vfloat32m1_t v1 = __riscv_vlse32_v_f32m1(cBase + 1, cStrideBytes, vl);
            vfloat32m1_t v2 = __riscv_vlse32_v_f32m1(cBase + 2, cStrideBytes, vl);
            vfloat32m1_t v3 = __riscv_vlse32_v_f32m1(cBase + 3, cStrideBytes, vl);
            if (biasPtr != nullptr) {
                const float* bias = biasPtr + dz * GEMM_INT8_UNIT;
                v0 = __riscv_vfadd_vf_f32m1(v0, bias[0], vl);
                v1 = __riscv_vfadd_vf_f32m1(v1, bias[1], vl);
                v2 = __riscv_vfadd_vf_f32m1(v2, bias[2], vl);
                v3 = __riscv_vfadd_vf_f32m1(v3, bias[3], vl);
            }
            if (needClamp) {
                v0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(v0, fp32min, vl), fp32max, vl);
                v1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(v1, fp32min, vl), fp32max, vl);
                v2 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(v2, fp32min, vl), fp32max, vl);
                v3 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(v3, fp32min, vl), fp32max, vl);
            }
            vfloat32m1x4_t values = __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
            float* dstBase = reinterpret_cast<float*>(dst + dz * dstStep + rowBase * GEMM_INT8_UNIT * sizeof(float));
            __riscv_vsseg4e32_v_f32m1x4(dstBase, values, vl);
        }
        rowBase += vl;
    }
}

static inline void MNNSpacemitIme2PostChunkMFused(int8_t* dst, size_t dstStep, const float* c, size_t countM,
                                                  size_t countN, size_t dzStart, size_t fullCountN,
                                                  const float* residual, const QuanPostTreatParameters* post,
                                                  const float* biasPtr, float fp32min, float fp32max,
                                                  bool scaleCByInput, bool tiledA, size_t tileRows) {
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const size_t localDstDepthQuad = countN / GEMM_INT8_UNIT;
    const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    for (size_t row = 0; row < countM; ++row) {
        const float* cRow = c + row * fullCountN;
        const float inputScale = scaleCByInput && post->inputScale != nullptr ? post->inputScale[row] : 1.0f;
        for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
            const size_t dz = dzStart + dzLocal;
            vfloat32m1_t value = __riscv_vle32_v_f32m1(cRow + dzLocal * GEMM_INT8_UNIT, vl4);
            if (scaleCByInput) {
                value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
            }
            for (size_t bk = 0; bk < post->blockNum; ++bk) {
                const float srcSum = MNNSpacemitIme2ReadSrcKernelSumM(post, bk, row, countM, tiledA, tileRows);
                value = __riscv_vfmacc_vf_f32m1(
                    value, srcSum, __riscv_vle32_v_f32m1(residual + bk * fullCountN + dz * GEMM_INT8_UNIT, vl4), vl4);
            }
            if (biasPtr != nullptr) {
                value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
            }
            if (needClamp) {
                value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
            }
            float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep + row * GEMM_INT8_UNIT * sizeof(float));
            __riscv_vse32_v_f32m1(dstRow, value, vl4);
        }
    }
}

static inline void MNNSpacemitIme2PostChunkMRowN(int8_t* dst, size_t dstStep, float* c, size_t countM, size_t countN,
                                                 size_t dzStart, size_t fullCountN, const float* residual,
                                                 const QuanPostTreatParameters* post, const float* biasPtr,
                                                 float fp32min, float fp32max, bool scaleCByInput, bool tiledA,
                                                 size_t tileRows) {
    const size_t nOffset = dzStart * GEMM_INT8_UNIT;
    const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    for (size_t row = 0; row < countM; ++row) {
        float* cRow = c + row * fullCountN;
        const float inputScale = scaleCByInput && post->inputScale != nullptr ? post->inputScale[row] : 1.0f;
        size_t x = 0;
        while (x < countN) {
            const size_t vl = __riscv_vsetvl_e32m4(countN - x);
            vfloat32m4_t value = __riscv_vle32_v_f32m4(cRow + x, vl);
            if (scaleCByInput) {
                value = __riscv_vfmul_vf_f32m4(value, inputScale, vl);
            }
            for (size_t bk = 0; bk < post->blockNum; ++bk) {
                const float srcSum = MNNSpacemitIme2ReadSrcKernelSumM(post, bk, row, countM, tiledA, tileRows);
                const float* residualBk = residual + bk * fullCountN + nOffset;
                value = __riscv_vfmacc_vf_f32m4(value, srcSum, __riscv_vle32_v_f32m4(residualBk + x, vl), vl);
            }
            __riscv_vse32_v_f32m4(cRow + x, value, vl);
            x += vl;
        }
    }
    MNNSpacemitIme2PostChunkMStoreSeg4(dst, dstStep, c, countM, countN, dzStart, fullCountN, biasPtr, fp32min, fp32max,
                                       needClamp);
}

static inline void
MNNSpacemitIme2PostChunkM(int8_t* dst, size_t dstStep, float* c, size_t countM, size_t countN, size_t dzStart,
                          size_t fullCountN, const float* residual, const QuanPostTreatParameters* post,
                          const float* biasPtr, float fp32min, float fp32max, bool scaleCByInput = false,
                          bool skipResidualOverride = false, const float* directSrcKernelSum = nullptr,
                          size_t directSrcKernelSumStride = 0, size_t directSrcRowOffset = 0, size_t ime2BlkLen = 0) {
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const size_t localDstDepthQuad = countN / GEMM_INT8_UNIT;
    const size_t tileRows = MNNSpacemitIme2TileRows();
    const bool tiledA = false;
    if (post->inputBias == nullptr) {
        if (skipResidualOverride || MNNSpacemitIme2SkipResidualEnabled()) {
            const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
            const bool fusedScaleRow = skipResidualOverride && ime2BlkLen == 258 && scaleCByInput &&
                                       post->inputScale != nullptr && !MNNSpacemitIme2PostMSeg4Enabled() &&
                                       MNNSpacemitIme2FusedScaleRowEnabled();
            if (!fusedScaleRow && scaleCByInput && post->inputScale != nullptr) {
                for (size_t row = 0; row < countM; ++row) {
                    float* cRow = c + row * fullCountN;
                    size_t x = 0;
                    while (x < countN) {
                        const size_t vl = __riscv_vsetvl_e32m4(countN - x);
                        vfloat32m4_t value = __riscv_vle32_v_f32m4(cRow + x, vl);
                        value = __riscv_vfmul_vf_f32m4(value, post->inputScale[row], vl);
                        __riscv_vse32_v_f32m4(cRow + x, value, vl);
                        x += vl;
                    }
                }
            }
            if (MNNSpacemitIme2PostMSeg4Enabled()) {
                MNNSpacemitIme2PostChunkMStoreSeg4(dst, dstStep, c, countM, countN, dzStart, fullCountN, biasPtr,
                                                   fp32min, fp32max, needClamp);
            } else {
                for (size_t row = 0; row < countM; ++row) {
                    float* cRow = c + row * fullCountN;
                    const float inputScale = fusedScaleRow ? post->inputScale[row] : 1.0f;
                    for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
                        const size_t dz = dzStart + dzLocal;
                        vfloat32m1_t value = __riscv_vle32_v_f32m1(cRow + dzLocal * GEMM_INT8_UNIT, vl4);
                        if (fusedScaleRow) {
                            value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
                        }
                        if (biasPtr != nullptr) {
                            value = __riscv_vfadd_vv_f32m1(
                                value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
                        }
                        if (needClamp) {
                            value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
                        }
                        float* dstRow =
                            reinterpret_cast<float*>(dst + dz * dstStep + row * GEMM_INT8_UNIT * sizeof(float));
                        __riscv_vse32_v_f32m1(dstRow, value, vl4);
                    }
                }
            }
            return;
        }
        if (MNNSpacemitIme2PostMRowNEnabled()) {
            MNNSpacemitIme2PostChunkMRowN(dst, dstStep, c, countM, countN, dzStart, fullCountN, residual, post, biasPtr,
                                          fp32min, fp32max, scaleCByInput, tiledA, tileRows);
            return;
        }
        if (MNNSpacemitIme2FusedPostMEnabled()) {
            MNNSpacemitIme2PostChunkMFused(dst, dstStep, c, countM, countN, dzStart, fullCountN, residual, post,
                                           biasPtr, fp32min, fp32max, scaleCByInput, tiledA, tileRows);
            return;
        }
        const size_t nOffset = dzStart * GEMM_INT8_UNIT;
        const bool needClamp = post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
        for (size_t bk = 0; bk < post->blockNum; ++bk) {
            const float* residualBk = residual + bk * fullCountN + nOffset;
            size_t x = 0;
            while (x < countN) {
                const size_t vl = __riscv_vsetvl_e32m4(countN - x);
                const vfloat32m4_t residualValue = __riscv_vle32_v_f32m4(residualBk + x, vl);
                for (size_t row = 0; row < countM; ++row) {
                    float* cRow = c + row * fullCountN;
                    const float srcSum = MNNSpacemitIme2ReadSrcKernelSumMDirect(
                        post, bk, row, countM, tiledA, tileRows, directSrcKernelSum, directSrcKernelSumStride,
                        directSrcRowOffset);
                    vfloat32m4_t value = __riscv_vle32_v_f32m4(cRow + x, vl);
                    if (bk == 0 && scaleCByInput && post->inputScale != nullptr) {
                        value = __riscv_vfmul_vf_f32m4(value, post->inputScale[row], vl);
                    }
                    value = __riscv_vfmacc_vf_f32m4(value, srcSum, residualValue, vl);
                    __riscv_vse32_v_f32m4(cRow + x, value, vl);
                }
                x += vl;
            }
        }
        if (MNNSpacemitIme2PostMSeg4Enabled()) {
            MNNSpacemitIme2PostChunkMStoreSeg4(dst, dstStep, c, countM, countN, dzStart, fullCountN, biasPtr, fp32min,
                                               fp32max, needClamp);
        } else {
            for (size_t row = 0; row < countM; ++row) {
                float* cRow = c + row * fullCountN;
                for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
                    const size_t dz = dzStart + dzLocal;
                    vfloat32m1_t value = __riscv_vle32_v_f32m1(cRow + dzLocal * GEMM_INT8_UNIT, vl4);
                    if (biasPtr != nullptr) {
                        value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4),
                                                       vl4);
                    }
                    if (needClamp) {
                        value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
                    }
                    float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep + row * GEMM_INT8_UNIT * sizeof(float));
                    __riscv_vse32_v_f32m1(dstRow, value, vl4);
                }
            }
        }
        return;
    }
    for (size_t row = 0; row < countM; ++row) {
        const float* cRow = c + row * fullCountN;
        for (size_t dzLocal = 0; dzLocal < localDstDepthQuad; ++dzLocal) {
            const size_t dz = dzStart + dzLocal;
            vfloat32m1_t value = __riscv_vle32_v_f32m1(cRow + dzLocal * GEMM_INT8_UNIT, vl4);
            if (scaleCByInput && post->inputScale != nullptr) {
                const float inputScale = post->inputBias ? post->inputScale[row] : post->inputScale[row];
                value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
            }
            for (size_t bk = 0; bk < post->blockNum; ++bk) {
                value = __riscv_vfmacc_vf_f32m1(
                    value, MNNSpacemitIme2ReadSrcKernelSumM(post, bk, row, countM, tiledA, tileRows),
                    __riscv_vle32_v_f32m1(residual + bk * fullCountN + dz * GEMM_INT8_UNIT, vl4), vl4);
                if (post->inputBias != nullptr && post->weightKernelSum != nullptr) {
                    const float* weightKernelSum =
                        post->weightKernelSum + dz * (post->blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;
                    value = __riscv_vfmacc_vf_f32m1(value, post->inputBias[bk * countM + row],
                                                    __riscv_vle32_v_f32m1(weightKernelSum, vl4), vl4);
                }
            }
            if (biasPtr != nullptr) {
                value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
            }
            if (post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max)) {
                value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
            }
            float* dstRow = reinterpret_cast<float*>(dst + dz * dstStep + row * GEMM_INT8_UNIT * sizeof(float));
            __riscv_vse32_v_f32m1(dstRow, value, vl4);
        }
    }
}

using MNNSpacemitIme2TcmTask = void (*)(size_t, void*, size_t, void*);

struct MNNSpacemitIme2PairBarrier {
    void reset() {
        pending.store(2, std::memory_order_relaxed);
        round.store(0, std::memory_order_relaxed);
        tcmReady.store(true, std::memory_order_relaxed);
    }

    void arriveAndWait() {
        const uint32_t current = round.load(std::memory_order_acquire);
        if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            pending.store(2, std::memory_order_relaxed);
            round.store(current + 1, std::memory_order_release);
            return;
        }
        while (round.load(std::memory_order_relaxed) == current) {
            MNNSpacemitIme2Relax();
        }
    }

    alignas(64) std::atomic<uint32_t> pending{2};
    alignas(64) std::atomic<uint32_t> round{0};
    alignas(64) std::atomic<bool> tcmReady{true};
};

struct MNNSpacemitIme2Job {
    MNNSpacemitIme2GemmI8I4 gemm = nullptr;
    size_t blkLen = 0;
    const uint8_t* a = nullptr;
    const uint8_t* b = nullptr;
    const uint8_t* bZp = nullptr;
    float* c = nullptr;
    size_t countM = 0;
    size_t countN = 0;
    size_t kBlks = 0;
    size_t ldc = 0;
    bool doPost = false;
    int8_t* dst = nullptr;
    size_t dstStep = 0;
    size_t dzStart = 0;
    size_t fullCountN = 0;
    const float* residual = nullptr;
    const QuanPostTreatParameters* post = nullptr;
    const float* biasPtr = nullptr;
    float fp32min = 0.0f;
    float fp32max = 0.0f;
    bool skipResidual = false;
    const float* directSrcKernelSum = nullptr;
    size_t directSrcKernelSumStride = 0;
    size_t directSrcRowOffset = 0;
    bool directC4Epilogue = false;
    bool linearStride = false;
    size_t linearRowStart = 0;
    size_t linearRowEnd = 0;
    size_t linearRowStep = 0;
    size_t linearRowsPerBlock = 0;
    size_t linearTotalRows = 0;
    bool packFloatA = false;
    const float* packFloatSrc = nullptr;
    const float* packInputScale = nullptr;
    const float* packQuantScale = nullptr;
    size_t packSrcDepthQuad = 0;
    size_t packBlockNum = 0;
    size_t packSrcRows = 0;
    size_t packRowBegin = 0;
    MNNSpacemitIme2TcmTask tcmTask = nullptr;
    void* tcmTaskContext = nullptr;
    size_t tcmTaskIndex = 0;
    MNNSpacemitIme2PairBarrier* decodeTcmPairBarrier = nullptr;
    size_t decodeTcmPairRole = 0;
    size_t decodeTcmPairRounds = 0;
    size_t decodeTcmPairBStride = 0;
    size_t decodeTcmPairGroupStep = 0;
};

struct MNNSpacemitIme2TcmInfo {
    size_t blkSize = 0;
    size_t blkNum = 0;
    int isFakeTcm = 0;
};

struct MNNSpacemitIme2TcmApi {
    using IsAvailableFunc = int (*)();
    using LayoutInfoFunc = int (*)(MNNSpacemitIme2TcmInfo*);
    using MemGetFunc = void* (*)(int);
    using MemFreeFunc = int (*)(int);

    void* handle = nullptr;
    IsAvailableFunc isAvailable = nullptr;
    LayoutInfoFunc layoutInfo = nullptr;
    MemGetFunc memGet = nullptr;
    MemFreeFunc memFree = nullptr;
    MNNSpacemitIme2TcmInfo info;
    bool ready = false;
    bool available = false;

    MNNSpacemitIme2TcmApi() {
        if (!MNNSpacemitIme2TcmEnabled()) {
            return;
        }
        handle = dlopen("libspine_tcm.so", RTLD_LAZY | RTLD_GLOBAL);
        if (handle == nullptr) {
            return;
        }
        isAvailable = reinterpret_cast<IsAvailableFunc>(dlsym(handle, "spine_tcm_runtime_is_available"));
        layoutInfo = reinterpret_cast<LayoutInfoFunc>(dlsym(handle, "spine_tcm_runtime_layout_info"));
        memGet = reinterpret_cast<MemGetFunc>(dlsym(handle, "spine_tcm_runtime_mem_get"));
        memFree = reinterpret_cast<MemFreeFunc>(dlsym(handle, "spine_tcm_runtime_mem_free"));
        ready = isAvailable != nullptr && layoutInfo != nullptr && memGet != nullptr && memFree != nullptr;
        if (!ready || isAvailable() != 1 || layoutInfo(&info) != 0 || info.blkSize == 0 || info.blkNum == 0 ||
            info.isFakeTcm != 0) {
            return;
        }
        available = true;
    }
};

static inline MNNSpacemitIme2TcmApi& MNNSpacemitIme2Tcm() {
    static MNNSpacemitIme2TcmApi api;
    return api;
}

static inline void* MNNSpacemitIme2TcmAcquire(size_t index, size_t* size) {
    if (size != nullptr) {
        *size = 0;
    }
    auto& api = MNNSpacemitIme2Tcm();
    if (!api.available || index >= api.info.blkNum) {
        return nullptr;
    }
    const int tcmId = static_cast<int>(index);
    void* buffer = api.memGet(tcmId);
    if (buffer == nullptr) {
        return nullptr;
    }
    if (size != nullptr) {
        *size = api.info.blkSize;
    }
    return buffer;
}

static inline void MNNSpacemitIme2TcmRelease(size_t index, void* buffer) {
    if (buffer == nullptr) {
        return;
    }
    auto& api = MNNSpacemitIme2Tcm();
    if (!api.available || index >= api.info.blkNum) {
        return;
    }
    const int tcmId = static_cast<int>(index);
    api.memFree(tcmId);
}

static inline size_t MNNSpacemitIme2BStrideForJob(const MNNSpacemitIme2Job& job) {
    const bool useZp = job.bZp != nullptr;
    if (job.blkLen == 258) {
        return job.kBlks * size_t(8) * (sizeof(uint16_t) * 32 + 32 * 32 / 2 + sizeof(uint16_t) * 32);
    }
    if (MNNSpacemitIme2IsHpBlkLen(job.blkLen)) {
        return job.kBlks * (size_t(8) * (sizeof(uint16_t) * 32 + 32 * 32 / 2) + (useZp ? size_t(8) * 32 : 0));
    }
    return job.kBlks * (sizeof(uint16_t) * 32 + (useZp ? 32 : 0) + 32 * 32 / 2);
}

static constexpr bool MNNSpacemitIme2TcmStageBEnabled() {
    return false;
}

static inline size_t MNNSpacemitIme2ARowStrideForJob(const MNNSpacemitIme2Job& job) {
    const size_t aRowBlockSize =
        MNNSpacemitIme2IsHpBlkLen(job.blkLen)
            ? (size_t(256) + size_t(8) * sizeof(uint16_t) + size_t(8) * sizeof(uint16_t) + sizeof(uint16_t))
            : MNN_IME2_Q8_BLOCK_BYTES;
    return aRowBlockSize * job.kBlks;
}

static __attribute__((noinline)) void MNNSpacemitIme2RvvMemcpy1d(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<uint8_t*>(dst);
    auto* s = static_cast<const uint8_t*>(src);
    const size_t fullVl = __riscv_vsetvl_e8m8(bytes);
    const size_t unrollBytes = fullVl * 4;
    while (fullVl > 0 && bytes >= unrollBytes) {
        const vuint8m8_t v0 = __riscv_vle8_v_u8m8(s, fullVl);
        const vuint8m8_t v1 = __riscv_vle8_v_u8m8(s + fullVl, fullVl);
        const vuint8m8_t v2 = __riscv_vle8_v_u8m8(s + fullVl * 2, fullVl);
        const vuint8m8_t v3 = __riscv_vle8_v_u8m8(s + fullVl * 3, fullVl);
        __riscv_vse8_v_u8m8(d, v0, fullVl);
        __riscv_vse8_v_u8m8(d + fullVl, v1, fullVl);
        __riscv_vse8_v_u8m8(d + fullVl * 2, v2, fullVl);
        __riscv_vse8_v_u8m8(d + fullVl * 3, v3, fullVl);
        s += unrollBytes;
        d += unrollBytes;
        bytes -= unrollBytes;
    }
    while (bytes > 0) {
        const size_t vl = __riscv_vsetvl_e8m8(bytes);
        vuint8m8_t v = __riscv_vle8_v_u8m8(s, vl);
        __riscv_vse8_v_u8m8(d, v, vl);
        s += vl;
        d += vl;
        bytes -= vl;
    }
}

static inline MNNSpacemitIme2Job MNNSpacemitIme2MaybeStageToTcm(const MNNSpacemitIme2Job& job, void* tcmBuffer,
                                                                size_t tcmSize) {
    if (!MNNSpacemitIme2TcmEnabled() || tcmBuffer == nullptr || tcmSize == 0) {
        return job;
    }
    if (!MNNSpacemitIme2TcmStageBEnabled()) {
        const size_t aBytes = MNNSpacemitIme2ARowStrideForJob(job) * job.countM;
        if (job.a == nullptr || aBytes == 0 || aBytes > tcmSize) {
            return job;
        }
        MNNSpacemitIme2RvvMemcpy1d(tcmBuffer, job.a, aBytes);
        MNNSpacemitIme2Job staged = job;
        staged.a = static_cast<const uint8_t*>(tcmBuffer);
        return staged;
    }
    if (job.b == nullptr || job.countN == 0) {
        return job;
    }
    const size_t groupCount = (job.countN + 31) / 32;
    const size_t bBytes = MNNSpacemitIme2BStrideForJob(job) * groupCount;
    constexpr size_t minBytes = 4096;
    if (bBytes < minBytes || bBytes > tcmSize) {
        return job;
    }
    MNNSpacemitIme2RvvMemcpy1d(tcmBuffer, job.b, bBytes);
    MNNSpacemitIme2Job staged = job;
    staged.b = static_cast<const uint8_t*>(tcmBuffer);
    if (job.bZp != nullptr) {
        staged.bZp = staged.b;
    }
    return staged;
}

static inline size_t MNNSpacemitIme2RunGemmRows(const MNNSpacemitIme2Job& originalJob, void* tcmBuffer = nullptr,
                                                size_t tcmSize = 0) {
    const MNNSpacemitIme2Job job = MNNSpacemitIme2MaybeStageToTcm(originalJob, tcmBuffer, tcmSize);
    const size_t aRowBlockSize =
        MNNSpacemitIme2IsHpBlkLen(job.blkLen)
            ? (size_t(256) + size_t(8) * sizeof(uint16_t) + size_t(8) * sizeof(uint16_t) + sizeof(uint16_t))
            : MNN_IME2_Q8_BLOCK_BYTES;
    const size_t aRowStride = aRowBlockSize * job.kBlks;
    constexpr size_t maxKernelM = 8;
    size_t handled = 0;
    size_t remaining = job.countM;
    const uint8_t* a = job.a;
    float* c = job.c;
    while (remaining > 0) {
        const size_t callM = std::min(remaining, maxKernelM);
        const size_t rows = job.gemm(job.blkLen, a, job.b, job.bZp, c, callM, job.countN, job.kBlks, job.ldc);
        if (rows == 0 || rows > callM) {
            return handled;
        }
        handled += rows;
        remaining -= rows;
        a += rows * aRowStride;
        c += rows * job.ldc;
    }
    return handled;
}

static inline void MNNSpacemitIme2PostJob(const MNNSpacemitIme2Job& job, size_t result) {
    if (!job.doPost || result != job.countM) {
        return;
    }
    const bool scaleCByInput = MNNSpacemitIme2IsHpBlkLen(job.blkLen);
    if (job.countM == 1) {
        MNNSpacemitIme2PostChunk(job.dst, job.dstStep, job.c, job.countN, job.dzStart, job.fullCountN, job.residual,
                                 job.post, job.biasPtr, job.fp32min, job.fp32max, scaleCByInput, job.skipResidual);
    } else {
        MNNSpacemitIme2PostChunkM(job.dst, job.dstStep, job.c, job.countM, job.countN, job.dzStart, job.fullCountN,
                                  job.residual, job.post, job.biasPtr, job.fp32min, job.fp32max, scaleCByInput,
                                  job.skipResidual, job.directSrcKernelSum, job.directSrcKernelSumStride,
                                  job.directSrcRowOffset, job.blkLen);
    }
}

static inline bool MNNSpacemitIme2RunDecodeTcmPairTile(const MNNSpacemitIme2Job& job, size_t round,
                                                       const uint8_t* packedA, const uint8_t* packedB) {
    MNNSpacemitIme2Job tile = job;
    const size_t groupDelta = round * job.decodeTcmPairGroupStep;
    tile.decodeTcmPairBarrier = nullptr;
    tile.a = packedA;
    tile.b = packedB;
    tile.bZp = job.bZp != nullptr ? packedB : nullptr;
    tile.c = job.c + groupDelta * 32;
    tile.dzStart = job.dzStart + groupDelta * 8;
    const size_t rows = tile.gemm(tile.blkLen, tile.a, tile.b, tile.bZp, tile.c, 1, tile.countN, tile.kBlks, tile.ldc);
    MNNSpacemitIme2PostJob(tile, rows);
    return rows == 1;
}

static inline size_t MNNSpacemitIme2RunDecodeTcmPairDirect(const MNNSpacemitIme2Job& job) {
    size_t completed = 0;
    for (size_t round = 0; round < job.decodeTcmPairRounds; ++round) {
        const uint8_t* packedB = job.b + round * job.decodeTcmPairGroupStep * job.decodeTcmPairBStride;
        completed += MNNSpacemitIme2RunDecodeTcmPairTile(job, round, job.a, packedB) ? 1 : 0;
    }
    return completed;
}

static inline size_t MNNSpacemitIme2RunDecodeTcmPairJob(const MNNSpacemitIme2Job& job, void* tcmBuffer,
                                                        size_t tcmSize) {
    auto* barrier = job.decodeTcmPairBarrier;
    if (barrier == nullptr) {
        return 0;
    }
    const size_t aBytes = MNNSpacemitIme2ARowStrideForJob(job);
    const size_t alignedABytes = (aBytes + 63) & ~static_cast<size_t>(63);
    const size_t groupsPerTile = job.countN / 32;
    const size_t stagedBBytes = job.decodeTcmPairBStride * groupsPerTile;
    const bool directReady = job.decodeTcmPairRole < 2 && job.decodeTcmPairRounds > 0 && job.decodeTcmPairBStride > 0 &&
                             groupsPerTile > 0 && job.countN % 32 == 0 &&
                             job.decodeTcmPairGroupStep >= groupsPerTile * 2 && job.gemm != nullptr &&
                             job.a != nullptr && job.b != nullptr && job.c != nullptr;
    const bool localReady =
        directReady && tcmBuffer != nullptr && alignedABytes <= tcmSize && stagedBBytes <= tcmSize - alignedABytes;
    if (!localReady) {
        barrier->tcmReady.store(false, std::memory_order_release);
    }

    auto* localA = static_cast<uint8_t*>(tcmBuffer);
    auto* localB = localReady ? localA + alignedABytes : nullptr;
    const size_t role = job.decodeTcmPairRole;
    size_t completed = 0;
    // Fold the readiness handshake into the first copy barrier. A failed allocation may make
    // role 0 perform one harmless local copy, but saves one pair rendezvous on the hot path.
    if (role == 0 && localReady) {
        MNNSpacemitIme2RvvMemcpy1d(localA, job.a, aBytes);
        MNNSpacemitIme2RvvMemcpy1d(localB, job.b, stagedBBytes);
    }
    barrier->arriveAndWait();
    if (!barrier->tcmReady.load(std::memory_order_acquire)) {
        return directReady ? MNNSpacemitIme2RunDecodeTcmPairDirect(job) : 0;
    }

    if (role == 1) {
        MNNSpacemitIme2RvvMemcpy1d(localA, job.a, aBytes);
        MNNSpacemitIme2RvvMemcpy1d(localB, job.b, stagedBBytes);
    }
    for (size_t round = 0; round < job.decodeTcmPairRounds; ++round) {
        if (role == 1) {
            barrier->arriveAndWait();
        }
        completed += MNNSpacemitIme2RunDecodeTcmPairTile(job, round, localA, localB) ? 1 : 0;
        if (role == 0) {
            barrier->arriveAndWait();
        }
        if (round + 1 < job.decodeTcmPairRounds) {
            const uint8_t* packedB = job.b + (round + 1) * job.decodeTcmPairGroupStep * job.decodeTcmPairBStride;
            MNNSpacemitIme2RvvMemcpy1d(localB, packedB, stagedBBytes);
        }
    }
    return completed;
}

static inline bool MNNSpacemitIme2CanRunDirectC4Epilogue(const MNNSpacemitIme2Job& job) {
    return job.directC4Epilogue && job.blkLen == 258 && job.bZp == nullptr && job.countM == 4 && job.countN != 0 &&
           job.countN % 32 == 0 && job.doPost && job.dst != nullptr &&
           job.dstStep >= job.countM * GEMM_INT8_UNIT * sizeof(float) && job.dzStart == 0 &&
           job.fullCountN == job.countN && job.post != nullptr && job.post->useInt8 == 0 &&
           job.post->inputScale != nullptr && job.post->inputBias == nullptr && job.skipResidual;
}

static inline size_t MNNSpacemitIme2RunDirectC4Epilogue(const MNNSpacemitIme2Job& originalJob, void* tcmBuffer,
                                                        size_t tcmSize) {
    if (!MNNSpacemitIme2CanRunDirectC4Epilogue(originalJob)) {
        return 0;
    }
#if defined(MNN_USE_SPACEMIT_IME2)
    const MNNSpacemitIme2Job job = MNNSpacemitIme2MaybeStageToTcm(originalJob, tcmBuffer, tcmSize);
    const bool needClamp = job.post->fp32minmax != nullptr && !MNNRvvFp32MinMaxIsFullRange(job.fp32min, job.fp32max);
    return MNNSpacemitIme2GemmI8I4HpM4DirectC4Local(job.blkLen, job.a, job.b, job.bZp, job.dst, job.dstStep, job.countM,
                                                    job.countN, job.kBlks, job.post->inputScale, job.biasPtr,
                                                    job.fp32min, job.fp32max, needClamp ? 1 : 0);
#else
    (void)tcmBuffer;
    (void)tcmSize;
    return 0;
#endif
}

static inline size_t MNNSpacemitIme2RunLinearStrideJob(const MNNSpacemitIme2Job& job, void* tcmBuffer = nullptr,
                                                       size_t tcmSize = 0) {
    if (!job.linearStride || job.post == nullptr || job.linearRowsPerBlock == 0 || job.linearRowStep == 0 ||
        job.linearRowStart >= job.linearRowEnd || job.linearTotalRows == 0) {
        return 0;
    }
    const size_t aRowStride = MNNSpacemitIme2ARowStrideForJob(job);
    size_t handled = 0;
    for (size_t rowOffset = job.linearRowStart; rowOffset < job.linearRowEnd; rowOffset += job.linearRowStep) {
        const size_t rows = std::min(job.linearRowsPerBlock, job.linearTotalRows - rowOffset);
        MNNSpacemitIme2Job chunk = job;
        QuanPostTreatParameters chunkPost = *job.post;
        chunk.linearStride = false;
        chunk.a = job.a + rowOffset * aRowStride;
        chunk.c = job.c;
        chunk.countM = rows;
        chunk.dst = job.dst + rowOffset * GEMM_INT8_UNIT * sizeof(float);
        if (chunkPost.inputScale != nullptr) {
            chunkPost.inputScale = job.post->inputScale + rowOffset;
        }
        chunk.post = &chunkPost;
        if (job.directSrcKernelSum != nullptr) {
            chunk.directSrcKernelSum = job.directSrcKernelSum;
            chunk.directSrcKernelSumStride = job.directSrcKernelSumStride;
            chunk.directSrcRowOffset = rowOffset;
        }
        size_t rowsHandled = MNNSpacemitIme2RunDirectC4Epilogue(chunk, tcmBuffer, tcmSize);
        if (rowsHandled != rows) {
            rowsHandled = MNNSpacemitIme2RunGemmRows(chunk, tcmBuffer, tcmSize);
            MNNSpacemitIme2PostJob(chunk, rowsHandled);
        }
        if (rowsHandled != rows) {
            return handled;
        }
        handled += rowsHandled;
    }
    return handled;
}

static size_t MNNSpacemitIme2RunPackedFloatAJob(const MNNSpacemitIme2Job& job, void* tcmBuffer, size_t tcmSize) {
    if (!job.packFloatA || job.packFloatSrc == nullptr || job.packInputScale == nullptr ||
        job.packQuantScale == nullptr || job.post == nullptr || job.countM == 0 ||
        job.packRowBegin + job.countM > job.packSrcRows) {
        return 0;
    }
    const size_t aRowStride = MNNSpacemitIme2ARowStrideForJob(job);
    if (aRowStride == 0) {
        return 0;
    }
    thread_local std::vector<uint8_t> packedA;
    thread_local std::vector<float> srcKernelSum;
    packedA.resize(aRowStride * job.countM);
    srcKernelSum.resize(job.packBlockNum * job.countM);
    if (MNNSpacemitIme2PackFloatAHpStridedRowsWithSum(
            packedA.data(), srcKernelSum.data(), job.packInputScale, job.packQuantScale, job.packFloatSrc,
            job.packSrcDepthQuad, job.packBlockNum, job.packSrcRows, job.packRowBegin, job.countM) == 0) {
        return 0;
    }

    QuanPostTreatParameters localPost = *job.post;
    localPost.inputScale = job.packInputScale + job.packRowBegin;
    localPost.inputBias = nullptr;
    localPost.srcKernelSum = srcKernelSum.data();

    MNNSpacemitIme2Job runJob = job;
    runJob.packFloatA = false;
    runJob.a = packedA.data();
    runJob.post = &localPost;
    runJob.directSrcKernelSum = nullptr;
    runJob.directSrcKernelSumStride = 0;
    runJob.directSrcRowOffset = 0;
    const size_t rows = MNNSpacemitIme2RunGemmRows(runJob, tcmBuffer, tcmSize);
    MNNSpacemitIme2PostJob(runJob, rows);
    return rows;
}

class MNNSpacemitIme2Worker {
public:
    static MNNSpacemitIme2Worker& get(size_t index) {
        auto& workers = allWorkers();
        return *workers[index % workers.size()];
    }

    static size_t count() { return allWorkers().size(); }

    void startJob(const MNNSpacemitIme2Job& job) {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this]() { return mJobDone && !mHasJob; });
        mJob = job;
        mJobDone = false;
        mHasJob = true;
        mCv.notify_all();
    }

    void start(MNNSpacemitIme2GemmI8I4 gemm, size_t blkLen, const uint8_t* a, const uint8_t* b, const uint8_t* bZp,
               float* c, size_t countM, size_t countN, size_t kBlks, size_t ldc, bool doPost = false,
               int8_t* dst = nullptr, size_t dstStep = 0, size_t dzStart = 0, size_t fullCountN = 0,
               const float* residual = nullptr, const QuanPostTreatParameters* post = nullptr,
               const float* biasPtr = nullptr, float fp32min = 0.0f, float fp32max = 0.0f, bool skipResidual = false,
               const float* directSrcKernelSum = nullptr, size_t directSrcKernelSumStride = 0,
               size_t directSrcRowOffset = 0) {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this]() { return mJobDone && !mHasJob; });
        mJob = MNNSpacemitIme2Job();
        mJob.gemm = gemm;
        mJob.blkLen = blkLen;
        mJob.a = a;
        mJob.b = b;
        mJob.bZp = bZp;
        mJob.c = c;
        mJob.countM = countM;
        mJob.countN = countN;
        mJob.kBlks = kBlks;
        mJob.ldc = ldc;
        mJob.doPost = doPost;
        mJob.dst = dst;
        mJob.dstStep = dstStep;
        mJob.dzStart = dzStart;
        mJob.fullCountN = fullCountN;
        mJob.residual = residual;
        mJob.post = post;
        mJob.biasPtr = biasPtr;
        mJob.fp32min = fp32min;
        mJob.fp32max = fp32max;
        mJob.skipResidual = skipResidual;
        mJob.directSrcKernelSum = directSrcKernelSum;
        mJob.directSrcKernelSumStride = directSrcKernelSumStride;
        mJob.directSrcRowOffset = directSrcRowOffset;
        mJobDone = false;
        mHasJob = true;
        mCv.notify_all();
    }

    size_t wait() {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this]() { return mJobDone; });
        return mResult;
    }

    size_t run(MNNSpacemitIme2GemmI8I4 gemm, size_t blkLen, const uint8_t* a, const uint8_t* b, const uint8_t* bZp,
               float* c, size_t countM, size_t countN, size_t kBlks, size_t ldc, bool doPost = false,
               int8_t* dst = nullptr, size_t dstStep = 0, size_t dzStart = 0, size_t fullCountN = 0,
               const float* residual = nullptr, const QuanPostTreatParameters* post = nullptr,
               const float* biasPtr = nullptr, float fp32min = 0.0f, float fp32max = 0.0f, bool skipResidual = false,
               const float* directSrcKernelSum = nullptr, size_t directSrcKernelSumStride = 0,
               size_t directSrcRowOffset = 0) {
        start(gemm, blkLen, a, b, bZp, c, countM, countN, kBlks, ldc, doPost, dst, dstStep, dzStart, fullCountN,
              residual, post, biasPtr, fp32min, fp32max, skipResidual, directSrcKernelSum, directSrcKernelSumStride,
              directSrcRowOffset);
        return wait();
    }

    ~MNNSpacemitIme2Worker() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
            mCv.notify_all();
        }
        if (mThread.joinable()) {
            mThread.join();
        }
    }

private:
    static std::vector<std::shared_ptr<MNNSpacemitIme2Worker>>& allWorkers() {
        static std::vector<std::shared_ptr<MNNSpacemitIme2Worker>> workers = createWorkers();
        return workers;
    }

    static std::vector<std::shared_ptr<MNNSpacemitIme2Worker>> createWorkers() {
        constexpr int count = 8;
        std::vector<std::shared_ptr<MNNSpacemitIme2Worker>> workers;
        workers.reserve(count);
        for (int i = 0; i < count; ++i) {
            workers.emplace_back(std::shared_ptr<MNNSpacemitIme2Worker>(new MNNSpacemitIme2Worker(i)));
        }
        return workers;
    }

    explicit MNNSpacemitIme2Worker(size_t index) : mIndex(index) {
        mThread = std::thread([this]() { this->loop(); });
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this]() { return mReady; });
    }

    void loop() {
        MNNSpacemitIme2BindWorker(mIndex);
        size_t tcmSize = 0;
        void* tcmBuffer = MNNSpacemitIme2TcmAcquire(mIndex, &tcmSize);
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mReady = true;
            mCv.notify_all();
        }
        while (true) {
            MNNSpacemitIme2Job job;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCv.wait(lock, [this]() { return mHasJob || mStop; });
                if (mStop) {
                    MNNSpacemitIme2TcmRelease(mIndex, tcmBuffer);
                    MNNSpacemitIme2ClearWorker(mIndex);
                    return;
                }
                job = mJob;
                mHasJob = false;
            }
            const size_t result = [&]() {
                if (job.decodeTcmPairBarrier != nullptr) {
                    return MNNSpacemitIme2RunDecodeTcmPairJob(job, tcmBuffer, tcmSize);
                }
                if (job.tcmTask != nullptr) {
                    job.tcmTask(job.tcmTaskIndex, tcmBuffer, tcmSize, job.tcmTaskContext);
                    return size_t(1);
                }
                return job.packFloatA ? MNNSpacemitIme2RunPackedFloatAJob(job, tcmBuffer, tcmSize)
                                      : (job.linearStride ? MNNSpacemitIme2RunLinearStrideJob(job, tcmBuffer, tcmSize)
                                                          : MNNSpacemitIme2RunGemmRows(job, tcmBuffer, tcmSize));
            }();
            if (job.decodeTcmPairBarrier == nullptr && job.tcmTask == nullptr && !job.linearStride && !job.packFloatA) {
                MNNSpacemitIme2PostJob(job, result);
            }
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mResult = result;
                mJobDone = true;
                mCv.notify_all();
            }
        }
    }

    std::mutex mMutex;
    std::condition_variable mCv;
    std::thread mThread;
    MNNSpacemitIme2Job mJob;
    size_t mIndex = 0;
    size_t mResult = 0;
    bool mReady = false;
    bool mHasJob = false;
    bool mJobDone = true;
    bool mStop = false;
};

class MNNSpacemitIme2SpinWorker {
public:
    static MNNSpacemitIme2SpinWorker& get(size_t index) {
        auto& workers = allWorkers();
        return *workers[index % workers.size()];
    }

    static size_t count() { return allWorkers().size(); }

    void startJob(const MNNSpacemitIme2Job& job) {
        while (mStartLock.test_and_set(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
        while (mBusy.load(std::memory_order_acquire) || mResultReady.load(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
        mJob = job;
        mBusy.store(true, std::memory_order_release);
        mStartLock.clear(std::memory_order_release);
    }

    void start(MNNSpacemitIme2GemmI8I4 gemm, size_t blkLen, const uint8_t* a, const uint8_t* b, const uint8_t* bZp,
               float* c, size_t countM, size_t countN, size_t kBlks, size_t ldc, bool doPost = false,
               int8_t* dst = nullptr, size_t dstStep = 0, size_t dzStart = 0, size_t fullCountN = 0,
               const float* residual = nullptr, const QuanPostTreatParameters* post = nullptr,
               const float* biasPtr = nullptr, float fp32min = 0.0f, float fp32max = 0.0f, bool skipResidual = false,
               const float* directSrcKernelSum = nullptr, size_t directSrcKernelSumStride = 0,
               size_t directSrcRowOffset = 0) {
        while (mStartLock.test_and_set(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
        while (mBusy.load(std::memory_order_acquire) || mResultReady.load(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
        mJob = MNNSpacemitIme2Job();
        mJob.gemm = gemm;
        mJob.blkLen = blkLen;
        mJob.a = a;
        mJob.b = b;
        mJob.bZp = bZp;
        mJob.c = c;
        mJob.countM = countM;
        mJob.countN = countN;
        mJob.kBlks = kBlks;
        mJob.ldc = ldc;
        mJob.doPost = doPost;
        mJob.dst = dst;
        mJob.dstStep = dstStep;
        mJob.dzStart = dzStart;
        mJob.fullCountN = fullCountN;
        mJob.residual = residual;
        mJob.post = post;
        mJob.biasPtr = biasPtr;
        mJob.fp32min = fp32min;
        mJob.fp32max = fp32max;
        mJob.skipResidual = skipResidual;
        mJob.directSrcKernelSum = directSrcKernelSum;
        mJob.directSrcKernelSumStride = directSrcKernelSumStride;
        mJob.directSrcRowOffset = directSrcRowOffset;
        mBusy.store(true, std::memory_order_release);
        mStartLock.clear(std::memory_order_release);
    }

    size_t wait() {
        while (mBusy.load(std::memory_order_acquire) || !mResultReady.load(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
        const size_t result = mResult;
        mResultReady.store(false, std::memory_order_release);
        return result;
    }

    size_t run(MNNSpacemitIme2GemmI8I4 gemm, size_t blkLen, const uint8_t* a, const uint8_t* b, const uint8_t* bZp,
               float* c, size_t countM, size_t countN, size_t kBlks, size_t ldc, bool doPost = false,
               int8_t* dst = nullptr, size_t dstStep = 0, size_t dzStart = 0, size_t fullCountN = 0,
               const float* residual = nullptr, const QuanPostTreatParameters* post = nullptr,
               const float* biasPtr = nullptr, float fp32min = 0.0f, float fp32max = 0.0f, bool skipResidual = false,
               const float* directSrcKernelSum = nullptr, size_t directSrcKernelSumStride = 0,
               size_t directSrcRowOffset = 0) {
        start(gemm, blkLen, a, b, bZp, c, countM, countN, kBlks, ldc, doPost, dst, dstStep, dzStart, fullCountN,
              residual, post, biasPtr, fp32min, fp32max, skipResidual, directSrcKernelSum, directSrcKernelSumStride,
              directSrcRowOffset);
        return wait();
    }

    ~MNNSpacemitIme2SpinWorker() {
        mStop.store(true, std::memory_order_release);
        if (mThread.joinable()) {
            mThread.join();
        }
    }

private:
    static std::vector<std::shared_ptr<MNNSpacemitIme2SpinWorker>>& allWorkers() {
        static std::vector<std::shared_ptr<MNNSpacemitIme2SpinWorker>> workers = createWorkers();
        return workers;
    }

    static std::vector<std::shared_ptr<MNNSpacemitIme2SpinWorker>> createWorkers() {
        constexpr int count = 8;
        std::vector<std::shared_ptr<MNNSpacemitIme2SpinWorker>> workers;
        workers.reserve(count);
        for (int i = 0; i < count; ++i) {
            workers.emplace_back(std::shared_ptr<MNNSpacemitIme2SpinWorker>(new MNNSpacemitIme2SpinWorker(i)));
        }
        return workers;
    }

    explicit MNNSpacemitIme2SpinWorker(size_t index) : mIndex(index) {
        mThread = std::thread([this]() { this->loop(); });
        while (!mReady.load(std::memory_order_acquire)) {
            MNNSpacemitIme2Relax();
        }
    }

    size_t executeJob(const MNNSpacemitIme2Job& job, void* tcmBuffer, size_t tcmSize) {
        const size_t result = [&]() {
            if (job.decodeTcmPairBarrier != nullptr) {
                return MNNSpacemitIme2RunDecodeTcmPairJob(job, tcmBuffer, tcmSize);
            }
            if (job.tcmTask != nullptr) {
                job.tcmTask(job.tcmTaskIndex, tcmBuffer, tcmSize, job.tcmTaskContext);
                return size_t(1);
            }
            return job.packFloatA ? MNNSpacemitIme2RunPackedFloatAJob(job, tcmBuffer, tcmSize)
                                  : (job.linearStride ? MNNSpacemitIme2RunLinearStrideJob(job, tcmBuffer, tcmSize)
                                                      : MNNSpacemitIme2RunGemmRows(job, tcmBuffer, tcmSize));
        }();
        if (job.decodeTcmPairBarrier == nullptr && job.tcmTask == nullptr && !job.linearStride && !job.packFloatA) {
            MNNSpacemitIme2PostJob(job, result);
        }
        return result;
    }

    void loop() {
        MNNSpacemitIme2BindWorker(mIndex);
        size_t tcmSize = 0;
        void* tcmBuffer = MNNSpacemitIme2TcmAcquire(mIndex, &tcmSize);
        mReady.store(true, std::memory_order_release);
        while (!mStop.load(std::memory_order_acquire)) {
            if (!mBusy.load(std::memory_order_acquire)) {
                MNNSpacemitIme2Relax();
                continue;
            }
            const auto job = mJob;
            const size_t result = executeJob(job, tcmBuffer, tcmSize);
            mResult = result;
            mResultReady.store(true, std::memory_order_release);
            mBusy.store(false, std::memory_order_release);
        }
        MNNSpacemitIme2TcmRelease(mIndex, tcmBuffer);
        MNNSpacemitIme2ClearWorker(mIndex);
    }

    std::thread mThread;
    MNNSpacemitIme2Job mJob;
    size_t mIndex = 0;
    size_t mResult = 0;
    std::atomic<bool> mReady{false};
    std::atomic<bool> mBusy{false};
    std::atomic<bool> mResultReady{false};
    std::atomic<bool> mStop{false};
    std::atomic_flag mStartLock = ATOMIC_FLAG_INIT;
};

extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t taskCount, MNNSpacemitIme2TcmTask task, void* context) {
    if (task == nullptr || taskCount == 0) {
        return 0;
    }
    const bool spinWorker = MNNSpacemitIme2TcmTaskSpinEnabled();
    const size_t workerCount = spinWorker ? MNNSpacemitIme2SpinWorker::count() : MNNSpacemitIme2Worker::count();
    if (taskCount > workerCount) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatchLock(MNNSpacemitIme2DispatchMutex());
    for (size_t i = 0; i < taskCount; ++i) {
        MNNSpacemitIme2Job job;
        job.tcmTask = task;
        job.tcmTaskContext = context;
        job.tcmTaskIndex = i;
        if (spinWorker) {
            MNNSpacemitIme2SpinWorker::get(i).startJob(job);
        } else {
            MNNSpacemitIme2Worker::get(i).startJob(job);
        }
    }
    size_t completed = 0;
    for (size_t i = 0; i < taskCount; ++i) {
        completed += spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
    }
    return completed;
}

static inline uint16_t MNNFloatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x00001000u) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x00001000u) >> 13));
}

static inline float MNNHalfBitsToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    int32_t exponent = static_cast<int32_t>((value >> 10) & 0x1fu);
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits |= static_cast<uint32_t>(exponent + (127 - 15)) << 23;
            bits |= mantissa << 13;
        }
    } else if (exponent == 31) {
        bits |= 0xffu << 23;
        bits |= mantissa << 13;
    } else {
        bits |= static_cast<uint32_t>(exponent + (127 - 15)) << 23;
        bits |= mantissa << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline float MNNIme2LoadHalf(const uint8_t* src) {
    uint16_t half = 0;
    std::memcpy(&half, src, sizeof(half));
    return MNNHalfBitsToFloat(half);
}

static inline bool MNNRvvFp32MinMaxIsFullRange(float minValue, float maxValue) {
    return minValue < -3.0e38f && maxValue > 3.0e38f;
}

static inline uint8_t MNNReadW4(const uint8_t* weight, int oc, int k) {
    if (oc == 0) {
        return static_cast<uint8_t>(weight[k] >> 4);
    }
    if (oc == 1) {
        return static_cast<uint8_t>(weight[GEMM_INT8_SRC_UNIT + k] >> 4);
    }
    if (oc == 2) {
        return static_cast<uint8_t>(weight[k] & 0x0f);
    }
    return static_cast<uint8_t>(weight[GEMM_INT8_SRC_UNIT + k] & 0x0f);
}

static inline float MNNIme2SymW4Scale(const int8_t* weightDz, size_t srcDepthQuad, int weightStepY, int oc, float scale,
                                      float bias) {
    float maxPositive = 0.0f;
    float maxNegative = 0.0f;
    for (size_t sz = 0; sz < srcDepthQuad; ++sz) {
        const uint8_t* weightSz = reinterpret_cast<const uint8_t*>(weightDz + weightStepY * sz);
        for (int k = 0; k < GEMM_INT8_SRC_UNIT; ++k) {
            const float value = static_cast<float>(MNNReadW4(weightSz, oc, k)) * scale + bias;
            maxPositive = std::max(maxPositive, value);
            maxNegative = std::max(maxNegative, -value);
        }
    }
    const float symScale = std::max(maxPositive / 7.0f, maxNegative / 8.0f);
    return symScale > 0.0f && std::isfinite(symScale) ? symScale : 1.0e-8f;
}

static inline uint8_t MNNIme2QuantizeSymW4(float value, float scale) {
    int q = static_cast<int>(std::round(value / scale));
    q = std::min(7, std::max(-8, q));
    return static_cast<uint8_t>(q + 8);
}

static inline size_t MNNIme2Q8BlockSize() {
    return MNN_IME2_Q8_BLOCK_BYTES;
}

static inline size_t MNNIme2Q4BlockSize(bool useZp = false) {
    return sizeof(uint16_t) * 32 + (useZp ? 32 : 0) + 32 * 32 / 2;
}

static inline size_t MNNIme2Q8HpBlockSize() {
    return size_t(256) + size_t(8) * sizeof(uint16_t) + size_t(8) * sizeof(uint16_t) + sizeof(uint16_t);
}

static inline size_t MNNIme2Q4HpSuperBlockSize(bool useZp = false) {
    return size_t(8) * (sizeof(uint16_t) * 32 + 32 * 32 / 2) + (useZp ? size_t(8) * 32 : 0);
}

static inline size_t MNNIme2Q4HpResidualSuperBlockSize() {
    return size_t(8) * (sizeof(uint16_t) * 32 + 32 * 32 / 2 + sizeof(uint16_t) * 32);
}

static inline size_t MNNIme2Q4HpAsymPairSuperBlockSize() {
    // A HP super-block covers eight K32 blocks. For block64 weights, store four pairs as
    // [scale N32, -weight_bias/8 N32, q(K32_0), q(K32_1)]. Replacing the duplicated
    // scale of the second K32 keeps the exact same 4608-byte super-block footprint.
    constexpr size_t size = size_t(4) * (sizeof(uint16_t) * 32 + sizeof(uint16_t) * 32 + 2 * (32 * 32 / 2));
    static_assert(size == size_t(8) * (sizeof(uint16_t) * 32 + 32 * 32 / 2),
                  "asymmetric pair layout must preserve the HP B stride");
    return size;
}

static inline uint8_t MNNIme2ChooseZp(float scale, float weightBias) {
    if (scale == 0.0f || !std::isfinite(scale) || !std::isfinite(weightBias)) {
        return 8;
    }
    const float ratio = -weightBias / scale;
    const int zp = static_cast<int>(std::round(ratio));
    if (zp < 0 || zp > 15 || !std::isfinite(ratio)) {
        return static_cast<uint8_t>(std::min(std::max(zp, 0), 15));
    }
    return static_cast<uint8_t>(zp);
}

static inline void MNNIme2StoreHalf(uint8_t* dst, float value) {
    const uint16_t half = MNNFloatToHalfBits(value);
    std::memcpy(dst, &half, sizeof(half));
}

static void MNNPackIme2A1(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t realCount, size_t row,
                          const float* inputScale) {
    const size_t kBlocks = srcDepthQuad / 2;
    const float scale = inputScale == nullptr ? 1.0f : inputScale[row];
    for (size_t kb = 0; kb < kBlocks; ++kb) {
        uint8_t* block = dst + kb * MNNIme2Q8BlockSize();
        std::memcpy(block, &scale, sizeof(float));
        int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(float) + sizeof(int16_t));
        int32_t sum = 0;
        for (int kk = 0; kk < 32; ++kk) {
            const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
            const int k = kk % GEMM_INT8_SRC_UNIT;
            const int8_t v = src[sz * realCount * GEMM_INT8_SRC_UNIT + row * GEMM_INT8_SRC_UNIT + k];
            q[kk] = v;
            sum += v;
        }
        const int16_t negSum = static_cast<int16_t>(-sum);
        std::memcpy(block + sizeof(float), &negSum, sizeof(int16_t));
    }
}

static void MNNPackIme2A1AllBlocks(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                   const QuanPostTreatParameters* post) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    for (size_t bk = 0; bk < blockNum; ++bk) {
        const int8_t* srcBk = src + bk * srcDepthQuad * GEMM_INT8_SRC_UNIT;
        const float scale =
            post->inputScale == nullptr ? 1.0f : (post->inputBias ? post->inputScale[bk] : post->inputScale[0]);
        for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
            uint8_t* block = dst + (bk * kBlocksPerBlock + kb) * MNNIme2Q8BlockSize();
            std::memcpy(block, &scale, sizeof(float));
            int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(float) + sizeof(int16_t));
            int32_t sum = 0;
            for (int kk = 0; kk < 32; ++kk) {
                const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                const int k = kk % GEMM_INT8_SRC_UNIT;
                const int8_t v = srcBk[sz * GEMM_INT8_SRC_UNIT + k];
                q[kk] = v;
                sum += v;
            }
            const int16_t negSum = static_cast<int16_t>(-sum);
            std::memcpy(block + sizeof(float), &negSum, sizeof(int16_t));
        }
    }
}

static void MNNPackIme2AMAllBlocks(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                   size_t realCount, const QuanPostTreatParameters* post) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t rowStride = MNNIme2Q8BlockSize() * kBlocksPerBlock * blockNum;
    const size_t srcBlockStride = srcDepthQuad * GEMM_INT8_SRC_UNIT * realCount;
    const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * realCount;
    for (size_t rowBase = 0; rowBase < realCount;) {
        const size_t rows = std::min<size_t>(4, realCount - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * rowStride;
            for (size_t bk = 0; bk < blockNum; ++bk) {
                const int8_t* srcBk = src + bk * srcBlockStride;
                float scales[4];
                for (size_t r = 0; r < 4; ++r) {
                    const size_t row = rowBase + r;
                    scales[r] = post->inputScale == nullptr ? 1.0f
                                                            : (post->inputBias ? post->inputScale[bk * realCount + row]
                                                                               : post->inputScale[row]);
                }
                for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
                    uint8_t* block = tileDst + (bk * kBlocksPerBlock + kb) * MNNIme2Q8BlockSize() * 4;
                    int16_t negSums[4] = {0, 0, 0, 0};
                    std::memcpy(block, scales, sizeof(scales));
                    int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(scales) + sizeof(negSums));
                    for (size_t r = 0; r < 4; ++r) {
                        int32_t sum = 0;
                        const int8_t* srcRow = srcBk + (rowBase + r) * GEMM_INT8_SRC_UNIT;
                        for (int kk = 0; kk < 32; ++kk) {
                            const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                            const int k = kk % GEMM_INT8_SRC_UNIT;
                            const int8_t v = srcRow[sz * srcDepthStride + k];
                            q[r * 32 + kk] = v;
                            sum += v;
                        }
                        negSums[r] = static_cast<int16_t>(-sum);
                    }
                    std::memcpy(block + sizeof(scales), negSums, sizeof(negSums));
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t row = rowBase + r;
            uint8_t* rowDst = dst + row * rowStride;
            for (size_t bk = 0; bk < blockNum; ++bk) {
                const int8_t* srcBk = src + bk * srcBlockStride + row * GEMM_INT8_SRC_UNIT;
                const float scale =
                    post->inputScale == nullptr
                        ? 1.0f
                        : (post->inputBias ? post->inputScale[bk * realCount + row] : post->inputScale[row]);
                for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
                    uint8_t* block = rowDst + (bk * kBlocksPerBlock + kb) * MNNIme2Q8BlockSize();
                    std::memcpy(block, &scale, sizeof(float));
                    int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(float) + sizeof(int16_t));
                    int32_t sum = 0;
                    for (int kk = 0; kk < 32; ++kk) {
                        const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                        const int k = kk % GEMM_INT8_SRC_UNIT;
                        const int8_t v = srcBk[sz * srcDepthStride + k];
                        q[kk] = v;
                        sum += v;
                    }
                    const int16_t negSum = static_cast<int16_t>(-sum);
                    std::memcpy(block + sizeof(float), &negSum, sizeof(int16_t));
                }
            }
        }
        rowBase += rows;
    }
}

static void MNNPackIme2AMAllBlocksTiled(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                        size_t realCount, size_t tileRows, const QuanPostTreatParameters* post) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t rowStride = MNNIme2Q8BlockSize() * kBlocksPerBlock * blockNum;
    const size_t fullTileStride = blockNum * srcDepthQuad * tileRows * GEMM_INT8_SRC_UNIT;
    for (size_t rowBase = 0; rowBase < realCount;) {
        const size_t rows = std::min<size_t>(4, realCount - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * rowStride;
            for (size_t bk = 0; bk < blockNum; ++bk) {
                float scales[4];
                for (size_t r = 0; r < 4; ++r) {
                    const size_t row = rowBase + r;
                    scales[r] = post->inputScale == nullptr ? 1.0f
                                                            : (post->inputBias ? post->inputScale[bk * realCount + row]
                                                                               : post->inputScale[row]);
                }
                for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
                    uint8_t* block = tileDst + (bk * kBlocksPerBlock + kb) * MNNIme2Q8BlockSize() * 4;
                    int16_t negSums[4] = {0, 0, 0, 0};
                    std::memcpy(block, scales, sizeof(scales));
                    int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(scales) + sizeof(negSums));
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t row = rowBase + r;
                        const size_t tile = row / tileRows;
                        const size_t localRow = row - tile * tileRows;
                        const size_t rowsInTile = std::min(tileRows, realCount - tile * tileRows);
                        const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * rowsInTile;
                        const int8_t* srcTile = src + tile * fullTileStride;
                        const int8_t* srcBk =
                            srcTile + bk * srcDepthQuad * srcDepthStride + localRow * GEMM_INT8_SRC_UNIT;
                        int32_t sum = 0;
                        for (int kk = 0; kk < 32; ++kk) {
                            const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                            const int k = kk % GEMM_INT8_SRC_UNIT;
                            const int8_t v = srcBk[sz * srcDepthStride + k];
                            q[r * 32 + kk] = v;
                            sum += v;
                        }
                        negSums[r] = static_cast<int16_t>(-sum);
                    }
                    std::memcpy(block + sizeof(scales), negSums, sizeof(negSums));
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t row = rowBase + r;
            const size_t tile = row / tileRows;
            const size_t localRow = row - tile * tileRows;
            const size_t rowsInTile = std::min(tileRows, realCount - tile * tileRows);
            const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * rowsInTile;
            const int8_t* srcTile = src + tile * fullTileStride;
            uint8_t* rowDst = dst + row * rowStride;
            for (size_t bk = 0; bk < blockNum; ++bk) {
                const int8_t* srcBk = srcTile + bk * srcDepthQuad * srcDepthStride + localRow * GEMM_INT8_SRC_UNIT;
                const float scale =
                    post->inputScale == nullptr
                        ? 1.0f
                        : (post->inputBias ? post->inputScale[bk * realCount + row] : post->inputScale[row]);
                for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
                    uint8_t* block = rowDst + (bk * kBlocksPerBlock + kb) * MNNIme2Q8BlockSize();
                    std::memcpy(block, &scale, sizeof(float));
                    int8_t* q = reinterpret_cast<int8_t*>(block + sizeof(float) + sizeof(int16_t));
                    int32_t sum = 0;
                    for (int kk = 0; kk < 32; ++kk) {
                        const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                        const int k = kk % GEMM_INT8_SRC_UNIT;
                        const int8_t v = srcBk[sz * srcDepthStride + k];
                        q[kk] = v;
                        sum += v;
                    }
                    const int16_t negSum = static_cast<int16_t>(-sum);
                    std::memcpy(block + sizeof(float), &negSum, sizeof(int16_t));
                }
            }
        }
        rowBase += rows;
    }
}

static inline int32_t MNNIme2Copy32AndSum(int8_t* dst, const int8_t* src0, const int8_t* src1) {
    std::memcpy(dst, src0, GEMM_INT8_SRC_UNIT);
    std::memcpy(dst + GEMM_INT8_SRC_UNIT, src1, GEMM_INT8_SRC_UNIT);
    int32_t sum = 0;
    for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
        sum += src0[i] + src1[i];
    }
    return sum;
}

static void MNNPackIme2A1AllBlocksHp(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                     const float* blockInputScale, bool exactCenterSum) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t hpBlockSize = MNNIme2Q8HpBlockSize();
    MNN_ASSERT(kBlocks % 8 == 0);
    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* block = dst + super * hpBlockSize;
        std::memset(block, 0, hpBlockSize);
        uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
        for (size_t sub = 0; sub < 8; ++sub) {
            const size_t linearKb = super * 8 + sub;
            const size_t bk = linearKb / kBlocksPerBlock;
            const size_t kb = linearKb - bk * kBlocksPerBlock;
            const int8_t* srcBk = src + bk * srcDepthQuad * GEMM_INT8_SRC_UNIT;
            uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
            const float scale = blockInputScale == nullptr ? 1.0f : blockInputScale[bk];
            MNNIme2StoreHalf(subBlock, scale);
            int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
            const int8_t* src0 = srcBk + (kb * 2) * GEMM_INT8_SRC_UNIT;
            const int8_t* src1 = src0 + GEMM_INT8_SRC_UNIT;
            const int32_t sum = MNNIme2Copy32AndSum(q, src0, src1);
            const int16_t negSum8 = static_cast<int16_t>(-sum * 8);
            if (exactCenterSum) {
                std::memcpy(sumBase + sub * sizeof(int16_t), &negSum8, sizeof(negSum8));
            } else {
                MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(negSum8));
            }
        }
        MNNIme2StoreHalf(block + hpBlockSize - sizeof(uint16_t), 1.0f);
    }
}

static void MNNPackIme2AMAllBlocksHp(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                     size_t realCount) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t srcBlockStride = srcDepthQuad * GEMM_INT8_SRC_UNIT * realCount;
    const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * realCount;
    MNN_ASSERT(kBlocks % 8 == 0);
    for (size_t rowBase = 0; rowBase < realCount;) {
        const size_t rows = std::min<size_t>(4, realCount - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
                std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    const int8_t* srcBk = src + bk * srcBlockStride;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        const int8_t* srcRow = srcBk + (rowBase + r) * GEMM_INT8_SRC_UNIT;
                        const int8_t* src0 = srcRow + (kb * 2) * srcDepthStride;
                        const int8_t* src1 = src0 + srcDepthStride;
                        const int32_t sum = MNNIme2Copy32AndSum(q + r * 32, src0, src1);
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-sum * 8));
                    }
                }
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t row = rowBase + r;
            uint8_t* rowDst = dst + row * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = rowDst + super * MNNIme2Q8HpBlockSize();
                std::memset(block, 0, MNNIme2Q8HpBlockSize());
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    const int8_t* srcBk = src + bk * srcBlockStride + row * GEMM_INT8_SRC_UNIT;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
                    const int8_t* src0 = srcBk + (kb * 2) * srcDepthStride;
                    const int8_t* src1 = src0 + srcDepthStride;
                    const int32_t sum = MNNIme2Copy32AndSum(q, src0, src1);
                    MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(-sum * 8));
                }
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() - sizeof(uint16_t), 1.0f);
            }
        }
        rowBase += rows;
    }
}

static void MNNPackIme2AMAllBlocksHpTiledRange(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                               size_t realCount, size_t tileRows, size_t rowBegin, size_t rowEnd,
                                               float* srcKernelSum = nullptr, const float* inputScale = nullptr) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t fullTileStride = blockNum * srcDepthQuad * tileRows * GEMM_INT8_SRC_UNIT;
    MNN_ASSERT(kBlocks % 8 == 0);
    MNN_ASSERT(rowBegin % 4 == 0);
    rowEnd = std::min(rowEnd, realCount);
    for (size_t rowBase = rowBegin; rowBase < rowEnd;) {
        const size_t rows = std::min<size_t>(4, rowEnd - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
                std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t row = rowBase + r;
                        const size_t tile = row / tileRows;
                        const size_t localRow = row - tile * tileRows;
                        const size_t rowsInTile = std::min(tileRows, realCount - tile * tileRows);
                        const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * rowsInTile;
                        const int8_t* srcTile = src + tile * fullTileStride;
                        const int8_t* srcBk =
                            srcTile + bk * srcDepthQuad * srcDepthStride + localRow * GEMM_INT8_SRC_UNIT;
                        const int8_t* src0 = srcBk + (kb * 2) * srcDepthStride;
                        const int8_t* src1 = src0 + srcDepthStride;
                        const int32_t sum = MNNIme2Copy32AndSum(q + r * 32, src0, src1);
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-sum * 8));
                        if (srcKernelSum != nullptr) {
                            float* dstSum = srcKernelSum + bk * realCount + row;
                            if (kb == 0) {
                                *dstSum = 0.0f;
                            }
                            const float scale = inputScale == nullptr ? 1.0f : inputScale[row];
                            *dstSum += static_cast<float>(sum) * scale;
                        }
                    }
                }
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t row = rowBase + r;
            const size_t tile = row / tileRows;
            const size_t localRow = row - tile * tileRows;
            const size_t rowsInTile = std::min(tileRows, realCount - tile * tileRows);
            const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * rowsInTile;
            const int8_t* srcTile = src + tile * fullTileStride;
            uint8_t* rowDst = dst + row * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = rowDst + super * MNNIme2Q8HpBlockSize();
                std::memset(block, 0, MNNIme2Q8HpBlockSize());
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    const int8_t* srcBk = srcTile + bk * srcDepthQuad * srcDepthStride + localRow * GEMM_INT8_SRC_UNIT;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
                    const int8_t* src0 = srcBk + (kb * 2) * srcDepthStride;
                    const int8_t* src1 = src0 + srcDepthStride;
                    const int32_t sum = MNNIme2Copy32AndSum(q, src0, src1);
                    MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(-sum * 8));
                    if (srcKernelSum != nullptr) {
                        float* dstSum = srcKernelSum + bk * realCount + row;
                        if (kb == 0) {
                            *dstSum = 0.0f;
                        }
                        const float scale = inputScale == nullptr ? 1.0f : inputScale[row];
                        *dstSum += static_cast<float>(sum) * scale;
                    }
                }
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() - sizeof(uint16_t), 1.0f);
            }
        }
        rowBase += rows;
    }
}

static void MNNPackIme2AMAllBlocksHpTiled(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                          size_t realCount, size_t tileRows) {
    MNNPackIme2AMAllBlocksHpTiledRange(dst, src, srcDepthQuad, blockNum, realCount, tileRows, 0, realCount);
}

static inline int MNNIme2QuantizeQ8ToQ4(int value, float scale) {
    int q = 0;
    if (scale > 0.0f && std::isfinite(scale)) {
        q = static_cast<int>(std::round(static_cast<float>(value) / scale));
    }
    return std::min(7, std::max(-8, q));
}

static inline int MNNIme2QuantizeQ8ToQ4Fixed16(int value) {
    int q = value >= 0 ? (value + 8) / 16 : -((-value + 8) / 16);
    return std::min(7, std::max(-8, q));
}

static inline int8_t MNNIme2PackQ4High(int q) {
    return static_cast<int8_t>(static_cast<uint8_t>(q & 0x0f) << 4);
}

static void MNNPackIme2AMAllBlocksHpA4(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                       size_t realCount, const float* inputScale, float* srcKernelSum) {
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t srcBlockStride = srcDepthQuad * GEMM_INT8_SRC_UNIT * realCount;
    const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * realCount;
    const bool dynamicScale = MNNSpacemitIme2W4A4DynamicEnabled();
    MNN_ASSERT(kBlocks % 8 == 0);
    MNN_ASSERT(realCount % 4 == 0);
    if (srcKernelSum != nullptr) {
        std::fill(srcKernelSum, srcKernelSum + blockNum * realCount, 0.0f);
    }
    for (size_t rowBase = 0; rowBase < realCount; rowBase += 4) {
        uint8_t* tileDst = dst + rowBase * hpRowStride;
        for (size_t super = 0; super < superBlocks; ++super) {
            uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
            std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
            uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
            for (size_t sub = 0; sub < 8; ++sub) {
                const size_t linearKb = super * 8 + sub;
                const size_t bk = linearKb / kBlocksPerBlock;
                const size_t kb = linearKb - bk * kBlocksPerBlock;
                const int8_t* srcBk = src + bk * srcBlockStride;
                uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                if (!dynamicScale) {
                    for (size_t r = 0; r < 4; ++r) {
                        MNNIme2StoreHalf(subBlock + r * sizeof(uint16_t), 16.0f);
                    }
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        int32_t qSum = 0;
                        const int8_t* srcRow = srcBk + (rowBase + r) * GEMM_INT8_UNIT;
                        for (int kk = 0; kk < 32; ++kk) {
                            const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                            const int k = kk % GEMM_INT8_SRC_UNIT;
                            const int value = static_cast<int>(srcRow[sz * srcDepthStride + k]);
                            const int qValue = MNNIme2QuantizeQ8ToQ4Fixed16(value);
                            qSum += qValue;
                            q[r * 32 + kk] = MNNIme2PackQ4High(qValue);
                        }
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-qSum * 8));
                        if (srcKernelSum != nullptr) {
                            const size_t row = rowBase + r;
                            const float srcScale = inputScale == nullptr ? 1.0f : inputScale[row];
                            srcKernelSum[bk * realCount + row] += static_cast<float>(qSum) * 16.0f * srcScale;
                        }
                    }
                    continue;
                }
                int maxAbs = 0;
                for (size_t r = 0; r < 4; ++r) {
                    const int8_t* srcRow = srcBk + (rowBase + r) * GEMM_INT8_UNIT;
                    for (int kk = 0; kk < 32; ++kk) {
                        const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                        const int k = kk % GEMM_INT8_SRC_UNIT;
                        const int value = static_cast<int>(srcRow[sz * srcDepthStride + k]);
                        maxAbs = std::max(maxAbs, std::abs(value));
                    }
                }
                const float a4Scale = maxAbs <= 7 ? 1.0f : static_cast<float>(maxAbs) / 7.0f;
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(subBlock + r * sizeof(uint16_t), a4Scale);
                }
                int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                for (size_t r = 0; r < 4; ++r) {
                    const int8_t* srcRow = srcBk + (rowBase + r) * GEMM_INT8_UNIT;
                    int32_t qSum = 0;
                    for (int kk = 0; kk < 32; ++kk) {
                        const size_t sz = kb * 2 + kk / GEMM_INT8_SRC_UNIT;
                        const int k = kk % GEMM_INT8_SRC_UNIT;
                        const int value = static_cast<int>(srcRow[sz * srcDepthStride + k]);
                        const int qValue = MNNIme2QuantizeQ8ToQ4(value, a4Scale);
                        qSum += qValue;
                        q[r * 32 + kk] = MNNIme2PackQ4High(qValue);
                    }
                    MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t),
                                     static_cast<float>(-static_cast<float>(qSum) * 8.0f));
                    if (srcKernelSum != nullptr) {
                        const size_t row = rowBase + r;
                        const float srcScale = inputScale == nullptr ? 1.0f : inputScale[row];
                        srcKernelSum[bk * realCount + row] += static_cast<float>(qSum) * a4Scale * srcScale;
                    }
                }
            }
            for (size_t r = 0; r < 4; ++r) {
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
            }
        }
    }
}

static void MNNPackIme2B32(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum, size_t bk,
                           size_t dzStart) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocks = srcDepthQuad / 2;
    const size_t bBlockSize = MNNIme2Q4BlockSize();

    for (size_t kb = 0; kb < kBlocks; ++kb) {
        uint8_t* block = dst + kb * bBlockSize;
        uint16_t* scales = reinterpret_cast<uint16_t*>(block);
        uint8_t* qs = block + sizeof(uint16_t) * 32;

        for (int dz = 0; dz < 8; ++dz) {
            const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
            const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);

            for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                const int col = dz * GEMM_INT8_UNIT + j;
                const float scale = scaleDz[j];
                scales[col] = MNNFloatToHalfBits(scale);

                for (int packed = 0; packed < 16; ++packed) {
                    const int kk0 = packed * 2;
                    const int kk1 = kk0 + 1;
                    const uint8_t* weight0 =
                        reinterpret_cast<const uint8_t*>(weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                    const uint8_t* weight1 =
                        reinterpret_cast<const uint8_t*>(weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                    const uint8_t q0 = MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT);
                    const uint8_t q1 = MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT);
                    qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                }
            }
        }
    }
}

static void MNNPackIme2B32AllBlocks(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum,
                                    size_t dzStart, bool useZp) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t bBlockSize = MNNIme2Q4BlockSize(useZp);

    for (size_t bk = 0; bk < blockNum; ++bk) {
        for (size_t kb = 0; kb < kBlocksPerBlock; ++kb) {
            uint8_t* block = dst + (bk * kBlocksPerBlock + kb) * bBlockSize;
            uint16_t* scales = reinterpret_cast<uint16_t*>(block);
            uint8_t* zps = useZp ? block + sizeof(uint16_t) * 32 : nullptr;
            uint8_t* qs = block + sizeof(uint16_t) * 32 + (useZp ? 32 : 0);

            for (int dz = 0; dz < 8; ++dz) {
                const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const int col = dz * GEMM_INT8_UNIT + j;
                    scales[col] = MNNFloatToHalfBits(scaleDz[j]);
                    if (useZp) {
                        zps[col] = MNNIme2ChooseZp(scaleDz[j], weightBiasDz[j]);
                    }

                    for (int packed = 0; packed < 16; ++packed) {
                        const int kk0 = packed * 2;
                        const int kk1 = kk0 + 1;
                        const uint8_t* weight0 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                        const uint8_t* weight1 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                        const uint8_t q0 = MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT);
                        const uint8_t q1 = MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT);
                        qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                    }
                }
            }
        }
    }
}

static void MNNPackIme2B32HpAllBlocks(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum,
                                      size_t dzStart, bool useZp) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t subBlockSize = MNNIme2Q4BlockSize(false);
    const size_t superBlockSize = MNNIme2Q4HpSuperBlockSize(useZp);
    MNN_ASSERT(kBlocks % 8 == 0);

    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* superBlock = dst + super * superBlockSize;
        for (size_t sub = 0; sub < 8; ++sub) {
            const size_t linearKb = super * 8 + sub;
            const size_t bk = linearKb / kBlocksPerBlock;
            const size_t kb = linearKb - bk * kBlocksPerBlock;
            uint8_t* block = superBlock + sub * subBlockSize;
            uint16_t* scales = reinterpret_cast<uint16_t*>(block);
            uint8_t* qs = block + sizeof(uint16_t) * 32;
            uint8_t* zps = useZp ? superBlock + subBlockSize * 8 + sub * 32 : nullptr;

            for (int dz = 0; dz < 8; ++dz) {
                const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const int col = dz * GEMM_INT8_UNIT + j;
                    scales[col] = MNNFloatToHalfBits(scaleDz[j]);
                    if (zps != nullptr) {
                        zps[col] = MNNIme2ChooseZp(scaleDz[j], weightBiasDz[j]);
                    }

                    for (int packed = 0; packed < 16; ++packed) {
                        const int kk0 = packed * 2;
                        const int kk1 = kk0 + 1;
                        const uint8_t* weight0 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                        const uint8_t* weight1 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                        const uint8_t q0 = MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT);
                        const uint8_t q1 = MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT);
                        qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                    }
                }
            }
        }
    }
}

static void MNNPackIme2B32HpResidualAllBlocks(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum,
                                              size_t dzStart) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t subBlockSize = MNNIme2Q4BlockSize(false);
    const size_t residualSubBytes = sizeof(uint16_t) * 32;
    const size_t fusedSubBlockSize = subBlockSize + residualSubBytes;
    const size_t superBlockSize = MNNIme2Q4HpResidualSuperBlockSize();
    MNN_ASSERT(kBlocks % 8 == 0);

    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* superBlock = dst + super * superBlockSize;
        for (size_t sub = 0; sub < 8; ++sub) {
            const size_t linearKb = super * 8 + sub;
            const size_t bk = linearKb / kBlocksPerBlock;
            const size_t kb = linearKb - bk * kBlocksPerBlock;
            uint8_t* block = superBlock + sub * fusedSubBlockSize;
            uint16_t* scales = reinterpret_cast<uint16_t*>(block);
            uint8_t* qs = block + sizeof(uint16_t) * 32;
            uint16_t* residualScales = reinterpret_cast<uint16_t*>(qs + 32 * 32 / 2);

            for (int dz = 0; dz < 8; ++dz) {
                const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const int col = dz * GEMM_INT8_UNIT + j;
                    const float scale = scaleDz[j];
                    scales[col] = MNNFloatToHalfBits(scale);
                    residualScales[col] = MNNFloatToHalfBits(-(weightBiasDz[j] + 8.0f * scale) * 0.125f);

                    for (int packed = 0; packed < 16; ++packed) {
                        const int kk0 = packed * 2;
                        const int kk1 = kk0 + 1;
                        const uint8_t* weight0 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                        const uint8_t* weight1 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                        const uint8_t q0 = MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT);
                        const uint8_t q1 = MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT);
                        qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                    }
                }
            }
        }
    }
}

static void MNNPackIme2B32HpAsymPairAllBlocks(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum,
                                              size_t dzStart) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    constexpr size_t pairCount = 4;
    constexpr size_t scaleBytes = sizeof(uint16_t) * 32;
    constexpr size_t qBytes = 32 * 32 / 2;
    constexpr size_t pairBytes = scaleBytes * 2 + qBytes * 2;
    const size_t superBlockSize = MNNIme2Q4HpAsymPairSuperBlockSize();
    MNN_ASSERT(srcDepthQuad == 4);
    MNN_ASSERT(kBlocksPerBlock == 2);
    MNN_ASSERT(kBlocks % 8 == 0);
    MNN_ASSERT(pairCount * pairBytes == superBlockSize);

    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* superBlock = dst + super * superBlockSize;
        for (size_t pair = 0; pair < pairCount; ++pair) {
            const size_t bk = super * pairCount + pair;
            uint8_t* pairBlock = superBlock + pair * pairBytes;
            uint16_t* scales = reinterpret_cast<uint16_t*>(pairBlock);
            uint16_t* centeredCorrection = reinterpret_cast<uint16_t*>(pairBlock + scaleBytes);
            uint8_t* qs0 = pairBlock + scaleBytes * 2;
            uint8_t* qs1 = qs0 + qBytes;

            for (int dz = 0; dz < 8; ++dz) {
                const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const int col = dz * GEMM_INT8_UNIT + j;
                    const float scale = scaleDz[j];
                    scales[col] = MNNFloatToHalfBits(scale);
                    // aSum is -8 * sum(qA). Multiplying it by -weightBias / 8 supplies
                    // sum(qA) * weightBias and avoids a second rounded FP16 addition in the kernel.
                    centeredCorrection[col] = MNNFloatToHalfBits(-weightBiasDz[j] * 0.125f);

                    for (int kb = 0; kb < 2; ++kb) {
                        uint8_t* qs = kb == 0 ? qs0 : qs1;
                        for (int packed = 0; packed < 16; ++packed) {
                            const int kk0 = packed * 2;
                            const int kk1 = kk0 + 1;
                            const uint8_t* weight0 = reinterpret_cast<const uint8_t*>(
                                weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                            const uint8_t* weight1 = reinterpret_cast<const uint8_t*>(
                                weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                            const uint8_t q0 = MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT);
                            const uint8_t q1 = MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT);
                            qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                        }
                    }
                }
            }
        }
    }
}

static void MNNPackIme2B32HpSymAllBlocks(uint8_t* dst, const int8_t* weight, size_t srcDepthQuad, size_t blockNum,
                                         size_t dzStart) {
    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t superBlocks = kBlocks / 8;
    const size_t subBlockSize = MNNIme2Q4BlockSize(false);
    const size_t superBlockSize = MNNIme2Q4HpSuperBlockSize(false);
    MNN_ASSERT(kBlocks % 8 == 0);

    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* superBlock = dst + super * superBlockSize;
        for (size_t sub = 0; sub < 8; ++sub) {
            const size_t linearKb = super * 8 + sub;
            const size_t bk = linearKb / kBlocksPerBlock;
            const size_t kb = linearKb - bk * kBlocksPerBlock;
            uint8_t* block = superBlock + sub * subBlockSize;
            uint16_t* scales = reinterpret_cast<uint16_t*>(block);
            uint8_t* qs = block + sizeof(uint16_t) * 32;

            for (int dz = 0; dz < 8; ++dz) {
                const int8_t* weightDz = weight + (dzStart + dz) * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const int col = dz * GEMM_INT8_UNIT + j;
                    const float symScale =
                        MNNIme2SymW4Scale(weightDz, srcDepthQuad, weightStepY, j, scaleDz[j], weightBiasDz[j]);
                    scales[col] = MNNFloatToHalfBits(symScale);

                    for (int packed = 0; packed < 16; ++packed) {
                        const int kk0 = packed * 2;
                        const int kk1 = kk0 + 1;
                        const uint8_t* weight0 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk0 / GEMM_INT8_SRC_UNIT));
                        const uint8_t* weight1 = reinterpret_cast<const uint8_t*>(
                            weightDz + weightStepY * (kb * 2 + kk1 / GEMM_INT8_SRC_UNIT));
                        const float value0 =
                            static_cast<float>(MNNReadW4(weight0, j, kk0 % GEMM_INT8_SRC_UNIT)) * scaleDz[j] +
                            weightBiasDz[j];
                        const float value1 =
                            static_cast<float>(MNNReadW4(weight1, j, kk1 % GEMM_INT8_SRC_UNIT)) * scaleDz[j] +
                            weightBiasDz[j];
                        const uint8_t q0 = MNNIme2QuantizeSymW4(value0, symScale);
                        const uint8_t q1 = MNNIme2QuantizeSymW4(value1, symScale);
                        qs[col * 16 + packed] = static_cast<uint8_t>(q0 | (q1 << 4));
                    }
                }
            }
        }
    }
}

struct MNNIme2BPackKey {
    size_t srcDepthQuad;
    size_t dstDepthQuad;
    size_t blockNum;
    bool useZp;
    bool hp;

    bool operator==(const MNNIme2BPackKey& other) const {
        return srcDepthQuad == other.srcDepthQuad && dstDepthQuad == other.dstDepthQuad && blockNum == other.blockNum &&
               useZp == other.useZp && hp == other.hp;
    }
};

struct MNNIme2BPackKeyHash {
    size_t operator()(const MNNIme2BPackKey& key) const {
        size_t h = std::hash<size_t>()(key.srcDepthQuad);
        h ^= std::hash<size_t>()(key.dstDepthQuad + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<size_t>()(key.blockNum + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<bool>()(key.useZp);
        h ^= std::hash<bool>()(key.hp);
        return h;
    }
};

template <typename T>
using MNNIme2BPackCache = std::unordered_map<MNNIme2BPackKey, std::shared_ptr<std::vector<T>>, MNNIme2BPackKeyHash>;

struct MNNIme2Residual {
    std::vector<float> values;
    bool allZero = false;
};

using MNNIme2ResidualCache = std::unordered_map<MNNIme2BPackKey, std::shared_ptr<MNNIme2Residual>, MNNIme2BPackKeyHash>;

struct MNNIme2LinearResource {
    std::mutex cacheMutex;
    // The generic executor binds one immutable STATIC weight before publishing the Execution. The executor's task
    // synchronization publishes the populated weight; this atomic only makes an accidental concurrent duplicate bind
    // well-defined, and a different weight is rejected instead of rebinding.
    std::atomic<uintptr_t> weightAddress{0};

    MNNIme2BPackCache<uint8_t> packedB;
    MNNIme2BPackCache<uint8_t> packedBAllBlocks;
    MNNIme2BPackCache<uint8_t> packedBHpAllBlocks;
    MNNIme2BPackCache<uint8_t> packedBHpResidualAllBlocks;
    MNNIme2BPackCache<uint8_t> packedBHpAsymPairAllBlocks;
    MNNIme2BPackCache<uint8_t> packedBHpSymAllBlocks;
    MNNIme2ResidualCache residualAllBlocks;
    MNNIme2ResidualCache weightBiasResidualAllBlocks;
};

using MNNIme2LinearResourceHandle = std::shared_ptr<MNNIme2LinearResource>;

struct MNNIme2ResolvedLinearResource {
    MNNIme2LinearResourceHandle resource;
};

static MNNIme2ResolvedLinearResource MNNIme2ResolveLinearResource(void* context, const int8_t* weight) {
    MNNIme2ResolvedLinearResource result;
    if (weight == nullptr) {
        return result;
    }
    if (context == nullptr) {
        return result;
    }
    auto handle = static_cast<MNNIme2LinearResourceHandle*>(context);
    result.resource = *handle;
    if (result.resource != nullptr &&
        result.resource->weightAddress.load(std::memory_order_relaxed) == reinterpret_cast<uintptr_t>(weight)) {
        return result;
    }
    result.resource.reset();
    return result;
}

template <int Layout, typename T>
struct MNNIme2LocalLinearCacheEntry {
    std::weak_ptr<MNNIme2LinearResource> owner;
    // Safe while resource caches never erase, replace, or clear values. Reintroduce value ownership if eviction is
    // added in the future.
    T* value = nullptr;
    MNNIme2BPackKey key = {};
};

template <int Layout, typename T>
using MNNIme2LocalLinearCache =
    std::unordered_map<const MNNIme2LinearResource*, MNNIme2LocalLinearCacheEntry<Layout, T>>;

template <int Layout, typename T>
static MNNIme2LocalLinearCache<Layout, T>& MNNIme2GetLocalLinearCache() {
    static thread_local MNNIme2LocalLinearCache<Layout, T> cache;
    return cache;
}

template <int Layout, typename T>
static std::shared_ptr<T> MNNIme2FindLocalLinearCache(const MNNIme2ResolvedLinearResource& resolved,
                                                      const MNNIme2BPackKey& key) {
    if (resolved.resource == nullptr) {
        return nullptr;
    }
    auto& cache = MNNIme2GetLocalLinearCache<Layout, T>();
    auto iter = cache.find(resolved.resource.get());
    if (iter == cache.end() || !(iter->second.key == key)) {
        return nullptr;
    }
    if (iter->second.owner.owner_before(resolved.resource) || resolved.resource.owner_before(iter->second.owner)) {
        cache.erase(iter);
        return nullptr;
    }
    if (iter->second.value == nullptr) {
        cache.erase(iter);
        return nullptr;
    }
    return std::shared_ptr<T>(resolved.resource, iter->second.value);
}

template <int Layout, typename T>
static void MNNIme2StoreLocalLinearCache(const MNNIme2ResolvedLinearResource& resolved, const MNNIme2BPackKey& key,
                                         const std::shared_ptr<T>& value) {
    if (resolved.resource == nullptr || value == nullptr) {
        return;
    }
    using Entry = MNNIme2LocalLinearCacheEntry<Layout, T>;
    auto& cache = MNNIme2GetLocalLinearCache<Layout, T>();
    constexpr size_t limit = 1024;
    if (cache.size() >= limit && cache.find(resolved.resource.get()) == cache.end()) {
        cache.clear();
    }
    Entry entry;
    entry.owner = resolved.resource;
    entry.value = value.get();
    entry.key = key;
    cache[resolved.resource.get()] = std::move(entry);
}

template <typename T, typename Creator>
static std::shared_ptr<std::vector<T>>
MNNIme2GetOrCreateLinearCache(const MNNIme2ResolvedLinearResource& resolved, const MNNIme2BPackKey& key,
                              MNNIme2BPackCache<T> MNNIme2LinearResource::*cacheMember, Creator creator) {
    if (resolved.resource == nullptr) {
        return creator();
    }
    std::lock_guard<std::mutex> lock(resolved.resource->cacheMutex);
    auto& cache = resolved.resource.get()->*cacheMember;
    auto iter = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }
    auto value = creator();
    cache.emplace(key, value);
    return value;
}

template <typename Creator>
static std::shared_ptr<MNNIme2Residual>
MNNIme2GetOrCreateLinearResidualCache(const MNNIme2ResolvedLinearResource& resolved, const MNNIme2BPackKey& key,
                                      MNNIme2ResidualCache MNNIme2LinearResource::*cacheMember, Creator creator) {
    if (resolved.resource == nullptr) {
        return creator();
    }
    std::lock_guard<std::mutex> lock(resolved.resource->cacheMutex);
    auto& cache = resolved.resource.get()->*cacheMember;
    auto iter = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }
    auto value = creator();
    cache.emplace(key, value);
    return value;
}

struct MNNIme2APackKey {
    uintptr_t src;
    uint64_t hash;
    size_t srcDepthQuad;
    size_t blockNum;
    size_t realCount;

    bool operator==(const MNNIme2APackKey& other) const {
        return src == other.src && hash == other.hash && srcDepthQuad == other.srcDepthQuad &&
               blockNum == other.blockNum && realCount == other.realCount;
    }
};

struct MNNIme2APackKeyHash {
    size_t operator()(const MNNIme2APackKey& key) const {
        size_t h = std::hash<uintptr_t>()(key.src);
        h ^= std::hash<uint64_t>()(key.hash + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<size_t>()(key.srcDepthQuad + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<size_t>()(key.blockNum + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<size_t>()(key.realCount + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        return h;
    }
};

static uint64_t MNNIme2HashAContent(const int8_t* src, size_t srcDepthQuad, size_t blockNum, size_t realCount) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(src);
    const size_t bytes = srcDepthQuad * GEMM_INT8_SRC_UNIT * blockNum * realCount;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<uint64_t>(ptr[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

using MNNIme2APackCacheMap =
    std::unordered_map<MNNIme2APackKey, std::shared_ptr<std::vector<uint8_t>>, MNNIme2APackKeyHash>;

static std::mutex& MNNIme2APackCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

static MNNIme2APackCacheMap& MNNIme2APackCache() {
    static MNNIme2APackCacheMap cache;
    return cache;
}

static void MNNIme2ClearPackedACache() {
    std::lock_guard<std::mutex> lock(MNNIme2APackCacheMutex());
    MNNIme2APackCache().clear();
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedAHpAllBlocks(const int8_t* src, size_t srcDepthQuad,
                                                                          size_t blockNum, size_t realCount,
                                                                          size_t packedBytes) {
    const MNNIme2APackKey key = {
        reinterpret_cast<uintptr_t>(src),
        MNNSpacemitIme2ACacheFastEnabled() ? 0 : MNNIme2HashAContent(src, srcDepthQuad, blockNum, realCount),
        srcDepthQuad, blockNum, realCount};
    std::lock_guard<std::mutex> lock(MNNIme2APackCacheMutex());
    auto& cache = MNNIme2APackCache();
    auto iter = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }

    constexpr size_t limit = 256;
    if (cache.size() >= limit) {
        cache.clear();
    }
    auto packed = std::make_shared<std::vector<uint8_t>>();
    packed->resize(packedBytes);
    MNNPackIme2AMAllBlocksHp(packed->data(), src, srcDepthQuad, blockNum, realCount);
    cache.emplace(key, packed);
    return packed;
}

static void MNNIme2PrepackPackedA(const int8_t* src, size_t srcDepthQuad, size_t blockNum, size_t realCount) {
    if (!MNNSpacemitIme2UseACacheEnabled() || src == nullptr || realCount <= 1) {
        return;
    }
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (srcDepthQuad % 2 != 0 || kBlocks == 0 || kBlocks % 8 != 0) {
        return;
    }
    const size_t superBlocks = kBlocks / 8;
    const size_t packedBytes = MNNIme2Q8HpBlockSize() * superBlocks * realCount;
    (void)MNNGetIme2PackedAHpAllBlocks(src, srcDepthQuad, blockNum, realCount, packedBytes);
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedB(const int8_t* weight, size_t srcDepthQuad,
                                                               size_t dstDepthQuad, size_t blockNum,
                                                               void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, false, false};
    return MNNIme2GetOrCreateLinearCache<uint8_t>(resolved, key, &MNNIme2LinearResource::packedB, [=]() {
        const size_t bStride = MNNIme2Q4BlockSize() * (srcDepthQuad / 2);
        const size_t groupCount = dstDepthQuad / 8;
        auto packed = std::make_shared<std::vector<uint8_t>>();
        packed->resize(groupCount * blockNum * bStride);
        for (size_t bk = 0; bk < blockNum; ++bk) {
            for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
                auto dst = packed->data() + (bk * groupCount + dzGroup / 8) * bStride;
                MNNPackIme2B32(dst, weight, srcDepthQuad, blockNum, bk, dzGroup);
            }
        }
        return packed;
    });
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedBAllBlocks(const int8_t* weight, size_t srcDepthQuad,
                                                                        size_t dstDepthQuad, size_t blockNum,
                                                                        bool useZp, void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, useZp, false};
    return MNNIme2GetOrCreateLinearCache<uint8_t>(resolved, key, &MNNIme2LinearResource::packedBAllBlocks, [=]() {
        const size_t bStride = MNNIme2Q4BlockSize(useZp) * (srcDepthQuad / 2) * blockNum;
        const size_t groupCount = dstDepthQuad / 8;
        auto packed = std::make_shared<std::vector<uint8_t>>();
        packed->resize(groupCount * bStride);
        for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
            auto dst = packed->data() + (dzGroup / 8) * bStride;
            MNNPackIme2B32AllBlocks(dst, weight, srcDepthQuad, blockNum, dzGroup, useZp);
        }
        return packed;
    });
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedBHpAllBlocks(const int8_t* weight, size_t srcDepthQuad,
                                                                          size_t dstDepthQuad, size_t blockNum,
                                                                          bool useZp, void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, useZp, true};
    return MNNIme2GetOrCreateLinearCache<uint8_t>(resolved, key, &MNNIme2LinearResource::packedBHpAllBlocks, [=]() {
        const size_t kBlocksPerBlock = srcDepthQuad / 2;
        const size_t superBlocks = (kBlocksPerBlock * blockNum) / 8;
        const size_t bStride = MNNIme2Q4HpSuperBlockSize(useZp) * superBlocks;
        const size_t groupCount = dstDepthQuad / 8;
        auto packed = std::make_shared<std::vector<uint8_t>>();
        packed->resize(groupCount * bStride);
        for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
            auto dst = packed->data() + (dzGroup / 8) * bStride;
            MNNPackIme2B32HpAllBlocks(dst, weight, srcDepthQuad, blockNum, dzGroup, useZp);
        }
        return packed;
    });
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedBHpResidualAllBlocks(const int8_t* weight,
                                                                                  size_t srcDepthQuad,
                                                                                  size_t dstDepthQuad, size_t blockNum,
                                                                                  void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, false, true};
    return MNNIme2GetOrCreateLinearCache<uint8_t>(
        resolved, key, &MNNIme2LinearResource::packedBHpResidualAllBlocks, [=]() {
            const size_t kBlocksPerBlock = srcDepthQuad / 2;
            const size_t superBlocks = (kBlocksPerBlock * blockNum) / 8;
            const size_t bStride = MNNIme2Q4HpResidualSuperBlockSize() * superBlocks;
            const size_t groupCount = dstDepthQuad / 8;
            auto packed = std::make_shared<std::vector<uint8_t>>();
            packed->resize(groupCount * bStride);
            for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
                auto dst = packed->data() + (dzGroup / 8) * bStride;
                MNNPackIme2B32HpResidualAllBlocks(dst, weight, srcDepthQuad, blockNum, dzGroup);
            }
            return packed;
        });
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedBHpAsymPairAllBlocks(const int8_t* weight,
                                                                                  size_t srcDepthQuad,
                                                                                  size_t dstDepthQuad, size_t blockNum,
                                                                                  void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, false, true};
    auto local = MNNIme2FindLocalLinearCache<1, std::vector<uint8_t>>(resolved, key);
    if (local != nullptr) {
        return local;
    }
    auto value = MNNIme2GetOrCreateLinearCache<uint8_t>(
        resolved, key, &MNNIme2LinearResource::packedBHpAsymPairAllBlocks, [=]() {
            const size_t kBlocks = (srcDepthQuad / 2) * blockNum;
            const size_t superBlocks = kBlocks / 8;
            const size_t bStride = MNNIme2Q4HpAsymPairSuperBlockSize() * superBlocks;
            const size_t groupCount = dstDepthQuad / 8;
            auto packed = std::make_shared<std::vector<uint8_t>>();
            packed->resize(groupCount * bStride);
            for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
                auto dst = packed->data() + (dzGroup / 8) * bStride;
                MNNPackIme2B32HpAsymPairAllBlocks(dst, weight, srcDepthQuad, blockNum, dzGroup);
            }
            return packed;
        });
    MNNIme2StoreLocalLinearCache<1, std::vector<uint8_t>>(resolved, key, value);
    return value;
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetIme2PackedBHpSymAllBlocks(const int8_t* weight, size_t srcDepthQuad,
                                                                             size_t dstDepthQuad, size_t blockNum,
                                                                             void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, false, true};
    return MNNIme2GetOrCreateLinearCache<uint8_t>(resolved, key, &MNNIme2LinearResource::packedBHpSymAllBlocks, [=]() {
        const size_t kBlocksPerBlock = srcDepthQuad / 2;
        const size_t superBlocks = (kBlocksPerBlock * blockNum) / 8;
        const size_t bStride = MNNIme2Q4HpSuperBlockSize(false) * superBlocks;
        const size_t groupCount = dstDepthQuad / 8;
        auto packed = std::make_shared<std::vector<uint8_t>>();
        packed->resize(groupCount * bStride);
        for (size_t dzGroup = 0; dzGroup < dstDepthQuad; dzGroup += 8) {
            auto dst = packed->data() + (dzGroup / 8) * bStride;
            MNNPackIme2B32HpSymAllBlocks(dst, weight, srcDepthQuad, blockNum, dzGroup);
        }
        return packed;
    });
}

static std::shared_ptr<MNNIme2Residual> MNNGetIme2ResidualAllBlocks(const int8_t* weight, size_t srcDepthQuad,
                                                                    size_t dstDepthQuad, size_t blockNum, bool useZp,
                                                                    void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, useZp, false};
    return MNNIme2GetOrCreateLinearResidualCache(resolved, key, &MNNIme2LinearResource::residualAllBlocks, [=]() {
        const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
        const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
        const size_t countN = dstDepthQuad * GEMM_INT8_UNIT;
        auto residual = std::make_shared<MNNIme2Residual>();
        residual->values.resize(blockNum * countN);
        residual->allZero = true;
        for (size_t bk = 0; bk < blockNum; ++bk) {
            for (size_t dz = 0; dz < dstDepthQuad; ++dz) {
                const int8_t* weightDz = weight + dz * blockNum * weightStepZ + bk * weightStepZ;
                const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const float zeroPoint =
                        useZp ? static_cast<float>(MNNIme2ChooseZp(scaleDz[j], weightBiasDz[j])) : 8.0f;
                    const float residualValue = weightBiasDz[j] + zeroPoint * scaleDz[j];
                    residual->values[bk * countN + dz * GEMM_INT8_UNIT + j] = residualValue;
                    residual->allZero = residual->allZero && residualValue == 0.0f;
                }
            }
        }
        return residual;
    });
}

static std::shared_ptr<MNNIme2Residual> MNNGetIme2WeightBiasResidualAllBlocks(const int8_t* weight, size_t srcDepthQuad,
                                                                              size_t dstDepthQuad, size_t blockNum,
                                                                              void* context = nullptr) {
    const auto resolved = MNNIme2ResolveLinearResource(context, weight);
    const MNNIme2BPackKey key = {srcDepthQuad, dstDepthQuad, blockNum, false, true};
    auto local = MNNIme2FindLocalLinearCache<2, MNNIme2Residual>(resolved, key);
    if (local != nullptr) {
        return local;
    }
    auto value = MNNIme2GetOrCreateLinearResidualCache(
        resolved, key, &MNNIme2LinearResource::weightBiasResidualAllBlocks, [=]() {
            const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
            const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
            const size_t countN = dstDepthQuad * GEMM_INT8_UNIT;
            auto residual = std::make_shared<MNNIme2Residual>();
            residual->values.resize(blockNum * countN);
            residual->allZero = true;
            for (size_t bk = 0; bk < blockNum; ++bk) {
                for (size_t dz = 0; dz < dstDepthQuad; ++dz) {
                    const int8_t* weightDz = weight + dz * blockNum * weightStepZ + bk * weightStepZ;
                    const float* scaleDz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weightStepY);
                    const float* weightBiasDz = scaleDz + GEMM_INT8_UNIT;
                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        const float residualValue = weightBiasDz[j];
                        residual->values[bk * countN + dz * GEMM_INT8_UNIT + j] = residualValue;
                        residual->allZero = residual->allZero && residualValue == 0.0f;
                    }
                }
            }
            return residual;
        });
    MNNIme2StoreLocalLinearCache<2, MNNIme2Residual>(resolved, key, value);
    return value;
}

static bool MNNIme2ResidualAllZeroExact(const std::shared_ptr<MNNIme2Residual>& residual) {
    return residual != nullptr && !residual->values.empty() && residual->allZero;
}

static bool MNNIme2ResidualAllZeroCached(const std::shared_ptr<MNNIme2Residual>& residual) {
    if (!MNNSpacemitIme2ZeroResidualSkipEnabled() || residual == nullptr || residual->values.empty()) {
        return false;
    }
    return MNNIme2ResidualAllZeroExact(residual);
}

static bool MNNRvvW4UseCenteredPost(const int8_t* weight, size_t srcDepthQuad, size_t dstDepthQuad, size_t blockNum,
                                    void* context = nullptr) {
    if (!MNNRvvW4CenteredPostEnabled()) {
        return false;
    }
    auto centeredResidual = MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, false, context);
    return MNNIme2ResidualAllZeroCached(centeredResidual);
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_Unit_IME2(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                      size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                      const QuanPostTreatParameters* post, size_t realCount,
                                                      bool directFloatHp = false, void* context = nullptr) {
    if (context == nullptr || !MNNSpacemitIme2Enabled() || post == nullptr || post->useInt8 != 0 ||
        srcDepthQuad % 2 != 0 || dstDepthQuad < 8 ||
        (directFloatHp && (realCount != 1 || post->inputBias != nullptr))) {
        return false;
    }
    const size_t ime2DstDepthQuad = dstDepthQuad & ~static_cast<size_t>(7);
    const size_t tailDstDepthQuad = dstDepthQuad - ime2DstDepthQuad;
    const bool hasTail = tailDstDepthQuad != 0;
    const bool decodeBiasPost = MNNSpacemitIme2UseDecodeBiasPost(post);
    const bool canRunRvvTail = realCount == 1 && post->inputScale != nullptr && post->inputBias == nullptr &&
                               post->biasFloat != nullptr && post->fp32minmax != nullptr;
    if (ime2DstDepthQuad == 0 || (directFloatHp && hasTail) ||
        (hasTail && (decodeBiasPost || !MNNSpacemitIme2DecodeTailEnabled() || !canRunRvvTail))) {
        return false;
    }
    if (realCount > 1 && (post->inputScale == nullptr || post->inputBias != nullptr)) {
        return false;
    }
    const size_t blockNum = post->blockNum;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    const size_t groupCount = ime2DstDepthQuad / 8;
    const size_t countN = ime2DstDepthQuad * GEMM_INT8_UNIT;
    const bool useHpM1AsymPair =
        directFloatHp && srcDepthQuad == 4 && blockNum % 4 == 0 && MNNSpacemitIme2HpM1AsymPairEnabled();
    auto gemm = MNNSpacemitIme2Gemm();
#if defined(MNN_USE_SPACEMIT_IME2)
    if (directFloatHp) {
        gemm = MNNSpacemitIme2GemmI8I4HpM1NativeLocal;
    }
#else
    if (directFloatHp) {
        return false;
    }
#endif
    if (gemm == nullptr) {
        return false;
    }
    const bool useZp = directFloatHp ? false : MNNSpacemitIme2ZpEnabled();
    bool useHp = directFloatHp && kBlocks % 8 == 0;
    if (!directFloatHp && MNNSpacemitIme2HpEnabled() && post->inputScale != nullptr && post->inputBias == nullptr &&
        kBlocks % 8 == 0) {
        useHp = (realCount >= 4 && MNNSpacemitIme2HpM4Enabled()) || (realCount == 1 && MNNSpacemitIme2HpM1Enabled());
    }
    if (directFloatHp && !useHp) {
        return false;
    }
    if (useHp && !directFloatHp) {
        const size_t hpMinCountM = realCount == 1 ? 1 : 4;
        constexpr size_t hpMinCountN = 0;
        constexpr size_t hpMinKBlocks = 1;
        useHp = realCount >= hpMinCountM && countN >= hpMinCountN && kBlocks / 8 >= hpMinKBlocks;
    }
    if (useHp && !directFloatHp && realCount == 1 && post->inputScale != nullptr) {
        constexpr float hpM1InputScaleMax = -1.0f;
        if (hpM1InputScaleMax >= 0.0f && std::fabs(post->inputScale[0]) > hpM1InputScaleMax) {
            useHp = false;
        }
    }
    const bool useA4 = MNNSpacemitIme2W4A4Enabled() && useHp && !useZp && realCount >= 4 && realCount % 4 == 0;
    const bool useSymW4 = MNNSpacemitIme2SymW4Enabled() && useHp && !useZp && !useA4 && realCount > 1;
    const bool useFusedResidual =
        MNNSpacemitIme2FuseResidualEnabled() && useHp && !useZp && !useA4 && !useSymW4 && realCount > 1;
    const bool useFixedAScale = MNNSpacemitIme2FixedAScaleEnabled() && useHp && !useZp && !useA4 && !useSymW4 &&
                                !useFusedResidual && realCount >= 4;
    const bool packedAInput = directFloatHp || (MNNSpacemitIme2PackedAInputEnabled() && useHp && !useZp && !useA4 &&
                                                !useSymW4 && !useFusedResidual && realCount > 1);
    const size_t kernelBlkLen =
        useHpM1AsymPair ? 261 : (useA4 ? 257 : (useFusedResidual ? 258 : (useFixedAScale ? 260 : (useHp ? 256 : 32))));
    const bool useHpM1RawDot = useHp && realCount == 1 && !directFloatHp;
    const bool canUseHpM1Centered = useHpM1RawDot && !useZp;
    const size_t kernelKBlocks = useHp ? kBlocks / 8 : kBlocks;
    const size_t a1Stride = (useHp ? MNNIme2Q8HpBlockSize() : MNNIme2Q8BlockSize()) * kernelKBlocks;
    const size_t bStride =
        (useHpM1AsymPair
             ? MNNIme2Q4HpAsymPairSuperBlockSize()
             : (useFusedResidual ? MNNIme2Q4HpResidualSuperBlockSize()
                                 : (useHp ? MNNIme2Q4HpSuperBlockSize(useZp) : MNNIme2Q4BlockSize(useZp)))) *
        kernelKBlocks;
    constexpr size_t minCountM = 1;
    constexpr size_t minCountN = 128;
    constexpr size_t minKBlocks = 2;
    if (!directFloatHp && (realCount < minCountM || countN < minCountN || kernelKBlocks < minKBlocks)) {
        return false;
    }
    thread_local std::vector<uint8_t> aBuffer;
    thread_local std::vector<float> cBuffer;
    thread_local std::vector<float> a4SrcKernelSum;
    if (!packedAInput) {
        aBuffer.resize(a1Stride * realCount);
    }
    const size_t cBufferSize = countN * realCount;
    if (cBuffer.size() < cBufferSize) {
        cBuffer.resize(cBufferSize);
    }
    QuanPostTreatParameters a4Post;
    QuanPostTreatParameters foldedScalePost;
    const QuanPostTreatParameters* postForIme2 = post;
    if (useA4) {
        a4SrcKernelSum.resize(blockNum * realCount);
        a4Post = *post;
        a4Post.srcKernelSum = a4SrcKernelSum.data();
        postForIme2 = &a4Post;
    }
    const bool foldBlockInputScale = !useA4 && useHp && realCount == 1 && !useZp && post->inputScale != nullptr &&
                                     post->inputBias == nullptr && blockNum > 1 &&
                                     MNNSpacemitIme2FoldBlockScaleEnabled() && MNNSpacemitIme2BlockInputScaleEnabled();
    if (foldBlockInputScale) {
        foldedScalePost = *post;
        foldedScalePost.inputScale = nullptr;
        postForIme2 = &foldedScalePost;
    }
    const auto packedB =
        useHpM1AsymPair
            ? MNNGetIme2PackedBHpAsymPairAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, context)
            : (useFusedResidual
                   ? MNNGetIme2PackedBHpResidualAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, context)
                   : (useSymW4
                          ? MNNGetIme2PackedBHpSymAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, context)
                          : (useHp ? MNNGetIme2PackedBHpAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum,
                                                                  useZp, context)
                                   : MNNGetIme2PackedBAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, useZp,
                                                                context))));
    std::shared_ptr<MNNIme2Residual> centeredResidual;
    bool useHpM1Centered = false;
    if (directFloatHp) {
        if (!useHpM1AsymPair) {
            centeredResidual =
                MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, false, context);
            if (!MNNIme2ResidualAllZeroExact(centeredResidual)) {
                return false;
            }
        }
    } else if (canUseHpM1Centered) {
        centeredResidual =
            MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, false, context);
        useHpM1Centered = MNNSpacemitIme2HpM1CenteredEnabled() || MNNIme2ResidualAllZeroCached(centeredResidual);
    }
    const auto residual =
        useHpM1AsymPair
            ? std::shared_ptr<MNNIme2Residual>()
            : ((useHpM1RawDot && !useHpM1Centered)
                   ? MNNGetIme2WeightBiasResidualAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad, blockNum, context)
                   : (centeredResidual != nullptr ? centeredResidual
                                                  : MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, ime2DstDepthQuad,
                                                                                blockNum, useZp, context)));
    const float* residualPtr = residual != nullptr ? residual->values.data() : nullptr;
    const bool skipPostResidual =
        directFloatHp || useHpM1AsymPair || useSymW4 || useFusedResidual ||
        (useHp && !useZp && !useA4 && (realCount > 1 || useHpM1Centered) && MNNIme2ResidualAllZeroCached(residual));
    const size_t callKernelBlkLen = useHpM1Centered ? 259 : kernelBlkLen;

    const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
    float fp32min = 0.0f;
    float fp32max = 0.0f;
    if (post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    const float* biasPtr = post->biasFloat;
    const bool directOutput =
        directFloatHp && MNNSpacemitIme2DecodeDirectOutputEnabled() && postForIme2->inputScale == nullptr &&
        postForIme2->inputBias == nullptr && skipPostResidual &&
        (postForIme2->fp32minmax == nullptr || MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max)) && biasPtr == nullptr;
    float* gemmOutput = directOutput ? reinterpret_cast<float*>(dst) : cBuffer.data();
    std::shared_ptr<std::vector<uint8_t>> cachedA;
    const uint8_t* packedAPtr = aBuffer.data();
    if (packedAInput) {
        packedAPtr = reinterpret_cast<const uint8_t*>(src);
    } else if (useA4) {
        MNNPackIme2AMAllBlocksHpA4(aBuffer.data(), src, srcDepthQuad, blockNum, realCount, post->inputScale,
                                   a4SrcKernelSum.data());
    } else if (useHp && realCount == 1) {
        MNNPackIme2A1AllBlocksHp(aBuffer.data(), src, srcDepthQuad, blockNum,
                                 foldBlockInputScale ? post->inputScale : nullptr, useHpM1Centered);
    } else if (useHp && realCount > MNNSpacemitIme2TileRows() && MNNSpacemitIme2TiledAEnabled() &&
               MNNSpacemitIme2HpTiledPackEnabled() && !MNNSpacemitIme2ExecutorPackAEnabled()) {
        MNNPackIme2AMAllBlocksHpTiled(aBuffer.data(), src, srcDepthQuad, blockNum, realCount,
                                      MNNSpacemitIme2TileRows());
    } else if (useHp && MNNSpacemitIme2UseACacheEnabled() && !useZp && !useSymW4 && !useFusedResidual) {
        cachedA = MNNGetIme2PackedAHpAllBlocks(src, srcDepthQuad, blockNum, realCount, a1Stride * realCount);
        packedAPtr = cachedA->data();
    } else if (useHp) {
        MNNPackIme2AMAllBlocksHp(aBuffer.data(), src, srcDepthQuad, blockNum, realCount);
    } else if (realCount == 1) {
        MNNPackIme2A1AllBlocks(aBuffer.data(), src, srcDepthQuad, blockNum, post);
    } else if (realCount > MNNSpacemitIme2TileRows() && MNNSpacemitIme2TiledAEnabled() &&
               !MNNSpacemitIme2ExecutorPackAEnabled()) {
        MNNPackIme2AMAllBlocksTiled(aBuffer.data(), src, srcDepthQuad, blockNum, realCount, MNNSpacemitIme2TileRows(),
                                    post);
    } else {
        MNNPackIme2AMAllBlocks(aBuffer.data(), src, srcDepthQuad, blockNum, realCount, post);
    }
    size_t handled = 0;
    size_t expected = realCount;
    const int directOuterMode = MNNSpacemitIme2DirectOuterMode();
    // Direct FP32 packing deliberately happens on a VLENB=32 application core, while its native HP GEMM must run
    // on a VLENB=128 AI worker. Do not let the generic DIRECT_OUTER experiment collapse those two stages.
    const bool directOuter = !directFloatHp && (directOuterMode >= 2 || (directOuterMode == 1 && realCount > 1));
    const bool workerPost =
        !directOutput && !directOuter &&
        (MNNSpacemitIme2WorkerPostEnabled() || (realCount == 1 && MNNSpacemitIme2DecodeWorkerPostEnabled()));
    if (directOuter) {
        if (MNNSpacemitIme2DirectOuterBindEnabled()) {
            MNNSpacemitIme2BindCurrentThreadOnce();
        }
        MNNSpacemitIme2Job job;
        job.gemm = gemm;
        job.blkLen = callKernelBlkLen;
        job.a = packedAPtr;
        job.b = packedB->data();
        job.bZp = useZp ? packedB->data() : nullptr;
        job.c = gemmOutput;
        job.countM = realCount;
        job.countN = countN;
        job.kBlks = kernelKBlocks;
        job.ldc = countN;
        handled = MNNSpacemitIme2RunGemmRows(job);
    } else {
        std::unique_lock<std::mutex> serialDispatchLock;
        const bool spinWorker = MNNSpacemitIme2SpinEnabled();
        const bool decodeTcmPipelineEnabled = MNNSpacemitIme2TcmDecodePipelineEnabled();
        // The copy/compute barriers only amortize once packed-B exceeds the cache-friendly working set.
        constexpr size_t decodeTcmMinPackedBBytes = 2 * 1024 * 1024;
        const bool decodeTcmPipelineRequested = decodeTcmPipelineEnabled && spinWorker && useHpM1AsymPair &&
                                                kernelKBlocks >= 8 && bStride * groupCount >= decodeTcmMinPackedBBytes;
        if (MNNSpacemitIme2SerialDispatchEnabled() || decodeTcmPipelineRequested) {
            serialDispatchLock = std::unique_lock<std::mutex>(MNNSpacemitIme2DispatchMutex());
        }
        size_t workerCount =
            std::min(spinWorker ? MNNSpacemitIme2SpinWorker::count() : MNNSpacemitIme2Worker::count(), groupCount);
        if (realCount > 1 && MNNSpacemitIme2OuterTileParallel() && MNNSpacemitIme2OuterTileSingleWorkerEnabled()) {
            workerCount = 1;
        }
        if (realCount == 1) {
            constexpr size_t decodeWorkers = 6;
            workerCount = std::min(workerCount, std::max<size_t>(decodeWorkers, 1));
        }
        constexpr size_t smallN = 384;
        if (smallN > 0 && countN <= smallN) {
            constexpr size_t smallWorkers = 1;
            workerCount = std::min(workerCount, std::max<size_t>(1, smallWorkers));
        }
        constexpr size_t worker2DMinM = 16;
        const bool split2DWorkers = !workerPost && useHp && !useA4 && realCount >= worker2DMinM &&
                                    MNNSpacemitIme2Worker2DEnabled() && MNNSpacemitIme2ExecutorPackAEnabled();
        const bool splitMWorkers = !workerPost && realCount >= 4 && MNNSpacemitIme2WorkerSplitMEnabled();
        size_t decodeTcmPairCount = 0;
        size_t decodeTcmChunkCount = groupCount & ~static_cast<size_t>(1);
        size_t decodeTcmGroupCount = groupCount & ~static_cast<size_t>(1);
        if (decodeTcmPipelineRequested && workerCount >= 2 && groupCount >= 2) {
            auto& tcm = MNNSpacemitIme2Tcm();
            MNNSpacemitIme2Job layoutJob;
            layoutJob.blkLen = callKernelBlkLen;
            layoutJob.kBlks = kernelKBlocks;
            const size_t aBytes = MNNSpacemitIme2ARowStrideForJob(layoutJob);
            const size_t alignedABytes = (aBytes + 63) & ~static_cast<size_t>(63);
            if (tcm.available && alignedABytes <= tcm.info.blkSize && bStride <= tcm.info.blkSize - alignedABytes) {
                const size_t tcmWorkers = std::min(workerCount, tcm.info.blkNum) & ~static_cast<size_t>(1);
                decodeTcmPairCount = std::min(tcmWorkers / 2, decodeTcmChunkCount / 2);
                decodeTcmPairCount = std::min<size_t>(decodeTcmPairCount, 4);
            }
        }
        if (decodeTcmPairCount > 0) {
            const size_t pipelineWorkers = decodeTcmPairCount * 2;
            MNNSpacemitIme2PairBarrier pairBarriers[4];
            MNNSpacemitIme2Job jobs[8];
            expected = 0;
            for (size_t pair = 0; pair < decodeTcmPairCount; ++pair) {
                pairBarriers[pair].reset();
                const size_t firstChunk = pair * 2;
                const size_t rounds = (decodeTcmChunkCount - firstChunk + pipelineWorkers - 1) / pipelineWorkers;
                for (size_t role = 0; role < 2; ++role) {
                    const size_t worker = pair * 2 + role;
                    const size_t group = firstChunk + role;
                    MNNSpacemitIme2Job& job = jobs[worker];
                    job.gemm = gemm;
                    job.blkLen = callKernelBlkLen;
                    job.a = packedAPtr;
                    job.b = packedB->data() + group * bStride;
                    job.bZp = useZp ? job.b : nullptr;
                    job.c = gemmOutput + group * 32;
                    job.countM = 1;
                    job.countN = 32;
                    job.kBlks = kernelKBlocks;
                    job.ldc = countN;
                    job.doPost = workerPost;
                    job.dst = dst;
                    job.dstStep = dstStep;
                    job.dzStart = group * 8;
                    job.fullCountN = countN;
                    job.residual = residualPtr;
                    job.post = postForIme2;
                    job.biasPtr = biasPtr;
                    job.fp32min = fp32min;
                    job.fp32max = fp32max;
                    job.skipResidual = skipPostResidual;
                    job.decodeTcmPairBarrier = &pairBarriers[pair];
                    job.decodeTcmPairRole = role;
                    job.decodeTcmPairRounds = rounds;
                    job.decodeTcmPairBStride = bStride;
                    job.decodeTcmPairGroupStep = pipelineWorkers;
                    MNNSpacemitIme2SpinWorker::get(worker).startJob(job);
                    expected += rounds;
                }
            }
            for (size_t worker = 0; worker < pipelineWorkers; ++worker) {
                handled += MNNSpacemitIme2SpinWorker::get(worker).wait();
            }
            if (decodeTcmGroupCount < groupCount) {
                const size_t group = decodeTcmGroupCount;
                const size_t tailGroups = groupCount - group;
                handled += MNNSpacemitIme2SpinWorker::get(0).run(
                    gemm, callKernelBlkLen, packedAPtr, packedB->data() + group * bStride,
                    useZp ? packedB->data() + group * bStride : nullptr, gemmOutput + group * 32, 1, tailGroups * 32,
                    kernelKBlocks, countN, workerPost, dst, dstStep, group * 8, countN, residualPtr, postForIme2,
                    biasPtr, fp32min, fp32max, skipPostResidual);
                expected += 1;
            }
        } else if (split2DWorkers && workerCount > 1) {
            struct Task2D {
                size_t rowOffset = 0;
                size_t rows = 0;
                size_t groupOffset = 0;
                size_t groups = 0;
            };
            constexpr size_t rawRowBlock = 16;
            size_t rowBlock = std::max<size_t>(4, rawRowBlock);
            rowBlock = (rowBlock / 4) * 4;
            rowBlock = std::max<size_t>(4, std::min(rowBlock, realCount));
            const size_t groupBlock = std::max<size_t>(1, std::min(size_t(8), groupCount));
            std::vector<Task2D> tasks;
            tasks.reserve(((realCount + rowBlock - 1) / rowBlock) * ((groupCount + groupBlock - 1) / groupBlock));
            for (size_t rowOffset = 0; rowOffset < realCount; rowOffset += rowBlock) {
                const size_t rows = std::min(rowBlock, realCount - rowOffset);
                for (size_t groupOffset = 0; groupOffset < groupCount; groupOffset += groupBlock) {
                    const size_t groups = std::min(groupBlock, groupCount - groupOffset);
                    Task2D task;
                    task.rowOffset = rowOffset;
                    task.rows = rows;
                    task.groupOffset = groupOffset;
                    task.groups = groups;
                    tasks.push_back(task);
                }
            }
            MNNSpacemitIme2Job strideJob;
            strideJob.blkLen = callKernelBlkLen;
            strideJob.kBlks = kernelKBlocks;
            const size_t aRowStride = MNNSpacemitIme2ARowStrideForJob(strideJob);
            expected = 0;
            size_t taskOffset = 0;
            while (taskOffset < tasks.size()) {
                const size_t running = std::min(workerCount, tasks.size() - taskOffset);
                for (size_t i = 0; i < running; ++i) {
                    const Task2D& task = tasks[taskOffset + i];
                    const size_t nOffset = task.groupOffset * 32;
                    const uint8_t* aPtr = packedAPtr + task.rowOffset * aRowStride;
                    const uint8_t* bPtr = packedB->data() + task.groupOffset * bStride;
                    float* cPtr = gemmOutput + task.rowOffset * countN + nOffset;
                    expected += task.rows;
                    if (spinWorker) {
                        MNNSpacemitIme2SpinWorker::get(i).start(gemm, callKernelBlkLen, aPtr, bPtr,
                                                                useZp ? bPtr : nullptr, cPtr, task.rows,
                                                                task.groups * 32, kernelKBlocks, countN);
                    } else {
                        MNNSpacemitIme2Worker::get(i).start(gemm, callKernelBlkLen, aPtr, bPtr, useZp ? bPtr : nullptr,
                                                            cPtr, task.rows, task.groups * 32, kernelKBlocks, countN);
                    }
                }
                for (size_t i = 0; i < running; ++i) {
                    handled +=
                        spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
                }
                taskOffset += running;
            }
        } else if (splitMWorkers && workerCount > 1) {
            const size_t maxWorkersByM = std::max<size_t>(1, realCount / 4);
            workerCount = std::min(workerCount, maxWorkersByM);
            expected = realCount;
            const size_t baseRows = realCount / workerCount;
            const size_t extraRows = realCount % workerCount;
            MNNSpacemitIme2Job strideJob;
            strideJob.blkLen = callKernelBlkLen;
            strideJob.kBlks = kernelKBlocks;
            const size_t aRowStride = MNNSpacemitIme2ARowStrideForJob(strideJob);
            size_t rowOffset = 0;
            for (size_t i = 0; i < workerCount; ++i) {
                const size_t rows = baseRows + (i < extraRows ? 1 : 0);
                const uint8_t* aPtr = packedAPtr + rowOffset * aRowStride;
                float* cPtr = gemmOutput + rowOffset * countN;
                if (spinWorker) {
                    MNNSpacemitIme2SpinWorker::get(i).start(gemm, callKernelBlkLen, aPtr, packedB->data(),
                                                            useZp ? packedB->data() : nullptr, cPtr, rows, countN,
                                                            kernelKBlocks, countN);
                } else {
                    MNNSpacemitIme2Worker::get(i).start(gemm, callKernelBlkLen, aPtr, packedB->data(),
                                                        useZp ? packedB->data() : nullptr, cPtr, rows, countN,
                                                        kernelKBlocks, countN);
                }
                rowOffset += rows;
            }
            for (size_t i = 0; i < workerCount; ++i) {
                handled += spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
            }
        } else if (workerCount <= 1) {
            static std::atomic<size_t> workerIndex(0);
            const size_t index = workerIndex.fetch_add(1, std::memory_order_relaxed);
            if (spinWorker) {
                handled = MNNSpacemitIme2SpinWorker::get(index).run(
                    gemm, callKernelBlkLen, packedAPtr, packedB->data(), useZp ? packedB->data() : nullptr, gemmOutput,
                    realCount, countN, kernelKBlocks, countN, workerPost, dst, dstStep, 0, countN, residualPtr,
                    postForIme2, biasPtr, fp32min, fp32max, skipPostResidual);
            } else {
                handled = MNNSpacemitIme2Worker::get(index).run(
                    gemm, callKernelBlkLen, packedAPtr, packedB->data(), useZp ? packedB->data() : nullptr, gemmOutput,
                    realCount, countN, kernelKBlocks, countN, workerPost, dst, dstStep, 0, countN, residualPtr,
                    postForIme2, biasPtr, fp32min, fp32max, skipPostResidual);
            }
        } else {
            expected = workerCount * realCount;
            const size_t baseGroups = groupCount / workerCount;
            const size_t extraGroups = groupCount % workerCount;
            size_t groupOffset = 0;
            for (size_t i = 0; i < workerCount; ++i) {
                const size_t groups = baseGroups + (i < extraGroups ? 1 : 0);
                const size_t nOffset = groupOffset * 32;
                if (spinWorker) {
                    MNNSpacemitIme2SpinWorker::get(i).start(
                        gemm, callKernelBlkLen, packedAPtr, packedB->data() + groupOffset * bStride,
                        useZp ? packedB->data() + groupOffset * bStride : nullptr, gemmOutput + nOffset, realCount,
                        groups * 32, kernelKBlocks, countN, workerPost, dst, dstStep, groupOffset * 8, countN,
                        residualPtr, postForIme2, biasPtr, fp32min, fp32max, skipPostResidual);
                } else {
                    MNNSpacemitIme2Worker::get(i).start(
                        gemm, callKernelBlkLen, packedAPtr, packedB->data() + groupOffset * bStride,
                        useZp ? packedB->data() + groupOffset * bStride : nullptr, gemmOutput + nOffset, realCount,
                        groups * 32, kernelKBlocks, countN, workerPost, dst, dstStep, groupOffset * 8, countN,
                        residualPtr, postForIme2, biasPtr, fp32min, fp32max, skipPostResidual);
                }
                groupOffset += groups;
            }
            for (size_t i = 0; i < workerCount; ++i) {
                handled += spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
            }
        }
    }
    if (handled != expected) {
        return false;
    }

    if (!workerPost && !directOutput) {
        if (realCount == 1) {
            MNNSpacemitIme2PostChunk(dst, dstStep, cBuffer.data(), countN, 0, countN, residualPtr, postForIme2, biasPtr,
                                     fp32min, fp32max, useHp, skipPostResidual);
        } else {
            MNNSpacemitIme2PostChunkM(dst, dstStep, cBuffer.data(), realCount, countN, 0, countN, residualPtr,
                                      postForIme2, biasPtr, fp32min, fp32max, useHp, skipPostResidual, nullptr, 0, 0,
                                      callKernelBlkLen);
        }
    }
    if (hasTail) {
        const int weightStepY = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
        const int weightStepZ = weightStepY * srcDepthQuad + 4 * 2 * GEMM_INT8_UNIT;
        QuanPostTreatParameters tailPost = *post;
        tailPost.biasFloat = post->biasFloat + ime2DstDepthQuad * GEMM_INT8_UNIT;
        if (post->weightKernelSum != nullptr) {
            tailPost.weightKernelSum = post->weightKernelSum + ime2DstDepthQuad * (post->blockNum * GEMM_INT8_UNIT);
        }
        if (!MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPost_RVV(dst + ime2DstDepthQuad * dstStep, src,
                                                                  weight + ime2DstDepthQuad * blockNum * weightStepZ,
                                                                  srcDepthQuad, dstStep, tailDstDepthQuad, &tailPost)) {
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" void* MNNSpacemitIme2CreateLinearResource() {
    return new MNNIme2LinearResourceHandle(std::make_shared<MNNIme2LinearResource>());
}

extern "C" void MNNSpacemitIme2DestroyLinearResource(void* context) {
    delete static_cast<MNNIme2LinearResourceHandle*>(context);
}

extern "C" int MNNSpacemitIme2BindLinearWeight(void* context, const int8_t* weight) {
    if (context == nullptr || weight == nullptr) {
        return 0;
    }
    auto handle = static_cast<MNNIme2LinearResourceHandle*>(context);
    if (*handle == nullptr) {
        return 0;
    }
    auto resource = *handle;
    const uintptr_t address = reinterpret_cast<uintptr_t>(weight);
    uintptr_t expected = 0;
    if (resource->weightAddress.compare_exchange_strong(expected, address, std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
        return 1;
    }
    return expected == address ? 1 : 0;
}

extern "C" void MNNSpacemitIme2ClearPackedACache() {
    MNNIme2ClearPackedACache();
}

extern "C" void MNNSpacemitIme2SetBlockInputScale(int enabled) {
    gMNNSpacemitIme2BlockInputScale = enabled != 0;
}

extern "C" void MNNSpacemitIme2PrepackPackedA(const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                              size_t realCount) {
    MNNIme2PrepackPackedA(src, srcDepthQuad, blockNum, realCount);
}

extern "C" size_t MNNSpacemitIme2PackedAHpBytes(size_t srcDepthQuad, size_t blockNum, size_t realCount) {
    if (srcDepthQuad % 2 != 0 || blockNum == 0 || realCount == 0) {
        return 0;
    }
    const size_t kBlocks = (srcDepthQuad / 2) * blockNum;
    if (kBlocks == 0 || kBlocks % 8 != 0) {
        return 0;
    }
    return MNNIme2Q8HpBlockSize() * (kBlocks / 8) * realCount;
}

extern "C" int MNNSpacemitIme2PackAHpTiled(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                           size_t realCount, size_t tileRows) {
    if (dst == nullptr || src == nullptr || tileRows == 0 ||
        MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    MNNPackIme2AMAllBlocksHpTiled(dst, src, srcDepthQuad, blockNum, realCount, tileRows);
    return 1;
}

extern "C" int MNNSpacemitIme2PackAHpTiledRange(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                                size_t realCount, size_t tileRows, size_t rowBegin, size_t rowEnd) {
    if (dst == nullptr || src == nullptr || tileRows == 0 || rowBegin > rowEnd || rowBegin % 4 != 0 ||
        MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    MNNPackIme2AMAllBlocksHpTiledRange(dst, src, srcDepthQuad, blockNum, realCount, tileRows, rowBegin, rowEnd);
    return 1;
}

extern "C" int MNNSpacemitIme2PackAHpTiledRangeWithSum(uint8_t* dst, float* srcKernelSum, const float* inputScale,
                                                       const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                                       size_t realCount, size_t tileRows, size_t rowBegin,
                                                       size_t rowEnd) {
    if (dst == nullptr || srcKernelSum == nullptr || src == nullptr || tileRows == 0 || rowBegin > rowEnd ||
        rowBegin % 4 != 0 || MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    MNNPackIme2AMAllBlocksHpTiledRange(dst, src, srcDepthQuad, blockNum, realCount, tileRows, rowBegin, rowEnd,
                                       srcKernelSum, inputScale);
    return 1;
}

extern "C" int MNNSpacemitIme2PackAHpContiguous(uint8_t* dst, const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                                size_t realCount) {
    if (dst == nullptr || src == nullptr || MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    MNNPackIme2AMAllBlocksHp(dst, src, srcDepthQuad, blockNum, realCount);
    return 1;
}

extern "C" int MNNSpacemitIme2PackAHpStridedRowsWithSum(uint8_t* dst, float* srcKernelSum, const float* inputScale,
                                                        const int8_t* src, size_t srcDepthQuad, size_t blockNum,
                                                        size_t srcRows, size_t rowBegin, size_t realCount) {
    if (dst == nullptr || src == nullptr || srcRows == 0 || realCount == 0 || rowBegin + realCount > srcRows ||
        MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (kBlocks % 8 != 0) {
        return 0;
    }
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t srcDepthStride = GEMM_INT8_SRC_UNIT * srcRows;
    for (size_t rowBase = 0; rowBase < realCount;) {
        const size_t rows = std::min<size_t>(4, realCount - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
                std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t localRow = rowBase + r;
                        const size_t srcRow = rowBegin + localRow;
                        const int8_t* srcBk = src + bk * srcDepthQuad * srcDepthStride + srcRow * GEMM_INT8_SRC_UNIT;
                        const int8_t* src0 = srcBk + (kb * 2) * srcDepthStride;
                        const int8_t* src1 = src0 + srcDepthStride;
                        const int32_t sum = MNNIme2Copy32AndSum(q + r * 32, src0, src1);
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-sum * 8));
                        if (srcKernelSum != nullptr) {
                            float* dstSum = srcKernelSum + bk * realCount + localRow;
                            if (kb == 0) {
                                *dstSum = 0.0f;
                            }
                            const float scale = inputScale == nullptr ? 1.0f : inputScale[srcRow];
                            *dstSum += static_cast<float>(sum) * scale;
                        }
                    }
                }
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t localRow = rowBase + r;
            const size_t srcRow = rowBegin + localRow;
            uint8_t* rowDst = dst + localRow * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = rowDst + super * MNNIme2Q8HpBlockSize();
                std::memset(block, 0, MNNIme2Q8HpBlockSize());
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    const int8_t* srcBk = src + bk * srcDepthQuad * srcDepthStride + srcRow * GEMM_INT8_SRC_UNIT;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
                    const int8_t* src0 = srcBk + (kb * 2) * srcDepthStride;
                    const int8_t* src1 = src0 + srcDepthStride;
                    const int32_t sum = MNNIme2Copy32AndSum(q, src0, src1);
                    MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(-sum * 8));
                    if (srcKernelSum != nullptr) {
                        float* dstSum = srcKernelSum + bk * realCount + localRow;
                        if (kb == 0) {
                            *dstSum = 0.0f;
                        }
                        const float scale = inputScale == nullptr ? 1.0f : inputScale[srcRow];
                        *dstSum += static_cast<float>(sum) * scale;
                    }
                }
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() - sizeof(uint16_t), 1.0f);
            }
        }
        rowBase += rows;
    }
    return 1;
}

extern "C" int MNNSpacemitIme2LinearPackedAGemm(int8_t* dst, const uint8_t* packedA, const int8_t* weight,
                                                size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                const QuanPostTreatParameters* post, size_t realCount, int threadCount,
                                                void* context) {
    if (context == nullptr || !MNNSpacemitIme2Enabled() || dst == nullptr || packedA == nullptr || weight == nullptr ||
        post == nullptr || post->useInt8 != 0 || post->inputScale == nullptr || post->inputBias != nullptr ||
        srcDepthQuad % 2 != 0 || dstDepthQuad < 8 || dstDepthQuad % 8 != 0 || realCount < 4) {
        return 0;
    }
    auto gemm = MNNSpacemitIme2Gemm();
    if (gemm == nullptr) {
        return 0;
    }

    const size_t blockNum = post->blockNum;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (blockNum == 0 || kBlocks == 0 || kBlocks % 8 != 0) {
        return 0;
    }
    const bool useZp = MNNSpacemitIme2ZpEnabled();
    const bool useFusedResidual = MNNSpacemitIme2FuseResidualEnabled() && !useZp;
    const size_t kernelKBlocks = kBlocks / 8;
    const size_t callKernelBlkLen =
        useFusedResidual ? 258 : ((!useZp && MNNSpacemitIme2FixedAScaleEnabled()) ? 260 : 256);
    const size_t groupCount = dstDepthQuad / 8;
    const size_t countN = dstDepthQuad * GEMM_INT8_UNIT;
    const size_t aRowStride = MNNIme2Q8HpBlockSize() * kernelKBlocks;
    const size_t bStride =
        (useFusedResidual ? MNNIme2Q4HpResidualSuperBlockSize() : MNNIme2Q4HpSuperBlockSize(useZp)) * kernelKBlocks;
    constexpr size_t minCountN = 128;
    constexpr size_t minKBlocks = 2;
    if (countN < minCountN || kernelKBlocks < minKBlocks) {
        return 0;
    }

    auto packedB = useFusedResidual
                       ? MNNGetIme2PackedBHpResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, context)
                       : MNNGetIme2PackedBHpAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, useZp, context);
    if (packedB == nullptr || packedB->empty()) {
        return 0;
    }
    std::shared_ptr<MNNIme2Residual> residual;
    if (!useFusedResidual) {
        residual = MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, useZp, context);
    }
    const bool skipPostResidual = useFusedResidual || (!useZp && MNNIme2ResidualAllZeroCached(residual));
    const float* residualPtr = residual == nullptr ? nullptr : residual->values.data();
    if (!useFusedResidual && (residual == nullptr || residual->values.empty())) {
        return 0;
    }

    float fp32min = 0.0f;
    float fp32max = 0.0f;
    if (post->fp32minmax != nullptr) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }
    const float* biasPtr = post->biasFloat;

    const bool spinWorker = useFusedResidual ? MNNSpacemitIme2LinearSpinEnabled() : MNNSpacemitIme2SpinEnabled();
    constexpr bool linearWorkerPost = true;
    size_t workerCount = spinWorker ? MNNSpacemitIme2SpinWorker::count() : MNNSpacemitIme2Worker::count();
    workerCount = std::min(workerCount, static_cast<size_t>(std::max(1, threadCount)));
    workerCount = std::max<size_t>(1, workerCount);

    size_t mStride = useFusedResidual ? 4 : (countN / realCount > 64 ? realCount : (realCount <= 128 ? 4 : 16));
    mStride = std::max<size_t>(4, std::min(mStride, realCount));
    if (mStride < realCount) {
        mStride = std::max<size_t>(4, (mStride / 4) * 4);
    }
    const size_t taskCountMForN = (realCount + mStride - 1) / mStride;
    const size_t maxNStride = (countN * taskCountMForN + workerCount - 1) / workerCount;
    size_t nStride = countN;
    if (maxNStride < countN) {
        nStride = std::min(countN, ((maxNStride + 31) / 32) * 32);
    }
    size_t nGroupStride = std::max<size_t>(1, nStride / 32);
    nGroupStride = std::max<size_t>(1, std::min(nGroupStride, groupCount));

    struct LinearTask {
        size_t rowOffset = 0;
        size_t rows = 0;
        size_t groupOffset = 0;
        size_t groups = 0;
    };
    std::vector<LinearTask> tasks;
    tasks.reserve(((realCount + mStride - 1) / mStride) * ((groupCount + nGroupStride - 1) / nGroupStride));
    for (size_t rowOffset = 0; rowOffset < realCount; rowOffset += mStride) {
        const size_t rows = std::min(mStride, realCount - rowOffset);
        for (size_t groupOffset = 0; groupOffset < groupCount; groupOffset += nGroupStride) {
            LinearTask task;
            task.rowOffset = rowOffset;
            task.rows = rows;
            task.groupOffset = groupOffset;
            task.groups = std::min(nGroupStride, groupCount - groupOffset);
            tasks.push_back(task);
        }
    }
    if (tasks.empty()) {
        return 0;
    }
    workerCount = std::min(workerCount, tasks.size());

    const bool directLinearSrcSum = linearWorkerPost && post->srcKernelSum != nullptr &&
                                    !MNNSpacemitIme2PostMRowNEnabled() && !MNNSpacemitIme2FusedPostMEnabled();
    const bool linearStridedWorkers =
        linearWorkerPost && directLinearSrcSum && nGroupStride == groupCount && mStride == 4;
    if (linearStridedWorkers) {
        const bool directC4Epilogue = MNNSpacemitIme2DirectC4EpilogueEnabled() && callKernelBlkLen == 258 &&
                                      useFusedResidual && skipPostResidual && post->useInt8 == 0 &&
                                      post->inputScale != nullptr && post->inputBias == nullptr && countN % 32 == 0;
        std::vector<std::vector<float>> cBuffers(workerCount);
        std::vector<size_t> activeWorkers;
        activeWorkers.reserve(workerCount);
        size_t expected = 0;
        for (size_t i = 0; i < workerCount; ++i) {
            const size_t rowStart = i * mStride;
            if (rowStart >= realCount) {
                continue;
            }
            for (size_t rowOffset = rowStart; rowOffset < realCount; rowOffset += workerCount * mStride) {
                expected += std::min(mStride, realCount - rowOffset);
            }
            cBuffers[i].resize(mStride * countN);
            MNNSpacemitIme2Job job;
            job.gemm = gemm;
            job.blkLen = callKernelBlkLen;
            job.a = packedA;
            job.b = packedB->data();
            job.bZp = useZp ? packedB->data() : nullptr;
            job.c = cBuffers[i].data();
            job.countM = mStride;
            job.countN = countN;
            job.kBlks = kernelKBlocks;
            job.ldc = countN;
            job.doPost = true;
            job.dst = dst;
            job.dstStep = dstStep;
            job.dzStart = 0;
            job.fullCountN = countN;
            job.residual = residualPtr;
            job.post = post;
            job.biasPtr = biasPtr;
            job.fp32min = fp32min;
            job.fp32max = fp32max;
            job.skipResidual = skipPostResidual;
            job.directSrcKernelSum = post->srcKernelSum;
            job.directSrcKernelSumStride = realCount;
            job.directC4Epilogue = directC4Epilogue;
            job.linearStride = true;
            job.linearRowStart = rowStart;
            job.linearRowEnd = realCount;
            job.linearRowStep = workerCount * mStride;
            job.linearRowsPerBlock = mStride;
            job.linearTotalRows = realCount;
            if (spinWorker) {
                MNNSpacemitIme2SpinWorker::get(i).startJob(job);
            } else {
                MNNSpacemitIme2Worker::get(i).startJob(job);
            }
            activeWorkers.push_back(i);
        }
        size_t handled = 0;
        for (size_t index : activeWorkers) {
            handled +=
                spinWorker ? MNNSpacemitIme2SpinWorker::get(index).wait() : MNNSpacemitIme2Worker::get(index).wait();
        }
        if (handled != expected || handled != realCount) {
            return 0;
        }
        return 1;
    }

    std::vector<std::vector<float>> cBuffers(workerCount);
    std::vector<std::vector<float>> srcSums(directLinearSrcSum ? 0 : workerCount);
    std::vector<QuanPostTreatParameters> taskPosts(workerCount);
    size_t handled = 0;
    size_t expected = 0;
    for (size_t taskOffset = 0; taskOffset < tasks.size();) {
        const size_t running = std::min(workerCount, tasks.size() - taskOffset);
        for (size_t i = 0; i < running; ++i) {
            const LinearTask& task = tasks[taskOffset + i];
            expected += task.rows;
            const size_t nOffset = task.groupOffset * 32;
            const size_t localCountN = task.groups * 32;
            cBuffers[i].resize(task.rows * countN);
            if (directLinearSrcSum) {
                taskPosts[i] = *post;
                taskPosts[i].inputScale = post->inputScale + task.rowOffset;
                taskPosts[i].srcKernelSum = post->srcKernelSum;
            } else {
                srcSums[i].resize(blockNum * task.rows);
                if (post->srcKernelSum != nullptr) {
                    for (size_t bk = 0; bk < blockNum; ++bk) {
                        std::memcpy(srcSums[i].data() + bk * task.rows,
                                    post->srcKernelSum + bk * realCount + task.rowOffset, task.rows * sizeof(float));
                    }
                } else {
                    std::fill(srcSums[i].begin(), srcSums[i].end(), 0.0f);
                }
                taskPosts[i] = *post;
                taskPosts[i].inputScale = post->inputScale + task.rowOffset;
                taskPosts[i].srcKernelSum = srcSums[i].data();
            }

            const uint8_t* aPtr = packedA + task.rowOffset * aRowStride;
            const uint8_t* bPtr = packedB->data() + task.groupOffset * bStride;
            float* cPtr = cBuffers[i].data() + nOffset;
            int8_t* dstPtr = dst + task.rowOffset * GEMM_INT8_UNIT * sizeof(float);
            const float* directSrcKernelSum = directLinearSrcSum ? post->srcKernelSum : nullptr;
            const size_t directSrcKernelSumStride = directLinearSrcSum ? realCount : 0;
            const size_t directSrcRowOffset = directLinearSrcSum ? task.rowOffset : 0;
            if (spinWorker) {
                MNNSpacemitIme2SpinWorker::get(i).start(
                    gemm, callKernelBlkLen, aPtr, bPtr, useZp ? bPtr : nullptr, cPtr, task.rows, localCountN,
                    kernelKBlocks, countN, linearWorkerPost, dstPtr, dstStep, task.groupOffset * 8, countN, residualPtr,
                    &taskPosts[i], biasPtr, fp32min, fp32max, skipPostResidual, directSrcKernelSum,
                    directSrcKernelSumStride, directSrcRowOffset);
            } else {
                MNNSpacemitIme2Worker::get(i).start(gemm, callKernelBlkLen, aPtr, bPtr, useZp ? bPtr : nullptr, cPtr,
                                                    task.rows, localCountN, kernelKBlocks, countN, linearWorkerPost,
                                                    dstPtr, dstStep, task.groupOffset * 8, countN, residualPtr,
                                                    &taskPosts[i], biasPtr, fp32min, fp32max, skipPostResidual,
                                                    directSrcKernelSum, directSrcKernelSumStride, directSrcRowOffset);
            }
        }
        for (size_t i = 0; i < running; ++i) {
            handled += spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
        }
        if (!linearWorkerPost) {
            for (size_t i = 0; i < running; ++i) {
                const LinearTask& task = tasks[taskOffset + i];
                const size_t nOffset = task.groupOffset * 32;
                const size_t localCountN = task.groups * 32;
                float* cPtr = cBuffers[i].data() + nOffset;
                int8_t* dstPtr = dst + task.rowOffset * GEMM_INT8_UNIT * sizeof(float);
                MNNSpacemitIme2PostChunkM(dstPtr, dstStep, cPtr, task.rows, localCountN, task.groupOffset * 8, countN,
                                          residualPtr, &taskPosts[i], biasPtr, fp32min, fp32max, true, skipPostResidual,
                                          nullptr, 0, 0, callKernelBlkLen);
            }
        }
        taskOffset += running;
    }
    if (handled != expected) {
        return 0;
    }
    return 1;
}

static inline int8_t MNNIme2QuantFloatToInt8(float value, float scale) {
    int v = static_cast<int>(std::nearbyint(value * scale));
    v = std::max(-128, std::min(127, v));
    return static_cast<int8_t>(v);
}

static inline void MNNIme2StoreNativeHalfRne(uint8_t* dst, float value) {
    // K3's scalar float-to-half conversion uses the default FRM (RNE). Keep this separate from
    // MNNIme2StoreHalf, whose software conversion rounds halfway cases away from zero.
    const _Float16 half = static_cast<_Float16>(value);
    std::memcpy(dst, &half, sizeof(half));
}

static inline bool MNNIme2QuantFloat32K32RneSingleLoad(int8_t* dst, const float* src, float* scale, int32_t* sum) {
    const size_t vl = __riscv_vsetvl_e32m4(32);
    if (vl != 32) {
        return false;
    }

    // The K3 caller runs on a VLENB=32 application core. e32m4 therefore holds the complete K32 block.
    // The packed row is consumed later by an IME2 worker after that worker is moved to a VLENB=128 AI core.
    // Keep the original values live across the reduction and quantize that same vector after the scalar scale is
    // known, avoiding the second source load.
    vfloat32m4_t value = __riscv_vle32_v_f32m4(src, vl);
    const vfloat32m4_t absolute = __riscv_vfabs_v_f32m4(value, vl);
    const vfloat32m1_t maximum = __riscv_vfredmax_vs_f32m4_f32m1(absolute, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
    const float absMax = __riscv_vfmv_f_s_f32m1_f32(maximum);
    if (!std::isfinite(absMax)) {
        return false;
    }
    const float localScale = absMax / 127.0f;
    const float reciprocalScale = localScale > 0.0f ? 1.0f / localScale : 0.0f;
    if (!std::isfinite(reciprocalScale)) {
        return false;
    }

    value = __riscv_vfmul_vf_f32m4(value, reciprocalScale, vl);
    value = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(value, -127.0f, vl), 127.0f, vl);
    // No explicit rounding-mode immediate: vfncvt follows the default FRM/RNE, matching llama.cpp.
    const vint16m2_t quant16 = __riscv_vfncvt_x_f_w_i16m2(value, vl);
    const vint8m1_t quant8 = __riscv_vncvt_x_x_w_i8m1(quant16, vl);
    __riscv_vse8_v_i8m1(dst, quant8, vl);
    const vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
    const vint16m1_t reduced = __riscv_vwredsum_vs_i8m1_i16m1(quant8, zero, vl);
    *scale = localScale;
    *sum = __riscv_vmv_x_s_i16m1_i16(reduced);
    return true;
}

static bool MNNIme2PackFloatA1HierarchicalHp(uint8_t* dst, const float* src, size_t superBlocks) {
    constexpr size_t kSubBlockSize = sizeof(uint16_t) + 32;
    constexpr size_t kSumOffset = 8 * kSubBlockSize;
    const size_t hpBlockSize = MNNIme2Q8HpBlockSize();
    for (size_t super = 0; super < superBlocks; ++super) {
        uint8_t* block = dst + super * hpBlockSize;
        const float* srcSuper = src + super * 256;
        float subScales[8];
        for (size_t sub = 0; sub < 8; ++sub) {
            const float* srcSub = srcSuper + sub * 32;
            uint8_t* subBlock = block + sub * kSubBlockSize;
            int32_t sum = 0;
            if (!MNNIme2QuantFloat32K32RneSingleLoad(reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t)), srcSub,
                                                     subScales + sub, &sum)) {
                return false;
            }
            MNNIme2StoreNativeHalfRne(block + kSumOffset + sub * sizeof(uint16_t), -8.0f * static_cast<float>(sum));
        }
        float scaleAverage = 0.0f;
        for (float scale : subScales) {
            scaleAverage += scale;
        }
        scaleAverage *= 0.125f;
        if (!std::isfinite(scaleAverage)) {
            return false;
        }
        for (size_t sub = 0; sub < 8; ++sub) {
            const float relativeScale = scaleAverage > 0.0f ? subScales[sub] / scaleAverage : 0.0f;
            MNNIme2StoreNativeHalfRne(block + sub * kSubBlockSize, relativeScale);
        }
        MNNIme2StoreNativeHalfRne(block + hpBlockSize - sizeof(uint16_t), scaleAverage);
    }
    return true;
}

static inline int32_t MNNIme2QuantFloatToInt8Rvv(int8_t* dst, const float* src, float scale, size_t count) {
    const size_t RV_RNU = 0x4;
    size_t offset = 0;
    int32_t sum = 0;
    while (offset < count) {
        const size_t vl = __riscv_vsetvl_e32m4(count - offset);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(src + offset, vl);
        v = __riscv_vfmul_vf_f32m4(v, scale, vl);
        vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4_rm(v, RV_RNU, vl);
        vi = __riscv_vmax_vx_i32m4(vi, -128, vl);
        vi = __riscv_vmin_vx_i32m4(vi, 127, vl);
        vint32m1_t vzero = __riscv_vmv_s_x_i32m1(0, vl);
        vint32m1_t vred = __riscv_vredsum_vs_i32m4_i32m1(vi, vzero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(vred);
        vint16m2_t v16 = __riscv_vncvt_x_x_w_i16m2(vi, vl);
        vint8m1_t v8 = __riscv_vncvt_x_x_w_i8m1(v16, vl);
        __riscv_vse8_v_i8m1(dst + offset, v8, vl);
        offset += vl;
    }
    return sum;
}

static inline int32_t MNNIme2Quant16FloatToInt8Rvv(int8_t* dst, const float* src, float scale) {
    return MNNIme2QuantFloatToInt8Rvv(dst, src, scale, GEMM_INT8_SRC_UNIT);
}

static inline int32_t MNNIme2QuantFloatC4x4ToInt8(int8_t* dst, const float* srcC4Base, size_t c4Stride, float scale) {
    int32_t sum = 0;
    for (size_t c4 = 0; c4 < 4; ++c4) {
        const float* srcC4 = srcC4Base + c4 * c4Stride;
        int8_t* dstC4 = dst + c4 * 4;
        for (size_t i = 0; i < 4; ++i) {
            const int8_t q = MNNIme2QuantFloatToInt8(srcC4[i], scale);
            dstC4[i] = q;
            sum += q;
        }
    }
    return sum;
}

static inline int32_t MNNIme2QuantFloatC4x8ToInt8(int8_t* dst, const float* srcC4Base, size_t c4Stride, float scale) {
    const size_t vl = __riscv_vsetvl_e32m1(8);
    if (vl != 8) {
        return MNNIme2QuantFloatC4x4ToInt8(dst, srcC4Base, c4Stride, scale) +
               MNNIme2QuantFloatC4x4ToInt8(dst + 16, srcC4Base + 4 * c4Stride, c4Stride, scale);
    }
    const vfloat32m1x4_t values = __riscv_vlsseg4e32_v_f32m1x4(srcC4Base, c4Stride * sizeof(float), vl);
    vfloat32m1_t value0 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 0), scale, vl);
    vfloat32m1_t value1 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 1), scale, vl);
    vfloat32m1_t value2 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 2), scale, vl);
    vfloat32m1_t value3 = __riscv_vfmul_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 3), scale, vl);
    value0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value0, -128.0f, vl), 127.0f, vl);
    value1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value1, -128.0f, vl), 127.0f, vl);
    value2 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value2, -128.0f, vl), 127.0f, vl);
    value3 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value3, -128.0f, vl), 127.0f, vl);
    const vint8mf4_t quant0 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(value0, vl), vl);
    const vint8mf4_t quant1 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(value1, vl), vl);
    const vint8mf4_t quant2 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(value2, vl), vl);
    const vint8mf4_t quant3 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(value3, vl), vl);
    const vint16mf2_t sum01 = __riscv_vwadd_vv_i16mf2(quant0, quant1, vl);
    const vint16mf2_t sum23 = __riscv_vwadd_vv_i16mf2(quant2, quant3, vl);
    const vint16mf2_t laneSums = __riscv_vadd_vv_i16mf2(sum01, sum23, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    const int32_t sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16mf2_i32m1(laneSums, zero, vl));
    const vint8mf4x4_t quant = __riscv_vcreate_v_i8mf4x4(quant0, quant1, quant2, quant3);
    __riscv_vsseg4e8_v_i8mf4x4(dst, quant, vl);
    return sum;
}

extern "C" int MNNSpacemitIme2PackFloatAHpStridedRowsWithSum(uint8_t* dst, float* srcKernelSum, const float* inputScale,
                                                             const float* quantScale, const float* src,
                                                             size_t srcDepthQuad, size_t blockNum, size_t srcRows,
                                                             size_t rowBegin, size_t realCount) {
    if (dst == nullptr || src == nullptr || inputScale == nullptr || quantScale == nullptr || srcRows == 0 ||
        realCount == 0 || rowBegin + realCount > srcRows ||
        MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount) == 0) {
        return 0;
    }
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (kBlocks % 8 != 0) {
        return 0;
    }
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t c4Stride = 4 * srcRows;
    for (size_t rowBase = 0; rowBase < realCount;) {
        const size_t rows = std::min<size_t>(4, realCount - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
                std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t localRow = rowBase + r;
                        const size_t srcRow = rowBegin + localRow;
                        const float qScale = quantScale[srcRow];
                        const size_t c4Base = (bk * srcDepthQuad + kb * 2) * 4;
                        const int32_t sum = MNNIme2QuantFloatC4x8ToInt8(
                            q + r * 32, src + c4Base * c4Stride + srcRow * 4, c4Stride, qScale);
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-sum * 8));
                        if (srcKernelSum != nullptr) {
                            float* dstSum = srcKernelSum + bk * realCount + localRow;
                            if (kb == 0) {
                                *dstSum = 0.0f;
                            }
                            *dstSum += static_cast<float>(sum) * inputScale[srcRow];
                        }
                    }
                }
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t localRow = rowBase + r;
            const size_t srcRow = rowBegin + localRow;
            uint8_t* rowDst = dst + localRow * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = rowDst + super * MNNIme2Q8HpBlockSize();
                std::memset(block, 0, MNNIme2Q8HpBlockSize());
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
                    const float qScale = quantScale[srcRow];
                    const size_t c4Base = (bk * srcDepthQuad + kb * 2) * 4;
                    const int32_t sum =
                        MNNIme2QuantFloatC4x8ToInt8(q, src + c4Base * c4Stride + srcRow * 4, c4Stride, qScale);
                    MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(-sum * 8));
                    if (srcKernelSum != nullptr) {
                        float* dstSum = srcKernelSum + bk * realCount + localRow;
                        if (kb == 0) {
                            *dstSum = 0.0f;
                        }
                        *dstSum += static_cast<float>(sum) * inputScale[srcRow];
                    }
                }
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() - sizeof(uint16_t), 1.0f);
            }
        }
        rowBase += rows;
    }
    return 1;
}

extern "C" int MNNSpacemitIme2PackFloatAHpStridedRowsRangeWithSum(uint8_t* dst, float* srcKernelSum,
                                                                  const float* inputScale, const float* quantScale,
                                                                  const float* src, size_t srcDepthQuad,
                                                                  size_t blockNum, size_t srcRows, size_t rowBegin,
                                                                  size_t rowEnd) {
    if (dst == nullptr || src == nullptr || inputScale == nullptr || quantScale == nullptr || srcRows == 0 ||
        rowBegin > rowEnd || rowEnd > srcRows || rowBegin % 4 != 0 ||
        MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, srcRows) == 0) {
        return 0;
    }
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (kBlocks % 8 != 0) {
        return 0;
    }
    const size_t superBlocks = kBlocks / 8;
    const size_t hpRowStride = MNNIme2Q8HpBlockSize() * superBlocks;
    const size_t c4Stride = 4 * srcRows;
    for (size_t rowBase = rowBegin; rowBase < rowEnd;) {
        const size_t rows = std::min<size_t>(4, rowEnd - rowBase);
        if (rows == 4) {
            uint8_t* tileDst = dst + rowBase * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = tileDst + super * MNNIme2Q8HpBlockSize() * 4;
                std::memset(block, 0, MNNIme2Q8HpBlockSize() * 4);
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) * 4 + 32 * 4);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) * 4 + 32 * 4);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t) * 4);
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t srcRow = rowBase + r;
                        const float qScale = quantScale[srcRow];
                        const size_t c4Base = (bk * srcDepthQuad + kb * 2) * 4;
                        const int32_t sum = MNNIme2QuantFloatC4x8ToInt8(
                            q + r * 32, src + c4Base * c4Stride + srcRow * 4, c4Stride, qScale);
                        MNNIme2StoreHalf(sumBase + (r * 8 + sub) * sizeof(uint16_t), static_cast<float>(-sum * 8));
                        if (srcKernelSum != nullptr) {
                            float* dstSum = srcKernelSum + bk * srcRows + srcRow;
                            if (kb == 0) {
                                *dstSum = 0.0f;
                            }
                            *dstSum += static_cast<float>(sum) * inputScale[srcRow];
                        }
                    }
                }
                for (size_t r = 0; r < 4; ++r) {
                    MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() * 4 - sizeof(uint16_t) * (4 - r), 1.0f);
                }
            }
            rowBase += 4;
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            const size_t srcRow = rowBase + r;
            uint8_t* rowDst = dst + srcRow * hpRowStride;
            for (size_t super = 0; super < superBlocks; ++super) {
                uint8_t* block = rowDst + super * MNNIme2Q8HpBlockSize();
                std::memset(block, 0, MNNIme2Q8HpBlockSize());
                uint8_t* sumBase = block + 8 * (sizeof(uint16_t) + 32);
                for (size_t sub = 0; sub < 8; ++sub) {
                    const size_t linearKb = super * 8 + sub;
                    const size_t bk = linearKb / kBlocksPerBlock;
                    const size_t kb = linearKb - bk * kBlocksPerBlock;
                    uint8_t* subBlock = block + sub * (sizeof(uint16_t) + 32);
                    MNNIme2StoreHalf(subBlock, 1.0f);
                    int8_t* q = reinterpret_cast<int8_t*>(subBlock + sizeof(uint16_t));
                    const float qScale = quantScale[srcRow];
                    const size_t c4Base = (bk * srcDepthQuad + kb * 2) * 4;
                    const int32_t sum =
                        MNNIme2QuantFloatC4x8ToInt8(q, src + c4Base * c4Stride + srcRow * 4, c4Stride, qScale);
                    MNNIme2StoreHalf(sumBase + sub * sizeof(uint16_t), static_cast<float>(-sum * 8));
                    if (srcKernelSum != nullptr) {
                        float* dstSum = srcKernelSum + bk * srcRows + srcRow;
                        if (kb == 0) {
                            *dstSum = 0.0f;
                        }
                        *dstSum += static_cast<float>(sum) * inputScale[srcRow];
                    }
                }
                MNNIme2StoreHalf(block + MNNIme2Q8HpBlockSize() - sizeof(uint16_t), 1.0f);
            }
        }
        rowBase += rows;
    }
    return 1;
}

static inline float MNNIme2FloatRowAbsMax(const float* src, size_t srcDepthQuad, size_t blockNum, size_t srcRows,
                                          size_t srcRow) {
    const size_t c4Count = srcDepthQuad * blockNum * 4;
    const size_t c4Stride = 4 * srcRows;
    const size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t maxV = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (size_t c4 = 0; c4 < c4Count; ++c4) {
        const float* srcC4 = src + c4 * c4Stride + srcRow * 4;
        const vfloat32m1_t value = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(srcC4, vl), vl);
        maxV = __riscv_vfmax_vv_f32m1(maxV, value, vl);
    }
    const vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m1_f32m1(maxV, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
    return __riscv_vfmv_f_s_f32m1_f32(reduced);
}

static inline vfloat32m1_t MNNIme2FloatC4GroupAbsMax(const float* src, size_t c4StrideBytes, size_t vl) {
    const vfloat32m1x4_t values = __riscv_vlsseg4e32_v_f32m1x4(src, c4StrideBytes, vl);
    vfloat32m1_t maximum = __riscv_vfabs_v_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 0), vl);
    maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 1), vl), vl);
    maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 2), vl), vl);
    return __riscv_vfmax_vv_f32m1(maximum, __riscv_vfabs_v_f32m1(__riscv_vget_v_f32m1x4_f32m1(values, 3), vl), vl);
}

static inline void MNNIme2FloatRows4AbsMax(float* absMax, const float* src, size_t srcDepthQuad, size_t blockNum,
                                           size_t srcRows, size_t srcRow) {
    const size_t c4Count = srcDepthQuad * blockNum * 4;
    const size_t c4Stride = 4 * srcRows;
    const size_t vl = __riscv_vsetvl_e32m1(8);
    vfloat32m1_t max0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t max1 = max0;
    vfloat32m1_t max2 = max0;
    vfloat32m1_t max3 = max0;
    size_t c4 = 0;
    for (; c4 + vl <= c4Count; c4 += vl) {
        const float* srcC4 = src + c4 * c4Stride + srcRow * 4;
        const size_t c4StrideBytes = c4Stride * sizeof(float);
        max0 = __riscv_vfmax_vv_f32m1(max0, MNNIme2FloatC4GroupAbsMax(srcC4, c4StrideBytes, vl), vl);
        max1 = __riscv_vfmax_vv_f32m1(max1, MNNIme2FloatC4GroupAbsMax(srcC4 + 4, c4StrideBytes, vl), vl);
        max2 = __riscv_vfmax_vv_f32m1(max2, MNNIme2FloatC4GroupAbsMax(srcC4 + 8, c4StrideBytes, vl), vl);
        max3 = __riscv_vfmax_vv_f32m1(max3, MNNIme2FloatC4GroupAbsMax(srcC4 + 12, c4StrideBytes, vl), vl);
    }
    const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    absMax[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max0, zero, vl));
    absMax[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max1, zero, vl));
    absMax[2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max2, zero, vl));
    absMax[3] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max3, zero, vl));
    if (c4 < c4Count) {
        const size_t tailVl = __riscv_vsetvl_e32m1(c4Count - c4);
        const float* srcC4 = src + c4 * c4Stride + srcRow * 4;
        const size_t c4StrideBytes = c4Stride * sizeof(float);
        const vfloat32m1_t tailZero = __riscv_vfmv_s_f_f32m1(0.0f, tailVl);
        const vfloat32m1_t tail0 = MNNIme2FloatC4GroupAbsMax(srcC4, c4StrideBytes, tailVl);
        const vfloat32m1_t tail1 = MNNIme2FloatC4GroupAbsMax(srcC4 + 4, c4StrideBytes, tailVl);
        const vfloat32m1_t tail2 = MNNIme2FloatC4GroupAbsMax(srcC4 + 8, c4StrideBytes, tailVl);
        const vfloat32m1_t tail3 = MNNIme2FloatC4GroupAbsMax(srcC4 + 12, c4StrideBytes, tailVl);
        absMax[0] =
            std::max(absMax[0], __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(tail0, tailZero, tailVl)));
        absMax[1] =
            std::max(absMax[1], __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(tail1, tailZero, tailVl)));
        absMax[2] =
            std::max(absMax[2], __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(tail2, tailZero, tailVl)));
        absMax[3] =
            std::max(absMax[3], __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(tail3, tailZero, tailVl)));
    }
}

extern "C" int MNNSpacemitIme2PackFloatAHpStridedRowsRangeDynamicQuant(uint8_t* dst, float* srcKernelSum,
                                                                       float* inputScale, float* quantScale,
                                                                       const float* src, size_t srcDepthQuad,
                                                                       size_t blockNum, size_t srcRows, size_t rowBegin,
                                                                       size_t rowEnd) {
    if (dst == nullptr || src == nullptr || inputScale == nullptr || quantScale == nullptr || srcRows == 0 ||
        rowBegin > rowEnd || rowEnd > srcRows) {
        return 0;
    }
    size_t row = rowBegin;
    for (; row + 4 <= rowEnd; row += 4) {
        float absMax[4];
        MNNIme2FloatRows4AbsMax(absMax, src, srcDepthQuad, blockNum, srcRows, row);
        for (size_t r = 0; r < 4; ++r) {
            if (absMax[r] < 1e-7f) {
                quantScale[row + r] = 1.0f;
                inputScale[row + r] = 1.0f;
            } else {
                quantScale[row + r] = 127.0f / absMax[r];
                inputScale[row + r] = absMax[r] / 127.0f;
            }
        }
    }
    for (; row < rowEnd; ++row) {
        const float absMax = MNNIme2FloatRowAbsMax(src, srcDepthQuad, blockNum, srcRows, row);
        if (absMax < 1e-7f) {
            quantScale[row] = 1.0f;
            inputScale[row] = 1.0f;
        } else {
            quantScale[row] = 127.0f / absMax;
            inputScale[row] = absMax / 127.0f;
        }
    }
    return MNNSpacemitIme2PackFloatAHpStridedRowsRangeWithSum(dst, srcKernelSum, inputScale, quantScale, src,
                                                              srcDepthQuad, blockNum, srcRows, rowBegin, rowEnd);
}

extern "C" int MNNSpacemitIme2LinearFloatHpDecode(int8_t* dst, const float* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, void* context) {
#if defined(MNN_USE_SPACEMIT_IME2)
    if (context == nullptr || !MNNSpacemitIme2Enabled() || dst == nullptr || src == nullptr || weight == nullptr ||
        post == nullptr || post->useInt8 != 0 || post->blockNum <= 0 || srcDepthQuad == 0 || srcDepthQuad % 2 != 0 ||
        dstDepthQuad == 0 || dstDepthQuad % 8 != 0 || dstStep != GEMM_INT8_UNIT * sizeof(float) ||
        post->inputScale != nullptr || post->inputBias != nullptr || __riscv_vlenb() != 32) {
        return 0;
    }
    const size_t blockNum = static_cast<size_t>(post->blockNum);
    const size_t depthQuads = srcDepthQuad * blockNum;
    if (depthQuads % 16 != 0) {
        return 0;
    }
    const size_t superBlocks = depthQuads / 16;
    const bool useHpM1AsymPair = srcDepthQuad == 4 && blockNum % 4 == 0 && MNNSpacemitIme2HpM1AsymPairEnabled();
    if (!useHpM1AsymPair) {
        auto centeredResidual =
            MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, false, context);
        if (!MNNIme2ResidualAllZeroExact(centeredResidual)) {
            return 0;
        }
    }

    thread_local std::vector<uint8_t> packedA;
    const size_t packedABytes = superBlocks * MNNIme2Q8HpBlockSize();
    if (packedA.size() < packedABytes) {
        packedA.resize(packedABytes);
    }
    if (!MNNIme2PackFloatA1HierarchicalHp(packedA.data(), src, superBlocks)) {
        return 0;
    }
    return MNNGemmInt8AddBiasScale_16x4_w4_Unit_IME2(dst, reinterpret_cast<const int8_t*>(packedA.data()), weight,
                                                     srcDepthQuad, dstStep, dstDepthQuad, post, 1, true, context)
               ? 1
               : 0;
#else
    (void)dst;
    (void)src;
    (void)weight;
    (void)srcDepthQuad;
    (void)dstStep;
    (void)dstDepthQuad;
    (void)post;
    (void)context;
    return 0;
#endif
}

extern "C" int MNNSpacemitIme2LinearFloatFusedGemm(int8_t* dst, const float* src, const float* inputScale,
                                                   const float* quantScale, const int8_t* weight, size_t srcDepthQuad,
                                                   size_t dstStep, size_t dstDepthQuad,
                                                   const QuanPostTreatParameters* post, size_t realCount,
                                                   int threadCount) {
    if (!MNNSpacemitIme2Enabled() || dst == nullptr || src == nullptr || inputScale == nullptr ||
        quantScale == nullptr || weight == nullptr || post == nullptr || post->useInt8 != 0 ||
        post->inputBias != nullptr || srcDepthQuad % 2 != 0 || dstDepthQuad < 8 || dstDepthQuad % 8 != 0 ||
        realCount < 4 || post->blockNum == 0) {
        return 0;
    }
    auto gemm = MNNSpacemitIme2Gemm();
    if (gemm == nullptr) {
        return 0;
    }

    const size_t blockNum = post->blockNum;
    const size_t kBlocksPerBlock = srcDepthQuad / 2;
    const size_t kBlocks = kBlocksPerBlock * blockNum;
    if (kBlocks == 0 || kBlocks % 8 != 0) {
        return 0;
    }
    const bool useZp = MNNSpacemitIme2ZpEnabled();
    const size_t kernelKBlocks = kBlocks / 8;
    const size_t callKernelBlkLen = (!useZp && MNNSpacemitIme2FixedAScaleEnabled()) ? 260 : 256;
    const size_t groupCount = dstDepthQuad / 8;
    const size_t countN = dstDepthQuad * GEMM_INT8_UNIT;
    const size_t bStride = MNNIme2Q4HpSuperBlockSize(useZp) * kernelKBlocks;
    constexpr size_t minCountN = 128;
    constexpr size_t minKBlocks = 2;
    if (countN < minCountN || kernelKBlocks < minKBlocks) {
        return 0;
    }

    auto packedB = MNNGetIme2PackedBHpAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, useZp);
    if (packedB == nullptr || packedB->empty()) {
        return 0;
    }
    auto residual = MNNGetIme2ResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, useZp);
    const bool skipPostResidual = !useZp && MNNIme2ResidualAllZeroCached(residual);
    if (residual == nullptr || residual->values.empty()) {
        return 0;
    }

    float fp32min = 0.0f;
    float fp32max = 0.0f;
    if (post->fp32minmax != nullptr) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }
    const float* biasPtr = post->biasFloat;
    const bool spinWorker = MNNSpacemitIme2SpinEnabled();
    size_t workerCount = spinWorker ? MNNSpacemitIme2SpinWorker::count() : MNNSpacemitIme2Worker::count();
    workerCount = std::min(workerCount, static_cast<size_t>(std::max(1, threadCount)));
    workerCount = std::max<size_t>(1, workerCount);

    size_t mStride = countN / realCount > 64 ? realCount : (realCount <= 128 ? 4 : 16);
    mStride = std::max<size_t>(4, std::min(mStride, realCount));
    if (mStride < realCount) {
        mStride = std::max<size_t>(4, (mStride / 4) * 4);
    }
    const size_t taskCountMForN = (realCount + mStride - 1) / mStride;
    const size_t maxNStride = (countN * taskCountMForN + workerCount - 1) / workerCount;
    size_t nStride = countN;
    if (maxNStride < countN) {
        nStride = std::min(countN, ((maxNStride + 31) / 32) * 32);
    }
    size_t nGroupStride = std::max<size_t>(1, nStride / 32);
    nGroupStride = std::max<size_t>(1, std::min(nGroupStride, groupCount));

    struct LinearTask {
        size_t rowOffset = 0;
        size_t rows = 0;
        size_t groupOffset = 0;
        size_t groups = 0;
    };
    std::vector<LinearTask> tasks;
    tasks.reserve(((realCount + mStride - 1) / mStride) * ((groupCount + nGroupStride - 1) / nGroupStride));
    for (size_t rowOffset = 0; rowOffset < realCount; rowOffset += mStride) {
        const size_t rows = std::min(mStride, realCount - rowOffset);
        for (size_t groupOffset = 0; groupOffset < groupCount; groupOffset += nGroupStride) {
            LinearTask task;
            task.rowOffset = rowOffset;
            task.rows = rows;
            task.groupOffset = groupOffset;
            task.groups = std::min(nGroupStride, groupCount - groupOffset);
            tasks.push_back(task);
        }
    }
    if (tasks.empty()) {
        return 0;
    }
    workerCount = std::min(workerCount, tasks.size());

    std::vector<std::vector<float>> cBuffers(workerCount);
    size_t handled = 0;
    size_t expected = 0;
    for (size_t taskOffset = 0; taskOffset < tasks.size();) {
        const size_t running = std::min(workerCount, tasks.size() - taskOffset);
        for (size_t i = 0; i < running; ++i) {
            const LinearTask& task = tasks[taskOffset + i];
            expected += task.rows;
            const size_t nOffset = task.groupOffset * 8 * GEMM_INT8_UNIT;
            const size_t localCountN = task.groups * 8 * GEMM_INT8_UNIT;
            cBuffers[i].resize(task.rows * countN);
            MNNSpacemitIme2Job job;
            job.gemm = gemm;
            job.blkLen = callKernelBlkLen;
            job.b = packedB->data() + task.groupOffset * bStride;
            job.bZp = useZp ? job.b : nullptr;
            job.c = cBuffers[i].data() + nOffset;
            job.countM = task.rows;
            job.countN = localCountN;
            job.kBlks = kernelKBlocks;
            job.ldc = countN;
            job.doPost = true;
            job.dst = dst + task.rowOffset * GEMM_INT8_UNIT * sizeof(float);
            job.dstStep = dstStep;
            job.dzStart = task.groupOffset * 8;
            job.fullCountN = countN;
            job.residual = residual->values.data();
            job.post = post;
            job.biasPtr = biasPtr;
            job.fp32min = fp32min;
            job.fp32max = fp32max;
            job.skipResidual = skipPostResidual;
            job.packFloatA = true;
            job.packFloatSrc = src;
            job.packInputScale = inputScale;
            job.packQuantScale = quantScale;
            job.packSrcDepthQuad = srcDepthQuad;
            job.packBlockNum = blockNum;
            job.packSrcRows = realCount;
            job.packRowBegin = task.rowOffset;
            if (spinWorker) {
                MNNSpacemitIme2SpinWorker::get(i).startJob(job);
            } else {
                MNNSpacemitIme2Worker::get(i).startJob(job);
            }
        }
        for (size_t i = 0; i < running; ++i) {
            handled += spinWorker ? MNNSpacemitIme2SpinWorker::get(i).wait() : MNNSpacemitIme2Worker::get(i).wait();
        }
        taskOffset += running;
    }
    if (handled != expected) {
        return 0;
    }
    return 1;
}

void MNNSpacemitIme2PrepackW4Weight(void* context, const int8_t* weight, size_t srcDepthQuad, size_t dstDepthQuad,
                                    size_t blockNum) {
    if (context == nullptr || !MNNSpacemitIme2Enabled() || weight == nullptr || srcDepthQuad % 2 != 0 ||
        dstDepthQuad % 8 != 0) {
        return;
    }
    if (MNNSpacemitIme2Gemm() == nullptr) {
        return;
    }
    const size_t kBlocks = (srcDepthQuad / 2) * blockNum;
    if (MNNSpacemitIme2FuseResidualEnabled() && !MNNSpacemitIme2ZpEnabled() && MNNSpacemitIme2HpEnabled() &&
        MNNSpacemitIme2HpM4Enabled() && kBlocks % 8 == 0) {
        (void)MNNGetIme2PackedBHpResidualAllBlocks(weight, srcDepthQuad, dstDepthQuad, blockNum, context);
    }
}

static inline size_t MNNRvvW4DecodePrepackPairBytes() {
    return GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT * 2;
}

static inline bool MNNRvvW4DecodePrepackHotShape(size_t srcDepthQuad, size_t dstDepthQuad, size_t blockNum) {
    return srcDepthQuad == 4 && dstDepthQuad == 32 && (blockNum == 16 || blockNum == 32 || blockNum == 48);
}

static inline void MNNPackRvvW4DecodePair(uint8_t* dst, const uint8_t* weight0, const uint8_t* weight1) {
    const size_t vectorStride = GEMM_INT8_SRC_UNIT * 2;
    for (int row = 0; row < 2; ++row) {
        const uint8_t* weight = row == 0 ? weight0 : weight1;
        const size_t offset = row * GEMM_INT8_SRC_UNIT;
        for (int lane = 0; lane < GEMM_INT8_SRC_UNIT; ++lane) {
            const uint8_t packed02 = weight[lane];
            const uint8_t packed13 = weight[GEMM_INT8_SRC_UNIT + lane];
            dst[offset + lane] = static_cast<uint8_t>(packed02 >> 4);
            dst[vectorStride + offset + lane] = static_cast<uint8_t>(packed13 >> 4);
            dst[2 * vectorStride + offset + lane] = static_cast<uint8_t>(packed02 & 0x0f);
            dst[3 * vectorStride + offset + lane] = static_cast<uint8_t>(packed13 & 0x0f);
        }
    }
}

static std::shared_ptr<std::vector<uint8_t>> MNNGetRvvW4DecodePrepackedPairs(const int8_t* weight, size_t srcDepthQuad,
                                                                             size_t dstDepthQuad, size_t blockNum) {
    // This legacy RVV entry has no ResourceInt8 context, so retaining packed weights would be lifetime-unsafe.
    // Let the caller use the ordinary RVV decode kernel instead.
    (void)weight;
    (void)srcDepthQuad;
    (void)dstDepthQuad;
    (void)blockNum;
    return nullptr;
}

static inline int32_t MNNRVVDotI8U4Vector_16(vint8m1_t vsrc, vuint8m1_t w4, size_t vl) {
    const vint8m1_t vw = __riscv_vreinterpret_v_u8m1_i8m1(w4);
    const vint16m2_t prod = __riscv_vwmul_vv_i16m2(vsrc, vw, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(prod, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum);
}

static inline void MNNRVVDotI8U4_16x4_WithSrc(int32_t* acc, vint8m1_t vsrc, const uint8_t* weight, size_t vl) {
    const vuint8m1_t packed02 = __riscv_vle8_v_u8m1(weight, vl);
    const vuint8m1_t packed13 = __riscv_vle8_v_u8m1(weight + GEMM_INT8_SRC_UNIT, vl);
    acc[0] += MNNRVVDotI8U4Vector_16(vsrc, __riscv_vsrl_vx_u8m1(packed02, 4, vl), vl);
    acc[1] += MNNRVVDotI8U4Vector_16(vsrc, __riscv_vsrl_vx_u8m1(packed13, 4, vl), vl);
    acc[2] += MNNRVVDotI8U4Vector_16(vsrc, __riscv_vand_vx_u8m1(packed02, 0x0f, vl), vl);
    acc[3] += MNNRVVDotI8U4Vector_16(vsrc, __riscv_vand_vx_u8m1(packed13, 0x0f, vl), vl);
}

static inline void MNNRVVAccumI8U4_16x4(vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2, vint16m2_t& acc3,
                                        vint8m1_t vsrc, const uint8_t* weight, size_t vl) {
    const vuint8m1_t packed02 = __riscv_vle8_v_u8m1(weight, vl);
    const vuint8m1_t packed13 = __riscv_vle8_v_u8m1(weight + GEMM_INT8_SRC_UNIT, vl);
    const vint8m1_t w0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed02, 4, vl));
    const vint8m1_t w1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed13, 4, vl));
    const vint8m1_t w2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed02, 0x0f, vl));
    const vint8m1_t w3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed13, 0x0f, vl));
    acc0 = __riscv_vwmacc_vv_i16m2(acc0, vsrc, w0, vl);
    acc1 = __riscv_vwmacc_vv_i16m2(acc1, vsrc, w1, vl);
    acc2 = __riscv_vwmacc_vv_i16m2(acc2, vsrc, w2, vl);
    acc3 = __riscv_vwmacc_vv_i16m2(acc3, vsrc, w3, vl);
}

static inline void MNNRVVAccumI8U4_32x4PairWithSrc(vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2,
                                                   vint16m2_t& acc3, vint8m1_t vsrc, const uint8_t* weight0,
                                                   const uint8_t* weight1, size_t vl16, size_t vl32);

static inline void MNNRVVAccumI8U4_32x4Pair(vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2, vint16m2_t& acc3,
                                            const int8_t* src0, const int8_t* src1, const uint8_t* weight0,
                                            const uint8_t* weight1, size_t vl16, size_t vl32) {
    vint8m1_t vsrc = __riscv_vle8_v_i8m1(src0, vl16);
    vsrc = __riscv_vslideup_vx_i8m1_tu(vsrc, __riscv_vle8_v_i8m1(src1, vl16), GEMM_INT8_SRC_UNIT, vl32);
    MNNRVVAccumI8U4_32x4PairWithSrc(acc0, acc1, acc2, acc3, vsrc, weight0, weight1, vl16, vl32);
}

static inline void MNNRVVAccumI8U4_32x4PairWithSrc(vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2,
                                                   vint16m2_t& acc3, vint8m1_t vsrc, const uint8_t* weight0,
                                                   const uint8_t* weight1, size_t vl16, size_t vl32) {
    vuint8m1_t packed02 = __riscv_vle8_v_u8m1(weight0, vl16);
    packed02 = __riscv_vslideup_vx_u8m1_tu(packed02, __riscv_vle8_v_u8m1(weight1, vl16), GEMM_INT8_SRC_UNIT, vl32);

    vuint8m1_t packed13 = __riscv_vle8_v_u8m1(weight0 + GEMM_INT8_SRC_UNIT, vl16);
    packed13 = __riscv_vslideup_vx_u8m1_tu(packed13, __riscv_vle8_v_u8m1(weight1 + GEMM_INT8_SRC_UNIT, vl16),
                                           GEMM_INT8_SRC_UNIT, vl32);

    const vint8m1_t w0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed02, 4, vl32));
    const vint8m1_t w1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed13, 4, vl32));
    const vint8m1_t w2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed02, 0x0f, vl32));
    const vint8m1_t w3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed13, 0x0f, vl32));
    acc0 = __riscv_vwmacc_vv_i16m2(acc0, vsrc, w0, vl32);
    acc1 = __riscv_vwmacc_vv_i16m2(acc1, vsrc, w1, vl32);
    acc2 = __riscv_vwmacc_vv_i16m2(acc2, vsrc, w2, vl32);
    acc3 = __riscv_vwmacc_vv_i16m2(acc3, vsrc, w3, vl32);
}

static inline void MNNRVVReduceI8U4_32x4Pair(int32_t* acc, vint16m2_t acc0, vint16m2_t acc1, vint16m2_t acc2,
                                             vint16m2_t acc3, vint32m1_t zero32, size_t vl32, size_t vlOne) {
    alignas(16) int32_t groupAcc[GEMM_INT8_UNIT];
    __riscv_vse32_v_i32m1(groupAcc + 0, __riscv_vwredsum_vs_i16m2_i32m1(acc0, zero32, vl32), vlOne);
    __riscv_vse32_v_i32m1(groupAcc + 1, __riscv_vwredsum_vs_i16m2_i32m1(acc1, zero32, vl32), vlOne);
    __riscv_vse32_v_i32m1(groupAcc + 2, __riscv_vwredsum_vs_i16m2_i32m1(acc2, zero32, vl32), vlOne);
    __riscv_vse32_v_i32m1(groupAcc + 3, __riscv_vwredsum_vs_i16m2_i32m1(acc3, zero32, vl32), vlOne);
    acc[0] += groupAcc[0];
    acc[1] += groupAcc[1];
    acc[2] += groupAcc[2];
    acc[3] += groupAcc[3];
}

static inline void MNNRVVAccumI8U4_32x4PairPrepackedWithSrc(vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2,
                                                            vint16m2_t& acc3, vint8m1_t vsrc,
                                                            const uint8_t* packedWeight, size_t vl32) {
    const size_t vectorStride = GEMM_INT8_SRC_UNIT * 2;
    const auto weight = reinterpret_cast<const int8_t*>(packedWeight);
    const vint8m1_t w0 = __riscv_vle8_v_i8m1(weight + 0 * vectorStride, vl32);
    const vint8m1_t w1 = __riscv_vle8_v_i8m1(weight + 1 * vectorStride, vl32);
    const vint8m1_t w2 = __riscv_vle8_v_i8m1(weight + 2 * vectorStride, vl32);
    const vint8m1_t w3 = __riscv_vle8_v_i8m1(weight + 3 * vectorStride, vl32);
    acc0 = __riscv_vwmacc_vv_i16m2(acc0, vsrc, w0, vl32);
    acc1 = __riscv_vwmacc_vv_i16m2(acc1, vsrc, w1, vl32);
    acc2 = __riscv_vwmacc_vv_i16m2(acc2, vsrc, w2, vl32);
    acc3 = __riscv_vwmacc_vv_i16m2(acc3, vsrc, w3, vl32);
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPostPrepack_RVV(int8_t* dst, const int8_t* src,
                                                                        const int8_t* weight, size_t srcDepthQuad,
                                                                        size_t dst_step, size_t dst_depth_quad,
                                                                        const QuanPostTreatParameters* post) {
    const int blockNum = static_cast<int>(post->blockNum);
    if (!MNNRvvW4DecodePrepackHotShape(srcDepthQuad, dst_depth_quad, blockNum)) {
        return false;
    }
    auto packedWeights = MNNGetRvvW4DecodePrepackedPairs(weight, srcDepthQuad, dst_depth_quad, post->blockNum);
    if (packedWeights == nullptr) {
        return false;
    }

    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * static_cast<int>(srcDepthQuad) + 4 * 2 * GEMM_INT8_UNIT;
    const float* biasPtr = post->biasFloat;
    const float fastInputScale = post->inputScale[0];
    const float fp32min = post->fp32minmax[0];
    const float fp32max = post->fp32minmax[1];
    const bool needClamp = !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const size_t pairCount = srcDepthQuad / 2;
    const size_t pairBytes = MNNRvvW4DecodePrepackPairBytes();
    const size_t vl16 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const size_t vl32 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT * 2);
    const size_t vlOne = __riscv_vsetvl_e32m1(1);
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vl32);
    const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, vlOne);

    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); ++dz) {
        vfloat32m1_t partial = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
        auto dst_x = reinterpret_cast<float*>(dst + dz * dst_step);
        const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;
        for (int bk = 0; bk < blockNum; ++bk) {
            const int8_t* srcBk = src + bk * static_cast<int>(srcDepthQuad) * GEMM_INT8_SRC_UNIT;
            const auto weightDz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
            const uint8_t* packedPairBase = packedWeights->data() + ((dz * blockNum + bk) * pairCount) * pairBytes;

            vint8m1_t vsrc01 = __riscv_vle8_v_i8m1(srcBk, vl16);
            vsrc01 = __riscv_vslideup_vx_i8m1_tu(vsrc01, __riscv_vle8_v_i8m1(srcBk + GEMM_INT8_SRC_UNIT, vl16),
                                                 GEMM_INT8_SRC_UNIT, vl32);
            vint8m1_t vsrc23 = __riscv_vle8_v_i8m1(srcBk + 2 * GEMM_INT8_SRC_UNIT, vl16);
            vsrc23 = __riscv_vslideup_vx_i8m1_tu(vsrc23, __riscv_vle8_v_i8m1(srcBk + 3 * GEMM_INT8_SRC_UNIT, vl16),
                                                 GEMM_INT8_SRC_UNIT, vl32);

            alignas(16) int32_t acc[GEMM_INT8_UNIT] = {0, 0, 0, 0};
            vint16m2_t vacc0 = zero16;
            vint16m2_t vacc1 = zero16;
            vint16m2_t vacc2 = zero16;
            vint16m2_t vacc3 = zero16;
            MNNRVVAccumI8U4_32x4PairPrepackedWithSrc(vacc0, vacc1, vacc2, vacc3, vsrc01, packedPairBase, vl32);
            MNNRVVAccumI8U4_32x4PairPrepackedWithSrc(vacc0, vacc1, vacc2, vacc3, vsrc23, packedPairBase + pairBytes,
                                                     vl32);
            MNNRVVReduceI8U4_32x4Pair(acc, vacc0, vacc1, vacc2, vacc3, zero32, vl32, vlOne);

            const float* scale_dz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weight_step_Y);
            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            vfloat32m1_t value = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc, vl4), vl4);
            value = __riscv_vfmul_vv_f32m1(value, __riscv_vle32_v_f32m1(scale_dz, vl4), vl4);
            value = __riscv_vfmul_vf_f32m1(value, fastInputScale, vl4);
            value =
                __riscv_vfmacc_vf_f32m1(value, post->srcKernelSum[bk], __riscv_vle32_v_f32m1(weightBias_dz, vl4), vl4);
            partial = __riscv_vfadd_vv_f32m1(partial, value, vl4);
        }
        partial = __riscv_vfadd_vv_f32m1(partial, __riscv_vle32_v_f32m1(bias_dz, vl4), vl4);
        if (needClamp) {
            partial = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial, fp32min, vl4), fp32max, vl4);
        }
        __riscv_vse32_v_f32m1(dst_x, partial, vl4);
    }

    return true;
}

static inline void MNNRVVAccumI8U4_32x4PairLoop(int32_t* acc, vint16m2_t& acc0, vint16m2_t& acc1, vint16m2_t& acc2,
                                                vint16m2_t& acc3, vint16m2_t zero16, const int8_t* src,
                                                int srcDepthStride, const int8_t* weight, int weightStepY,
                                                int srcDepthQuad, vint32m1_t zero32, size_t vl16, size_t vl32,
                                                size_t vlOne) {
    int pairs = 0;
    for (int sz = 0; sz < srcDepthQuad; sz += 2) {
        const auto src0 = src + sz * srcDepthStride;
        const auto src1 = src0 + srcDepthStride;
        const auto weight0 = reinterpret_cast<const uint8_t*>(weight + weightStepY * sz);
        const auto weight1 = weight0 + weightStepY;
        MNNRVVAccumI8U4_32x4Pair(acc0, acc1, acc2, acc3, src0, src1, weight0, weight1, vl16, vl32);
        ++pairs;
        if (pairs == 16) {
            MNNRVVReduceI8U4_32x4Pair(acc, acc0, acc1, acc2, acc3, zero32, vl32, vlOne);
            acc0 = zero16;
            acc1 = zero16;
            acc2 = zero16;
            acc3 = zero16;
            pairs = 0;
        }
    }
    if (pairs > 0) {
        MNNRVVReduceI8U4_32x4Pair(acc, acc0, acc1, acc2, acc3, zero32, vl32, vlOne);
        acc0 = zero16;
        acc1 = zero16;
        acc2 = zero16;
        acc3 = zero16;
    }
}

static inline void MNNRVVDotI8U4_16x4(int32_t* acc, const int8_t* src, const uint8_t* weight) {
    const size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const vint8m1_t vsrc = __riscv_vle8_v_i8m1(src, vl);
    MNNRVVDotI8U4_16x4_WithSrc(acc, vsrc, weight, vl);
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPostDz2_RVV(int8_t* dst, const int8_t* src,
                                                                    const int8_t* weight, size_t srcDepthQuad,
                                                                    size_t dst_step, size_t dst_depth_quad,
                                                                    const QuanPostTreatParameters* post) {
    if (srcDepthQuad != 4) {
        return false;
    }
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * static_cast<int>(srcDepthQuad) + 4 * 2 * GEMM_INT8_UNIT;
    const int blockNum = static_cast<int>(post->blockNum);
    const float fastInputScale = post->inputScale[0];
    const float fp32min = post->fp32minmax[0];
    const float fp32max = post->fp32minmax[1];
    const bool needClamp = !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const size_t vl16 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const size_t vl32 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT * 2);
    const size_t vlOne = __riscv_vsetvl_e32m1(1);
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vl32);
    const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, vlOne);

    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); dz += 2) {
        const bool hasDz1 = dz + 1 < static_cast<int>(dst_depth_quad);
        vfloat32m1_t partial0 = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
        vfloat32m1_t partial1 = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
        for (int bk = 0; bk < blockNum; ++bk) {
            const int8_t* srcBk = src + bk * static_cast<int>(srcDepthQuad) * GEMM_INT8_SRC_UNIT;
            vint8m1_t vsrc01 = __riscv_vle8_v_i8m1(srcBk, vl16);
            vsrc01 = __riscv_vslideup_vx_i8m1_tu(vsrc01, __riscv_vle8_v_i8m1(srcBk + GEMM_INT8_SRC_UNIT, vl16),
                                                 GEMM_INT8_SRC_UNIT, vl32);
            vint8m1_t vsrc23 = __riscv_vle8_v_i8m1(srcBk + 2 * GEMM_INT8_SRC_UNIT, vl16);
            vsrc23 = __riscv_vslideup_vx_i8m1_tu(vsrc23, __riscv_vle8_v_i8m1(srcBk + 3 * GEMM_INT8_SRC_UNIT, vl16),
                                                 GEMM_INT8_SRC_UNIT, vl32);

            alignas(16) int32_t acc0[GEMM_INT8_UNIT] = {0, 0, 0, 0};
            const auto weightDz0 = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
            vint16m2_t vacc00 = zero16;
            vint16m2_t vacc01 = zero16;
            vint16m2_t vacc02 = zero16;
            vint16m2_t vacc03 = zero16;
            MNNRVVAccumI8U4_32x4PairWithSrc(vacc00, vacc01, vacc02, vacc03, vsrc01,
                                            reinterpret_cast<const uint8_t*>(weightDz0),
                                            reinterpret_cast<const uint8_t*>(weightDz0 + weight_step_Y), vl16, vl32);
            MNNRVVAccumI8U4_32x4PairWithSrc(
                vacc00, vacc01, vacc02, vacc03, vsrc23, reinterpret_cast<const uint8_t*>(weightDz0 + 2 * weight_step_Y),
                reinterpret_cast<const uint8_t*>(weightDz0 + 3 * weight_step_Y), vl16, vl32);
            MNNRVVReduceI8U4_32x4Pair(acc0, vacc00, vacc01, vacc02, vacc03, zero32, vl32, vlOne);
            const float* scale0 = reinterpret_cast<const float*>(weightDz0 + srcDepthQuad * weight_step_Y);
            vfloat32m1_t value0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc0, vl4), vl4);
            value0 = __riscv_vfmul_vv_f32m1(value0, __riscv_vle32_v_f32m1(scale0, vl4), vl4);
            value0 = __riscv_vfmul_vf_f32m1(value0, fastInputScale, vl4);
            value0 = __riscv_vfmacc_vf_f32m1(value0, post->srcKernelSum[bk],
                                             __riscv_vle32_v_f32m1(scale0 + GEMM_INT8_UNIT, vl4), vl4);
            partial0 = __riscv_vfadd_vv_f32m1(partial0, value0, vl4);

            if (hasDz1) {
                alignas(16) int32_t acc1[GEMM_INT8_UNIT] = {0, 0, 0, 0};
                const auto weightDz1 = weightDz0 + blockNum * weight_step_Z;
                vint16m2_t vacc10 = zero16;
                vint16m2_t vacc11 = zero16;
                vint16m2_t vacc12 = zero16;
                vint16m2_t vacc13 = zero16;
                MNNRVVAccumI8U4_32x4PairWithSrc(
                    vacc10, vacc11, vacc12, vacc13, vsrc01, reinterpret_cast<const uint8_t*>(weightDz1),
                    reinterpret_cast<const uint8_t*>(weightDz1 + weight_step_Y), vl16, vl32);
                MNNRVVAccumI8U4_32x4PairWithSrc(vacc10, vacc11, vacc12, vacc13, vsrc23,
                                                reinterpret_cast<const uint8_t*>(weightDz1 + 2 * weight_step_Y),
                                                reinterpret_cast<const uint8_t*>(weightDz1 + 3 * weight_step_Y), vl16,
                                                vl32);
                MNNRVVReduceI8U4_32x4Pair(acc1, vacc10, vacc11, vacc12, vacc13, zero32, vl32, vlOne);
                const float* scale1 = reinterpret_cast<const float*>(weightDz1 + srcDepthQuad * weight_step_Y);
                vfloat32m1_t value1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc1, vl4), vl4);
                value1 = __riscv_vfmul_vv_f32m1(value1, __riscv_vle32_v_f32m1(scale1, vl4), vl4);
                value1 = __riscv_vfmul_vf_f32m1(value1, fastInputScale, vl4);
                value1 = __riscv_vfmacc_vf_f32m1(value1, post->srcKernelSum[bk],
                                                 __riscv_vle32_v_f32m1(scale1 + GEMM_INT8_UNIT, vl4), vl4);
                partial1 = __riscv_vfadd_vv_f32m1(partial1, value1, vl4);
            }
        }
        partial0 =
            __riscv_vfadd_vv_f32m1(partial0, __riscv_vle32_v_f32m1(post->biasFloat + dz * GEMM_INT8_UNIT, vl4), vl4);
        if (needClamp) {
            partial0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial0, fp32min, vl4), fp32max, vl4);
        }
        __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst + dz * dst_step), partial0, vl4);
        if (hasDz1) {
            partial1 = __riscv_vfadd_vv_f32m1(
                partial1, __riscv_vle32_v_f32m1(post->biasFloat + (dz + 1) * GEMM_INT8_UNIT, vl4), vl4);
            if (needClamp) {
                partial1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial1, fp32min, vl4), fp32max, vl4);
            }
            __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst + (dz + 1) * dst_step), partial1, vl4);
        }
    }
    return true;
}

void MNNSpacemitIme2GemmInt8AddBiasScaleW8(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                           size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                           size_t realCount) {
    const int bytes = (post->useInt8 == 1) ? 1 : 4;

    float fp32min = 0.f, fp32max = 0.f;
    if (post->useInt8 == 0 && post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    const int weight_step_Z = src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) + 4 * 2 * GEMM_INT8_UNIT;

    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);

    float* biasPtr = (float*)post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z = dst + dz * dst_step;

        for (int bk = 0; bk < blockNum; ++bk) {
            const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;

            const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);

            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;

            const auto srcSumPtr = post->srcKernelSum + bk * realCount;

            const auto inputScalePtr = post->inputBias ? post->inputScale + bk * realCount : post->inputScale;

            for (int w = 0; w < realCount; ++w) {
                const auto src_x = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount + w * GEMM_INT8_SRC_UNIT;

                auto dst_x = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x = accumbuff + w * GEMM_INT8_UNIT;

                int32_t acc[4] = {0, 0, 0, 0};

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                    size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);

                    vint8m1_t vsrc = __riscv_vle8_v_i8m1(src_z, vl);

                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;

                        vint8m1_t vw = __riscv_vle8_v_i8m1(weight_j, vl);

                        vint16m2_t prod = __riscv_vwmul_vv_i16m2(vsrc, vw, vl);

                        vint32m1_t sum = __riscv_vwredsum_vs_i16m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

                        acc[j] += __riscv_vmv_x_s_i32m1_i32(sum);
                    }
                }

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    float value = acc[j] * scale_dz[j] + srcSumPtr[w] * weightBias_dz[j];

                    if (post->inputScale) {
                        value = acc[j] * scale_dz[j] * inputScalePtr[w] + srcSumPtr[w] * weightBias_dz[j];
                    }

                    if (post->inputBias) {
                        auto weightKernelSum =
                            post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;

                        value += (post->inputBias[bk * realCount + w] * weightKernelSum[j]);
                    }

                    if (post->useInt8 == 0) {
                        if (bk > 0) {
                            value += ((float*)accum_x)[j];
                        }

                        if (bk == blockNum - 1) {
                            if (biasPtr) {
                                value += bias_dz[j];
                            }

                            if (post->fp32minmax) {
                                value = std::min(std::max(fp32min, value), fp32max);
                            }

                            ((float*)dst_x)[j] = value;
                        } else {
                            ((float*)accum_x)[j] = value;
                        }
                    } else {
                        value += bias_dz[j];

                        value = std::max(value, (float)post->minValue);
                        value = std::min(value, (float)post->maxValue);

                        dst_x[j] = (int8_t)roundf(value);
                    }
                }
            }
        }
    }
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPost_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                                 size_t srcDepthQuad, size_t dst_step,
                                                                 size_t dst_depth_quad,
                                                                 const QuanPostTreatParameters* post) {
    const float* biasPtr = post->biasFloat;
    if ((srcDepthQuad != 4 && srcDepthQuad != 8 && srcDepthQuad != 16 && srcDepthQuad != 32 && srcDepthQuad != 64 &&
         srcDepthQuad != 128) ||
        post->inputScale == nullptr || post->inputBias != nullptr || biasPtr == nullptr ||
        post->fp32minmax == nullptr) {
        return false;
    }
    if (MNNRvvW4DecodePrepackEnabled() && MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPostPrepack_RVV(
                                              dst, src, weight, srcDepthQuad, dst_step, dst_depth_quad, post)) {
        return true;
    }
    if (MNNRvvW4DecodeDz2Enabled() && MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPostDz2_RVV(
                                          dst, src, weight, srcDepthQuad, dst_step, dst_depth_quad, post)) {
        return true;
    }
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * static_cast<int>(srcDepthQuad) + 4 * 2 * GEMM_INT8_UNIT;
    const int blockNum = static_cast<int>(post->blockNum);
    const float fastInputScale = post->inputScale[0];
    const float fp32min = post->fp32minmax[0];
    const float fp32max = post->fp32minmax[1];
    const bool needClamp = !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const size_t vlPair = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT * 2);
    const size_t vlOne = __riscv_vsetvl_e32m1(1);
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vlPair);
    const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, vlOne);
    const bool useCenteredPost = MNNRvvW4UseCenteredPost(weight, srcDepthQuad, dst_depth_quad, post->blockNum);
    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); ++dz) {
        vfloat32m1_t partial = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
        auto dst_x = reinterpret_cast<float*>(dst + dz * dst_step);
        const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;
        for (int bk = 0; bk < blockNum; ++bk) {
            const int8_t* srcBk = src + bk * srcDepthQuad * GEMM_INT8_SRC_UNIT;
            const auto weightDz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
            alignas(16) int32_t acc[GEMM_INT8_UNIT] = {0, 0, 0, 0};
            vint16m2_t vacc0 = zero16;
            vint16m2_t vacc1 = zero16;
            vint16m2_t vacc2 = zero16;
            vint16m2_t vacc3 = zero16;

#define MNN_RVV_ACC_S4_PAIR(SZ)                                                                         \
    do {                                                                                                \
        const auto src0 = srcBk + (SZ) * GEMM_INT8_SRC_UNIT;                                            \
        const auto src1 = src0 + GEMM_INT8_SRC_UNIT;                                                    \
        const auto weight0 = reinterpret_cast<const uint8_t*>(weightDz + weight_step_Y * (SZ));         \
        const auto weight1 = weight0 + weight_step_Y;                                                   \
        MNNRVVAccumI8U4_32x4Pair(vacc0, vacc1, vacc2, vacc3, src0, src1, weight0, weight1, vl, vlPair); \
    } while (false)
#define MNN_RVV_REDUCE_S4_PAIR()                                                                            \
    do {                                                                                                    \
        alignas(16) int32_t groupAcc[GEMM_INT8_UNIT];                                                       \
        __riscv_vse32_v_i32m1(groupAcc + 0, __riscv_vwredsum_vs_i16m2_i32m1(vacc0, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 1, __riscv_vwredsum_vs_i16m2_i32m1(vacc1, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 2, __riscv_vwredsum_vs_i16m2_i32m1(vacc2, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 3, __riscv_vwredsum_vs_i16m2_i32m1(vacc3, zero32, vlPair), vlOne); \
        acc[0] += groupAcc[0];                                                                              \
        acc[1] += groupAcc[1];                                                                              \
        acc[2] += groupAcc[2];                                                                              \
        acc[3] += groupAcc[3];                                                                              \
    } while (false)
#define MNN_RVV_RESET_S4_PAIR() \
    do {                        \
        vacc0 = zero16;         \
        vacc1 = zero16;         \
        vacc2 = zero16;         \
        vacc3 = zero16;         \
    } while (false)
            if (srcDepthQuad == 32) {
                MNN_RVV_ACC_S4_PAIR(0);
                MNN_RVV_ACC_S4_PAIR(2);
                MNN_RVV_ACC_S4_PAIR(4);
                MNN_RVV_ACC_S4_PAIR(6);
                MNN_RVV_ACC_S4_PAIR(8);
                MNN_RVV_ACC_S4_PAIR(10);
                MNN_RVV_ACC_S4_PAIR(12);
                MNN_RVV_ACC_S4_PAIR(14);
                MNN_RVV_ACC_S4_PAIR(16);
                MNN_RVV_ACC_S4_PAIR(18);
                MNN_RVV_ACC_S4_PAIR(20);
                MNN_RVV_ACC_S4_PAIR(22);
                MNN_RVV_ACC_S4_PAIR(24);
                MNN_RVV_ACC_S4_PAIR(26);
                MNN_RVV_ACC_S4_PAIR(28);
                MNN_RVV_ACC_S4_PAIR(30);
            } else if (srcDepthQuad == 4) {
                MNN_RVV_ACC_S4_PAIR(0);
                MNN_RVV_ACC_S4_PAIR(2);
            } else if (srcDepthQuad == 8) {
                MNN_RVV_ACC_S4_PAIR(0);
                MNN_RVV_ACC_S4_PAIR(2);
                MNN_RVV_ACC_S4_PAIR(4);
                MNN_RVV_ACC_S4_PAIR(6);
            } else if (srcDepthQuad == 16) {
                MNN_RVV_ACC_S4_PAIR(0);
                MNN_RVV_ACC_S4_PAIR(2);
                MNN_RVV_ACC_S4_PAIR(4);
                MNN_RVV_ACC_S4_PAIR(6);
                MNN_RVV_ACC_S4_PAIR(8);
                MNN_RVV_ACC_S4_PAIR(10);
                MNN_RVV_ACC_S4_PAIR(12);
                MNN_RVV_ACC_S4_PAIR(14);
            } else if (srcDepthQuad == 64) {
                MNN_RVV_ACC_S4_PAIR(0);
                MNN_RVV_ACC_S4_PAIR(2);
                MNN_RVV_ACC_S4_PAIR(4);
                MNN_RVV_ACC_S4_PAIR(6);
                MNN_RVV_ACC_S4_PAIR(8);
                MNN_RVV_ACC_S4_PAIR(10);
                MNN_RVV_ACC_S4_PAIR(12);
                MNN_RVV_ACC_S4_PAIR(14);
                MNN_RVV_ACC_S4_PAIR(16);
                MNN_RVV_ACC_S4_PAIR(18);
                MNN_RVV_ACC_S4_PAIR(20);
                MNN_RVV_ACC_S4_PAIR(22);
                MNN_RVV_ACC_S4_PAIR(24);
                MNN_RVV_ACC_S4_PAIR(26);
                MNN_RVV_ACC_S4_PAIR(28);
                MNN_RVV_ACC_S4_PAIR(30);
                MNN_RVV_REDUCE_S4_PAIR();
                MNN_RVV_RESET_S4_PAIR();
                MNN_RVV_ACC_S4_PAIR(32);
                MNN_RVV_ACC_S4_PAIR(34);
                MNN_RVV_ACC_S4_PAIR(36);
                MNN_RVV_ACC_S4_PAIR(38);
                MNN_RVV_ACC_S4_PAIR(40);
                MNN_RVV_ACC_S4_PAIR(42);
                MNN_RVV_ACC_S4_PAIR(44);
                MNN_RVV_ACC_S4_PAIR(46);
                MNN_RVV_ACC_S4_PAIR(48);
                MNN_RVV_ACC_S4_PAIR(50);
                MNN_RVV_ACC_S4_PAIR(52);
                MNN_RVV_ACC_S4_PAIR(54);
                MNN_RVV_ACC_S4_PAIR(56);
                MNN_RVV_ACC_S4_PAIR(58);
                MNN_RVV_ACC_S4_PAIR(60);
                MNN_RVV_ACC_S4_PAIR(62);
            } else {
                MNNRVVAccumI8U4_32x4PairLoop(acc, vacc0, vacc1, vacc2, vacc3, zero16, srcBk, GEMM_INT8_SRC_UNIT,
                                             weightDz, weight_step_Y, static_cast<int>(srcDepthQuad), zero32, vl,
                                             vlPair, vlOne);
            }
            MNN_RVV_REDUCE_S4_PAIR();
            const vint32m1_t accVec = __riscv_vle32_v_i32m1(acc, vl4);
#undef MNN_RVV_ACC_S4_PAIR
#undef MNN_RVV_REDUCE_S4_PAIR
#undef MNN_RVV_RESET_S4_PAIR

            const float* scale_dz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weight_step_Y);
            vfloat32m1_t value = __riscv_vfcvt_f_x_v_f32m1(accVec, vl4);
            const vfloat32m1_t scaleVec = __riscv_vle32_v_f32m1(scale_dz, vl4);
            if (useCenteredPost) {
                value = __riscv_vfmul_vf_f32m1(value, fastInputScale, vl4);
                value = __riscv_vfadd_vf_f32m1(value, -8.0f * post->srcKernelSum[bk], vl4);
                value = __riscv_vfmul_vv_f32m1(value, scaleVec, vl4);
            } else {
                const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
                value = __riscv_vfmul_vv_f32m1(value, scaleVec, vl4);
                value = __riscv_vfmul_vf_f32m1(value, fastInputScale, vl4);
                value = __riscv_vfmacc_vf_f32m1(value, post->srcKernelSum[bk],
                                                __riscv_vle32_v_f32m1(weightBias_dz, vl4), vl4);
            }
            partial = __riscv_vfadd_vv_f32m1(partial, value, vl4);
        }
        partial = __riscv_vfadd_vv_f32m1(partial, __riscv_vle32_v_f32m1(bias_dz, vl4), vl4);
        if (needClamp) {
            partial = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial, fp32min, vl4), fp32max, vl4);
        }
        __riscv_vse32_v_f32m1(dst_x, partial, vl4);
    }
    return true;
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_BatchS4FastPostDz2_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                                   size_t srcDepthQuad, size_t dst_step,
                                                                   size_t dst_depth_quad,
                                                                   const QuanPostTreatParameters* post,
                                                                   size_t realCount) {
    if (srcDepthQuad != 4 || realCount <= 1) {
        return false;
    }
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * static_cast<int>(srcDepthQuad) + 4 * 2 * GEMM_INT8_UNIT;
    const int blockNum = static_cast<int>(post->blockNum);
    const float fp32min = post->fp32minmax[0];
    const float fp32max = post->fp32minmax[1];
    const bool needClamp = !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const size_t vl16 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const size_t vl32 = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT * 2);
    const size_t vlOne = __riscv_vsetvl_e32m1(1);
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vl32);
    const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, vlOne);
    const int srcBlockStride = static_cast<int>(srcDepthQuad) * GEMM_INT8_SRC_UNIT * static_cast<int>(realCount);
    const int srcDepthStride = GEMM_INT8_SRC_UNIT * static_cast<int>(realCount);

    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); dz += 2) {
        const bool hasDz1 = dz + 1 < static_cast<int>(dst_depth_quad);
        auto dst0 = dst + dz * dst_step;
        auto dst1 = hasDz1 ? dst + (dz + 1) * dst_step : nullptr;
        for (int w = 0; w < static_cast<int>(realCount); ++w) {
            vfloat32m1_t partial0 = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
            vfloat32m1_t partial1 = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
            for (int bk = 0; bk < blockNum; ++bk) {
                const int8_t* srcBk = src + bk * srcBlockStride + w * GEMM_INT8_SRC_UNIT;
                vint8m1_t vsrc01 = __riscv_vle8_v_i8m1(srcBk, vl16);
                vsrc01 = __riscv_vslideup_vx_i8m1_tu(vsrc01, __riscv_vle8_v_i8m1(srcBk + srcDepthStride, vl16),
                                                     GEMM_INT8_SRC_UNIT, vl32);
                vint8m1_t vsrc23 = __riscv_vle8_v_i8m1(srcBk + 2 * srcDepthStride, vl16);
                vsrc23 = __riscv_vslideup_vx_i8m1_tu(vsrc23, __riscv_vle8_v_i8m1(srcBk + 3 * srcDepthStride, vl16),
                                                     GEMM_INT8_SRC_UNIT, vl32);
                const float srcSum = post->srcKernelSum[bk * realCount + w];
                const float inputScale = post->inputScale[w];

                alignas(16) int32_t acc0[GEMM_INT8_UNIT] = {0, 0, 0, 0};
                const auto weightDz0 = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
                vint16m2_t vacc00 = zero16;
                vint16m2_t vacc01 = zero16;
                vint16m2_t vacc02 = zero16;
                vint16m2_t vacc03 = zero16;
                MNNRVVAccumI8U4_32x4PairWithSrc(
                    vacc00, vacc01, vacc02, vacc03, vsrc01, reinterpret_cast<const uint8_t*>(weightDz0),
                    reinterpret_cast<const uint8_t*>(weightDz0 + weight_step_Y), vl16, vl32);
                MNNRVVAccumI8U4_32x4PairWithSrc(vacc00, vacc01, vacc02, vacc03, vsrc23,
                                                reinterpret_cast<const uint8_t*>(weightDz0 + 2 * weight_step_Y),
                                                reinterpret_cast<const uint8_t*>(weightDz0 + 3 * weight_step_Y), vl16,
                                                vl32);
                MNNRVVReduceI8U4_32x4Pair(acc0, vacc00, vacc01, vacc02, vacc03, zero32, vl32, vlOne);
                const float* scale0 = reinterpret_cast<const float*>(weightDz0 + srcDepthQuad * weight_step_Y);
                vfloat32m1_t value0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc0, vl4), vl4);
                value0 = __riscv_vfmul_vv_f32m1(value0, __riscv_vle32_v_f32m1(scale0, vl4), vl4);
                value0 = __riscv_vfmul_vf_f32m1(value0, inputScale, vl4);
                value0 =
                    __riscv_vfmacc_vf_f32m1(value0, srcSum, __riscv_vle32_v_f32m1(scale0 + GEMM_INT8_UNIT, vl4), vl4);
                partial0 = __riscv_vfadd_vv_f32m1(partial0, value0, vl4);

                if (hasDz1) {
                    alignas(16) int32_t acc1[GEMM_INT8_UNIT] = {0, 0, 0, 0};
                    const auto weightDz1 = weightDz0 + blockNum * weight_step_Z;
                    vint16m2_t vacc10 = zero16;
                    vint16m2_t vacc11 = zero16;
                    vint16m2_t vacc12 = zero16;
                    vint16m2_t vacc13 = zero16;
                    MNNRVVAccumI8U4_32x4PairWithSrc(
                        vacc10, vacc11, vacc12, vacc13, vsrc01, reinterpret_cast<const uint8_t*>(weightDz1),
                        reinterpret_cast<const uint8_t*>(weightDz1 + weight_step_Y), vl16, vl32);
                    MNNRVVAccumI8U4_32x4PairWithSrc(vacc10, vacc11, vacc12, vacc13, vsrc23,
                                                    reinterpret_cast<const uint8_t*>(weightDz1 + 2 * weight_step_Y),
                                                    reinterpret_cast<const uint8_t*>(weightDz1 + 3 * weight_step_Y),
                                                    vl16, vl32);
                    MNNRVVReduceI8U4_32x4Pair(acc1, vacc10, vacc11, vacc12, vacc13, zero32, vl32, vlOne);
                    const float* scale1 = reinterpret_cast<const float*>(weightDz1 + srcDepthQuad * weight_step_Y);
                    vfloat32m1_t value1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc1, vl4), vl4);
                    value1 = __riscv_vfmul_vv_f32m1(value1, __riscv_vle32_v_f32m1(scale1, vl4), vl4);
                    value1 = __riscv_vfmul_vf_f32m1(value1, inputScale, vl4);
                    value1 = __riscv_vfmacc_vf_f32m1(value1, srcSum,
                                                     __riscv_vle32_v_f32m1(scale1 + GEMM_INT8_UNIT, vl4), vl4);
                    partial1 = __riscv_vfadd_vv_f32m1(partial1, value1, vl4);
                }
            }
            partial0 = __riscv_vfadd_vv_f32m1(partial0,
                                              __riscv_vle32_v_f32m1(post->biasFloat + dz * GEMM_INT8_UNIT, vl4), vl4);
            if (needClamp) {
                partial0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial0, fp32min, vl4), fp32max, vl4);
            }
            __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst0 + w * GEMM_INT8_UNIT * 4), partial0, vl4);
            if (hasDz1) {
                partial1 = __riscv_vfadd_vv_f32m1(
                    partial1, __riscv_vle32_v_f32m1(post->biasFloat + (dz + 1) * GEMM_INT8_UNIT, vl4), vl4);
                if (needClamp) {
                    partial1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial1, fp32min, vl4), fp32max, vl4);
                }
                __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst1 + w * GEMM_INT8_UNIT * 4), partial1, vl4);
            }
        }
    }
    return true;
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_BatchS4FastPost_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                                size_t srcDepthQuad, size_t dst_step,
                                                                size_t dst_depth_quad,
                                                                const QuanPostTreatParameters* post, size_t realCount) {
    const float* biasPtr = post->biasFloat;
    if ((srcDepthQuad != 2 && srcDepthQuad != 4 && srcDepthQuad != 8 && srcDepthQuad != 16 && srcDepthQuad != 32 &&
         srcDepthQuad != 64 && srcDepthQuad != 128) ||
        realCount <= 1 || post->inputScale == nullptr || post->inputBias != nullptr || biasPtr == nullptr ||
        post->fp32minmax == nullptr) {
        return false;
    }
    if (MNNRvvW4BatchDz2Enabled() && MNNGemmInt8AddBiasScale_16x4_w4_BatchS4FastPostDz2_RVV(
                                         dst, src, weight, srcDepthQuad, dst_step, dst_depth_quad, post, realCount)) {
        return true;
    }

    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * static_cast<int>(srcDepthQuad) + 4 * 2 * GEMM_INT8_UNIT;
    const int blockNum = static_cast<int>(post->blockNum);
    const float fp32min = post->fp32minmax[0];
    const float fp32max = post->fp32minmax[1];
    const bool needClamp = !MNNRvvFp32MinMaxIsFullRange(fp32min, fp32max);
    const size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
    const size_t vlPair = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT * 2);
    const size_t vlOne = __riscv_vsetvl_e32m1(1);
    const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
    const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vlPair);
    const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, vlOne);
    const int srcBlockStride = srcDepthQuad * GEMM_INT8_SRC_UNIT * static_cast<int>(realCount);
    const int srcDepthStride = GEMM_INT8_SRC_UNIT * static_cast<int>(realCount);

    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); ++dz) {
        const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;
        auto dst_z = dst + dz * dst_step;
        for (int w = 0; w < static_cast<int>(realCount); ++w) {
            vfloat32m1_t partial = __riscv_vfmv_v_f_f32m1(0.0f, vl4);
            for (int bk = 0; bk < blockNum; ++bk) {
                alignas(16) int32_t acc[GEMM_INT8_UNIT] = {0, 0, 0, 0};
                const int8_t* srcBk = src + bk * srcBlockStride + w * GEMM_INT8_SRC_UNIT;
                const auto weightDz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
                vint16m2_t vacc0 = zero16;
                vint16m2_t vacc1 = zero16;
                vint16m2_t vacc2 = zero16;
                vint16m2_t vacc3 = zero16;

#define MNN_RVV_ACC_BATCH_S4_PAIR(SZ)                                                                   \
    do {                                                                                                \
        const auto src0 = srcBk + (SZ) * srcDepthStride;                                                \
        const auto src1 = src0 + srcDepthStride;                                                        \
        const auto weight0 = reinterpret_cast<const uint8_t*>(weightDz + weight_step_Y * (SZ));         \
        const auto weight1 = weight0 + weight_step_Y;                                                   \
        MNNRVVAccumI8U4_32x4Pair(vacc0, vacc1, vacc2, vacc3, src0, src1, weight0, weight1, vl, vlPair); \
    } while (false)
#define MNN_RVV_REDUCE_BATCH_S4_PAIR()                                                                      \
    do {                                                                                                    \
        alignas(16) int32_t groupAcc[GEMM_INT8_UNIT];                                                       \
        __riscv_vse32_v_i32m1(groupAcc + 0, __riscv_vwredsum_vs_i16m2_i32m1(vacc0, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 1, __riscv_vwredsum_vs_i16m2_i32m1(vacc1, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 2, __riscv_vwredsum_vs_i16m2_i32m1(vacc2, zero32, vlPair), vlOne); \
        __riscv_vse32_v_i32m1(groupAcc + 3, __riscv_vwredsum_vs_i16m2_i32m1(vacc3, zero32, vlPair), vlOne); \
        acc[0] += groupAcc[0];                                                                              \
        acc[1] += groupAcc[1];                                                                              \
        acc[2] += groupAcc[2];                                                                              \
        acc[3] += groupAcc[3];                                                                              \
    } while (false)
#define MNN_RVV_RESET_BATCH_S4_PAIR() \
    do {                              \
        vacc0 = zero16;               \
        vacc1 = zero16;               \
        vacc2 = zero16;               \
        vacc3 = zero16;               \
    } while (false)
                if (srcDepthQuad == 32) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                    MNN_RVV_ACC_BATCH_S4_PAIR(2);
                    MNN_RVV_ACC_BATCH_S4_PAIR(4);
                    MNN_RVV_ACC_BATCH_S4_PAIR(6);
                    MNN_RVV_ACC_BATCH_S4_PAIR(8);
                    MNN_RVV_ACC_BATCH_S4_PAIR(10);
                    MNN_RVV_ACC_BATCH_S4_PAIR(12);
                    MNN_RVV_ACC_BATCH_S4_PAIR(14);
                    MNN_RVV_ACC_BATCH_S4_PAIR(16);
                    MNN_RVV_ACC_BATCH_S4_PAIR(18);
                    MNN_RVV_ACC_BATCH_S4_PAIR(20);
                    MNN_RVV_ACC_BATCH_S4_PAIR(22);
                    MNN_RVV_ACC_BATCH_S4_PAIR(24);
                    MNN_RVV_ACC_BATCH_S4_PAIR(26);
                    MNN_RVV_ACC_BATCH_S4_PAIR(28);
                    MNN_RVV_ACC_BATCH_S4_PAIR(30);
                } else if (srcDepthQuad == 2) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                } else if (srcDepthQuad == 4) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                    MNN_RVV_ACC_BATCH_S4_PAIR(2);
                } else if (srcDepthQuad == 8) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                    MNN_RVV_ACC_BATCH_S4_PAIR(2);
                    MNN_RVV_ACC_BATCH_S4_PAIR(4);
                    MNN_RVV_ACC_BATCH_S4_PAIR(6);
                } else if (srcDepthQuad == 16) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                    MNN_RVV_ACC_BATCH_S4_PAIR(2);
                    MNN_RVV_ACC_BATCH_S4_PAIR(4);
                    MNN_RVV_ACC_BATCH_S4_PAIR(6);
                    MNN_RVV_ACC_BATCH_S4_PAIR(8);
                    MNN_RVV_ACC_BATCH_S4_PAIR(10);
                    MNN_RVV_ACC_BATCH_S4_PAIR(12);
                    MNN_RVV_ACC_BATCH_S4_PAIR(14);
                } else if (srcDepthQuad == 64) {
                    MNN_RVV_ACC_BATCH_S4_PAIR(0);
                    MNN_RVV_ACC_BATCH_S4_PAIR(2);
                    MNN_RVV_ACC_BATCH_S4_PAIR(4);
                    MNN_RVV_ACC_BATCH_S4_PAIR(6);
                    MNN_RVV_ACC_BATCH_S4_PAIR(8);
                    MNN_RVV_ACC_BATCH_S4_PAIR(10);
                    MNN_RVV_ACC_BATCH_S4_PAIR(12);
                    MNN_RVV_ACC_BATCH_S4_PAIR(14);
                    MNN_RVV_ACC_BATCH_S4_PAIR(16);
                    MNN_RVV_ACC_BATCH_S4_PAIR(18);
                    MNN_RVV_ACC_BATCH_S4_PAIR(20);
                    MNN_RVV_ACC_BATCH_S4_PAIR(22);
                    MNN_RVV_ACC_BATCH_S4_PAIR(24);
                    MNN_RVV_ACC_BATCH_S4_PAIR(26);
                    MNN_RVV_ACC_BATCH_S4_PAIR(28);
                    MNN_RVV_ACC_BATCH_S4_PAIR(30);
                    MNN_RVV_REDUCE_BATCH_S4_PAIR();
                    MNN_RVV_RESET_BATCH_S4_PAIR();
                    MNN_RVV_ACC_BATCH_S4_PAIR(32);
                    MNN_RVV_ACC_BATCH_S4_PAIR(34);
                    MNN_RVV_ACC_BATCH_S4_PAIR(36);
                    MNN_RVV_ACC_BATCH_S4_PAIR(38);
                    MNN_RVV_ACC_BATCH_S4_PAIR(40);
                    MNN_RVV_ACC_BATCH_S4_PAIR(42);
                    MNN_RVV_ACC_BATCH_S4_PAIR(44);
                    MNN_RVV_ACC_BATCH_S4_PAIR(46);
                    MNN_RVV_ACC_BATCH_S4_PAIR(48);
                    MNN_RVV_ACC_BATCH_S4_PAIR(50);
                    MNN_RVV_ACC_BATCH_S4_PAIR(52);
                    MNN_RVV_ACC_BATCH_S4_PAIR(54);
                    MNN_RVV_ACC_BATCH_S4_PAIR(56);
                    MNN_RVV_ACC_BATCH_S4_PAIR(58);
                    MNN_RVV_ACC_BATCH_S4_PAIR(60);
                    MNN_RVV_ACC_BATCH_S4_PAIR(62);
                } else {
                    MNNRVVAccumI8U4_32x4PairLoop(acc, vacc0, vacc1, vacc2, vacc3, zero16, srcBk, srcDepthStride,
                                                 weightDz, weight_step_Y, static_cast<int>(srcDepthQuad), zero32, vl,
                                                 vlPair, vlOne);
                }
                MNN_RVV_REDUCE_BATCH_S4_PAIR();
#undef MNN_RVV_ACC_BATCH_S4_PAIR
#undef MNN_RVV_REDUCE_BATCH_S4_PAIR
#undef MNN_RVV_RESET_BATCH_S4_PAIR

                const float* scale_dz = reinterpret_cast<const float*>(weightDz + srcDepthQuad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
                vfloat32m1_t value = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc, vl4), vl4);
                value = __riscv_vfmul_vv_f32m1(value, __riscv_vle32_v_f32m1(scale_dz, vl4), vl4);
                value = __riscv_vfmul_vf_f32m1(value, post->inputScale[w], vl4);
                const float srcSum = post->srcKernelSum[bk * realCount + w];
                value = __riscv_vfmacc_vf_f32m1(value, srcSum, __riscv_vle32_v_f32m1(weightBias_dz, vl4), vl4);
                partial = __riscv_vfadd_vv_f32m1(partial, value, vl4);
            }
            partial = __riscv_vfadd_vv_f32m1(partial, __riscv_vle32_v_f32m1(bias_dz, vl4), vl4);
            if (needClamp) {
                partial = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(partial, fp32min, vl4), fp32max, vl4);
            }
            __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst_z + w * GEMM_INT8_UNIT * 4), partial, vl4);
        }
    }
    return true;
}

static bool MNNGemmInt8AddBiasScale_16x4_w4_Decode_RVV(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                       size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                                       const QuanPostTreatParameters* post) {
    if (src_depth_quad > 129) {
        return false;
    }
    if ((src_depth_quad == 4 || src_depth_quad == 8 || src_depth_quad == 16 || src_depth_quad == 32 ||
         src_depth_quad == 64 || src_depth_quad == 128) &&
        MNNGemmInt8AddBiasScale_16x4_w4_DecodeS4FastPost_RVV(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad,
                                                             post)) {
        return true;
    }
    float fp32min = 0.f;
    float fp32max = 0.f;
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * src_depth_quad + 4 * 2 * GEMM_INT8_UNIT;
    if (post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }

    const float* biasPtr = post->biasFloat;
    const int blockNum = static_cast<int>(post->blockNum);
    const bool fastPost =
        post->inputScale != nullptr && post->inputBias == nullptr && biasPtr != nullptr && post->fp32minmax != nullptr;
    const float fastInputScale = fastPost ? post->inputScale[0] : 1.0f;
    for (int dz = 0; dz < static_cast<int>(dst_depth_quad); ++dz) {
        float partial[GEMM_INT8_UNIT] = {0.0f};
        auto dst_x = reinterpret_cast<float*>(dst + dz * dst_step);
        for (int bk = 0; bk < blockNum; ++bk) {
            alignas(16) int32_t acc[GEMM_INT8_UNIT];
            const int8_t* srcBk = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT;
            const size_t vl = __riscv_vsetvl_e8m1(GEMM_INT8_SRC_UNIT);
            const vint16m2_t zero16 = __riscv_vmv_v_x_i16m2(0, vl);
            vint16m2_t vacc0 = zero16;
            vint16m2_t vacc1 = zero16;
            vint16m2_t vacc2 = zero16;
            vint16m2_t vacc3 = zero16;
            const auto weightDz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
#define MNN_RVV_ACC1(SZ)                                                                         \
    do {                                                                                         \
        const auto src_z = srcBk + (SZ) * GEMM_INT8_SRC_UNIT;                                    \
        const vint8m1_t vsrc = __riscv_vle8_v_i8m1(src_z, vl);                                   \
        const auto weightSz = reinterpret_cast<const uint8_t*>(weightDz + weight_step_Y * (SZ)); \
        MNNRVVAccumI8U4_16x4(vacc0, vacc1, vacc2, vacc3, vsrc, weightSz, vl);                    \
    } while (false)
            if (src_depth_quad == 4) {
                MNN_RVV_ACC1(0);
                MNN_RVV_ACC1(1);
                MNN_RVV_ACC1(2);
                MNN_RVV_ACC1(3);
            } else if (src_depth_quad == 8) {
                MNN_RVV_ACC1(0);
                MNN_RVV_ACC1(1);
                MNN_RVV_ACC1(2);
                MNN_RVV_ACC1(3);
                MNN_RVV_ACC1(4);
                MNN_RVV_ACC1(5);
                MNN_RVV_ACC1(6);
                MNN_RVV_ACC1(7);
            } else {
                for (int sz = 0; sz < static_cast<int>(src_depth_quad); ++sz) {
                    const auto src_z = srcBk + sz * GEMM_INT8_SRC_UNIT;
                    const vint8m1_t vsrc = __riscv_vle8_v_i8m1(src_z, vl);
                    const auto weightSz = reinterpret_cast<const uint8_t*>(weightDz + weight_step_Y * sz);
                    MNNRVVAccumI8U4_16x4(vacc0, vacc1, vacc2, vacc3, vsrc, weightSz, vl);
                }
            }
#undef MNN_RVV_ACC1
            const vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);
            const size_t vlOne = __riscv_vsetvl_e32m1(1);
            __riscv_vse32_v_i32m1(acc + 0, __riscv_vwredsum_vs_i16m2_i32m1(vacc0, zero32, vl), vlOne);
            __riscv_vse32_v_i32m1(acc + 1, __riscv_vwredsum_vs_i16m2_i32m1(vacc1, zero32, vl), vlOne);
            __riscv_vse32_v_i32m1(acc + 2, __riscv_vwredsum_vs_i16m2_i32m1(vacc2, zero32, vl), vlOne);
            __riscv_vse32_v_i32m1(acc + 3, __riscv_vwredsum_vs_i16m2_i32m1(vacc3, zero32, vl), vlOne);

            const float srcSum = post->srcKernelSum[bk];
            const float* scale_dz = reinterpret_cast<const float*>(weightDz + src_depth_quad * weight_step_Y);
            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const size_t vl4 = __riscv_vsetvl_e32m1(GEMM_INT8_UNIT);
            vfloat32m1_t value = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(acc, vl4), vl4);
            value = __riscv_vfmul_vv_f32m1(value, __riscv_vle32_v_f32m1(scale_dz, vl4), vl4);
            if (fastPost) {
                value = __riscv_vfmul_vf_f32m1(value, fastInputScale, vl4);
                value = __riscv_vfmacc_vf_f32m1(value, srcSum, __riscv_vle32_v_f32m1(weightBias_dz, vl4), vl4);
                if (bk > 0) {
                    value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(partial, vl4), vl4);
                }
                if (bk == blockNum - 1) {
                    value =
                        __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
                    value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
                    __riscv_vse32_v_f32m1(dst_x, value, vl4);
                } else {
                    __riscv_vse32_v_f32m1(partial, value, vl4);
                }
                continue;
            }
            const bool hasInputScale = post->inputScale != nullptr;
            const float inputScale =
                hasInputScale ? (post->inputBias ? post->inputScale[bk] : post->inputScale[0]) : 1.0f;
            const float inputBias = post->inputBias ? post->inputBias[bk] : 0.0f;
            if (hasInputScale) {
                value = __riscv_vfmul_vf_f32m1(value, inputScale, vl4);
            }
            value = __riscv_vfmacc_vf_f32m1(value, srcSum, __riscv_vle32_v_f32m1(weightBias_dz, vl4), vl4);
            if (post->inputBias) {
                const auto weightKernelSum =
                    post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;
                value = __riscv_vfmacc_vf_f32m1(value, inputBias, __riscv_vle32_v_f32m1(weightKernelSum, vl4), vl4);
            }
            if (bk > 0) {
                value = __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(partial, vl4), vl4);
            }
            if (bk == blockNum - 1) {
                if (biasPtr) {
                    value =
                        __riscv_vfadd_vv_f32m1(value, __riscv_vle32_v_f32m1(biasPtr + dz * GEMM_INT8_UNIT, vl4), vl4);
                }
                if (post->fp32minmax) {
                    value = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(value, fp32min, vl4), fp32max, vl4);
                }
                __riscv_vse32_v_f32m1(dst_x, value, vl4);
            } else {
                __riscv_vse32_v_f32m1(partial, value, vl4);
            }
        }
    }
    return true;
}

void MNNSpacemitIme2GemmInt8AddBiasScaleW4(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                           size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post,
                                           size_t realCount) {
    const int bytes = 4;
    float fp32min = 0.f;
    float fp32max = 0.f;
    const int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) / 2;
    const int weight_step_Z = weight_step_Y * src_depth_quad + 4 * 2 * GEMM_INT8_UNIT;
    MNN_ASSERT(post->useInt8 == 0);
    const bool decodeBiasPost = MNNSpacemitIme2UseDecodeBiasPost(post);
    if (MNNSpacemitIme2Enabled() && MNNGemmInt8AddBiasScale_16x4_w4_Unit_IME2(
                                        dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realCount)) {
        return;
    }
    if (!decodeBiasPost && realCount == 1 &&
        MNNGemmInt8AddBiasScale_16x4_w4_Decode_RVV(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post)) {
        return;
    }
    if (!decodeBiasPost &&
        (src_depth_quad == 2 || src_depth_quad == 4 || src_depth_quad == 8 || src_depth_quad == 16 ||
         src_depth_quad == 32 || src_depth_quad == 64 || src_depth_quad == 128) &&
        MNNGemmInt8AddBiasScale_16x4_w4_BatchS4FastPost_RVV(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad,
                                                            post, realCount)) {
        return;
    }
    if (post->fp32minmax) {
        fp32min = post->fp32minmax[0];
        fp32max = post->fp32minmax[1];
    }
    const float* biasPtr = post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;

    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z = dst + dz * dst_step;
        auto accum_z = accumbuff;
        for (int bk = 0; bk < blockNum; ++bk) {
            const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk * weight_step_Z;
            const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz = biasPtr + dz * GEMM_INT8_UNIT;
            const auto srcSumPtr = post->srcKernelSum + bk * realCount;
            const auto inputScalePtr = post->inputBias ? post->inputScale + bk * realCount : post->inputScale;
            const auto weightKernelSum = post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;

            for (int w = 0; w < realCount; ++w) {
                const auto src_x = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount + w * GEMM_INT8_SRC_UNIT;
                auto dst_x = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x = accum_z + w * GEMM_INT8_UNIT;
                int32_t acc[4] = {0, 0, 0, 0};

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = reinterpret_cast<const uint8_t*>(weight_dz + weight_step_Y * sz);
                    const auto src_z = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;
                    MNNRVVDotI8U4_16x4(acc, src_z, weight_sz);
                }

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    float value = acc[j] * scale_dz[j] + srcSumPtr[w] * weightBias_dz[j];
                    if (post->inputScale) {
                        value = acc[j] * scale_dz[j] * inputScalePtr[w] + srcSumPtr[w] * weightBias_dz[j];
                    }
                    if (post->inputBias) {
                        value += post->inputBias[bk * realCount + w] * weightKernelSum[j];
                    }
                    if (bk > 0) {
                        value += reinterpret_cast<float*>(accum_x)[j];
                    }
                    if (bk == blockNum - 1) {
                        if (decodeBiasPost) {
                            value +=
                                std::fma(post->scale[0], post->weightKernelSum[dz * GEMM_INT8_UNIT + j], bias_dz[j]);
                        } else if (biasPtr) {
                            value += bias_dz[j];
                        }
                        if (post->fp32minmax) {
                            value = std::min(std::max(fp32min, value), fp32max);
                        }
                        reinterpret_cast<float*>(dst_x)[j] = value;
                    } else {
                        reinterpret_cast<float*>(accum_x)[j] = value;
                    }
                }
            }
        }
    }
}
