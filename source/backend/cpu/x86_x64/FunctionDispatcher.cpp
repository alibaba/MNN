//
//  FunctionDispatcher.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include "avx512/FunctionSummary.hpp"
#include "avx/FunctionSummary.hpp"
#include "AVX2Functions.hpp"
#include "avxfma/FunctionSummary.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "cpu_id.h"
#include "sse/FunctionSummary.hpp"
// https://stackoverflow.com/a/11230437
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

struct FunctionGroup {
    int tileNumber                                                                               = 8;
    int eP                                                                                       = 12;
    int lP                                                                                       = 1;
    int hP                                                                                       = 4;
    void (*MNNExpC8)(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) = _SSE_MNNExpC8;
    void (*MNNSoftmax)(float* dest, const float* source, size_t size) = _SSE_MNNSoftmax;
    void (*MNNReluInt8)(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) = _SSE_MNNReluInt8;
    void (*MNNHardSwish)(float* dst, const float* src, size_t size) = _SSE_MNNHardSwish;
    void (*MNNGelu)(float* dst, const float* src, size_t size, float* parameters) = _SSE_MNNGelu;
    void (*MNNNorm)(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) = _SSE_MNNNorm;
};

static FunctionGroup gFunc;

void _SSEMNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = gFunc.eP;
    *lP = gFunc.lP;
    *hP = gFunc.hP;
}
void MNNFunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    auto coreFunction = MNN::MNNGetCoreFunctions();
    if (cpuFlags & libyuv::kCpuHasSSSE3) {
        coreFunction->MNNGetMatMulPackMode = _SSEMNNGetMatMulPackMode;
        coreFunction->MNNPackedMatMul       = _SSE_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _SSE_MNNPackedMatMulRemain;
#ifdef MNN_LOW_MEMORY
        coreFunction->MNNPackedMatMul_int4       = _SSE_MNNPackedMatMul_int4;
        coreFunction->MNNPackedMatMulRemain_int4 = _SSE_MNNPackedMatMulRemain_int4;
        coreFunction->MNNPackedMatMul_int8       = _SSE_MNNPackedMatMul_int8;
        coreFunction->MNNPackedMatMulRemain_int8 = _SSE_MNNPackedMatMulRemain_int8;
        coreFunction->MNNAbsMax = _SSE_MNNAbsMaxFP32;
#endif
        coreFunction->MNNPackC4ForMatMul_A  = _SSE_MNNPackC4ForMatMul_A;
        coreFunction->MNNPackForMatMul_B    = _SSE_MNNPackForMatMul_B;
        // Dynamic Quant
        coreFunction->MNNCountMaxMinValue = _SSE_MNNComputeScaleZeroScalar;
    }
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        MNN::AVX2Functions::init(cpuFlags);
        gFunc.MNNExpC8 = _AVX_MNNExpC8;
        gFunc.MNNSoftmax = _AVX_MNNSoftmax;
        gFunc.MNNGelu = _AVX_MNNGelu;
        if (cpuFlags & libyuv::kCpuHasFMA3) {
            gFunc.MNNGelu = _AVX_MNNGeluFMA;
            gFunc.MNNExpC8 = _AVX_MNNExpC8FMA;
        }
        gFunc.MNNNorm = _AVX_MNNNorm;
    }
    _SSE_ImageProcessInit(coreFunction, cpuFlags);
}

void MNNAvgPoolUint8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx, ssize_t paddingx, ssize_t factor) {
    int pack = 16;
    uint32_t f = static_cast<uint32_t>(factor);
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
    const uint8_t* srcPtr = reinterpret_cast<uint8_t*>(src);
    for (int ox = 0; ox < outputWidth; ++ox) {
        std::vector<uint32_t> sum_(pack, 0);
        for (int y = 0; y < kernely; ++y) {
            for (int x = 0; x < kernelx; ++x) {
                const uint8_t *inputPtr = srcPtr + pack* (inputWidth* y + x);
                for (int idx = 0; idx < pack; ++idx) {
                    sum_[idx] += *(inputPtr + idx);
                }
            }
        }
        for (int idx = 0; idx < pack; ++idx) {
            *(dstPtr + idx) = static_cast<uint8_t>((sum_[idx] * f)>>24);
        }
        dstPtr = dstPtr + pack;
        srcPtr = srcPtr + pack* stridesx;
    }
}

void MNNMaxPoolInt8_(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx) {
    int pack = 16;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    for (int ox = 0; ox < outputWidth; ++ox){
        std::vector<int8_t> results(pack, INT8_MIN);
        for (int y = 0; y < kernely; ++y) {
            for (int x = 0; x < kernelx; ++x) {
                const int8_t* inputPtr = srcPtr + pack* (x + inputWidth* y);
                for (int idx = 0; idx < pack; ++idx) {   
                    results[idx] = std::max(results[idx], *(inputPtr + idx));
                }
            }
        }

        for (int idx = 0; idx < pack;++idx) {
            *(dstPtr + idx) = results[idx];
        }
        dstPtr = dstPtr + pack;
        srcPtr = srcPtr + pack* stridesx;
    }
}

void MNNInt8FunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    auto core = MNN::MNNGetInt8CoreFunctions();
    core->MNNAvgPoolInt8 = MNNAvgPoolUint8;
    core->MNNMaxPoolInt8 = MNNMaxPoolInt8_;
    core->MNNReluWithSlopeChannelInt8 = _SSE_MNNReluWithSlopeChannelInt8;
    if (cpuFlags & libyuv::kCpuHasSSE41) {
        core->MNNFloat2Int8 = _SSE_MNNFloat2Int8;
        core->MNNInt8ScaleToFloat = _SSE_MNNInt8ScaleToFloat;
        core->Int8GemmKernel = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->Int8GemmKernelFast = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->ConvDepthwiseLineInt8 = _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit;
#ifdef MNN_LOW_MEMORY
        core->Int8GemmKernel_W4 = _SSE_MNNGemmInt8AddBiasScale_16x4_w4;
#endif
    }
}


void _SSE_ImageProcessInit(void* functions, int cpuFlags) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNRGBAToBGRA = _SSE_MNNRGBAToBGRA;
    coreFunction->MNNNV21ToRGBA = _SSE_MNNNV21ToRGBA;
    coreFunction->MNNNV21ToRGB = _SSE_MNNNV21ToRGB;
    coreFunction->MNNNV21ToBGRA = _SSE_MNNNV21ToBGRA;
    coreFunction->MNNNV21ToBGR = _SSE_MNNNV21ToBGR;
    //coreFunction->MNNsampleBilinearCommon = _SSE_sampleBilinearCommon;
    if (cpuFlags & libyuv::kCpuHasSSE41) {
        coreFunction->MNNC1ToFloatC1 = _SSE_MNNC1ToFloatC1;
        coreFunction->MNNC3ToFloatC3 = _SSE_MNNC3ToFloatC3;
        coreFunction->MNNC3ToFloatRGBA = _SSE_MNNC3ToFloatRGBA;
        coreFunction->MNNSamplerC4Nearest = _SSE_MNNSamplerC4Nearest;
        coreFunction->MNNSamplerC4Bilinear = _SSE_MNNSampleC4Bilinear;
    }
}

// ========= CommonOptFunction.cpp ===========

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNCopyC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNAddC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return _SSE_MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) {
    return gFunc.MNNReluInt8(dst, src, size, zeroPoint);
}

void MNNHardSwish(float* dst, const float* src, size_t size) {
    return gFunc.MNNHardSwish(dst, src, size);
}

void MNNGelu(float* dst, const float* src, size_t size, float* parameters) {
    return gFunc.MNNGelu(dst, src, size, parameters);
}

void MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    gFunc.MNNExpC8(dest, source, offset, parameters, countC8);
}

void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    _SSE_MNNInt8ToInt16(dest, source, count);
}

void MNNSoftmax(float* dest, const float* source, size_t size) {
    gFunc.MNNSoftmax(dest, source, size);
}

void MNNNorm(float* dest, const float* source, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    gFunc.MNNNorm(dest, source, gamma, beta, epsilon, size, RMSNorm);
}
