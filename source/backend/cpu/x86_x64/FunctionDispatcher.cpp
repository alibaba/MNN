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

bool MNNReorder4x4ByPlatform(float* dst, size_t number) {
    return _SSE_MNNReorder4x4ByPlatform(dst, number);
}

struct FunctionGroup {
    int tileNumber                                                                               = 8;
    int eP                                                                                       = 12;
    int lP                                                                                       = 1;
    int hP                                                                                       = 4;
    void (*MNNExpC8)(float* dest, const float* source, const float* parameters, size_t countC8) = _SSE_MNNExpC8;
    void (*MNNSoftmax)(float* dest, const float* source, size_t size) = _SSE_MNNSoftmax;
    void (*MNNFloat2Int8)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                       ssize_t maxValue, ssize_t zeroPoint) = _SSE_MNNFloat2Int8;
    void (*MNNInt8ScaleToFloat)(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) = _SSE_MNNInt8ScaleToFloat;
    void (*MNNComputeMatMulForE_1)(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) = _SSE_MNNComputeMatMulForE_1;
    void (*MNNReluWithSlopeChannel)(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) = _SSE_MNNReluWithSlopeChannel;
    void (*MNNReluInt8)(int8_t* dst, const int8_t* src, size_t size) = _SSE_MNNReluInt8;
    void (*MNNHardSwish)(float* dst, const float* src, size_t size) = _SSE_MNNHardSwish;
    void (*MNNGelu)(float* dst, const float* src, size_t size) = _SSE_MNNGelu;
    void (*MNNNorm)(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size) = _SSE_MNNNorm;
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
        coreFunction->MNNMatrixAdd          = _SSE_MNNMatrixAdd;
        coreFunction->MNNMatrixSub          = _SSE_MNNMatrixSub;
        coreFunction->MNNPackedMatMul       = _SSE_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _SSE_MNNPackedMatMulRemain;
        coreFunction->MNNPackC4ForMatMul_A  = _SSE_MNNPackC4ForMatMul_A;
        coreFunction->MNNPackForMatMul_B    = _SSE_MNNPackForMatMul_B;
        coreFunction->MNNConvRunForLineDepthwise = _SSE_MNNConvRunForLineDepthwise;
        coreFunction->MNNAxByClampBroadcastUnit = _SSE_MNNAxByClampBroadcastUnit;
        coreFunction->MNNComputeMatMulForE_1 = _SSE_MNNComputeMatMulForE_1;
    }
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        MNN::AVX2Functions::init(cpuFlags);
        gFunc.MNNExpC8 = _AVX_MNNExpC8;
        gFunc.MNNSoftmax = _AVX_MNNSoftmax;
        gFunc.MNNGelu = _AVX_MNNGelu;
        gFunc.MNNNorm = _AVX_MNNNorm;
        gFunc.MNNFloat2Int8 = _AVX_MNNFloat2Int8;
        gFunc.MNNInt8ScaleToFloat = _AVX_MNNInt8ScaleToFloat;
    }
}

void MNNInt8FunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    auto core = MNN::MNNGetInt8CoreFunctions();
    if (cpuFlags & libyuv::kCpuHasSSSE3) {
        core->Int8GemmKernel = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->Int8GemmKernelFast = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->ConvDepthwiseLineInt8 = _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit;
    }
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        core->Int8GemmKernel = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->Int8GemmKernelFast = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast;
        core->ConvDepthwiseLineInt8 = _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit;
    }
#ifdef MNN_AVX512_VNNI
    if (cpuFlags & libyuv::kCpuHasAVX512VNNI) {
        core->Int8GemmKernel = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
        core->Int8GemmKernelFast = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
    }
#endif
}

// ========= CommonOptFunction.cpp ===========

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNCopyC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNAddC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return gFunc.MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size) {
    return gFunc.MNNReluInt8(dst, src, size);
}

void MNNHardSwish(float* dst, const float* src, size_t size) {
    return gFunc.MNNHardSwish(dst, src, size);
}

void MNNGelu(float* dst, const float* src, size_t size) {
    return gFunc.MNNGelu(dst, src, size);
}

void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint) {
    return gFunc.MNNFloat2Int8(src, dst, sizeQuad, scalep, minValue, maxValue, zeroPoint);
}
void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) {
    return gFunc.MNNInt8ScaleToFloat(dst, src, scale, size, zeroPoint);
}

void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    gFunc.MNNExpC8(dest, source, parameters, countC8);
}

void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    _SSE_MNNInt8ToInt16(dest, source, count);
}

void MNNSoftmax(float* dest, const float* source, size_t size) {
    gFunc.MNNSoftmax(dest, source, size);
}

void MNNNorm(float* dest, const float* source, const float *gamma, const float *beta, float epsilon, size_t size) {
    gFunc.MNNNorm(dest, source, gamma, beta, epsilon, size);
}
