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
    void (*MNNGemmInt8AddBiasScale_16x4_Unit)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
    void (*MNNGemmInt8AddBiasScale_16x4_Unit_FAST)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
    void (*MNNExpC8)(float* dest, const float* source, const float* parameters, size_t countC8) = _SSE_MNNExpC8;
    void (*MNNFloat2Int8)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                       ssize_t maxValue, ssize_t zeroPoint) = _SSE_MNNFloat2Int8;
    void (*MNNInt8ScaleToFloat)(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) = _SSE_MNNInt8ScaleToFloat;
    void (*MNNLineDepthWiseInt8AddBiasScaleUnit)(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) = _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit;
    void (*MNNComputeMatMulForE_1)(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) = _SSE_MNNComputeMatMulForE_1;
    void (*MNNReluWithSlopeChannel)(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) = _SSE_MNNReluWithSlopeChannel;
    void (*MNNReluInt8)(int8_t* dst, const int8_t* src, size_t size) = _SSE_MNNReluInt8;
    void (*MNNHardSwish)(float* dst, const float* src, size_t size) = _SSE_MNNHardSwish;
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
    }
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        gFunc.eP                    = 24;
        gFunc.lP                    = 1;
        gFunc.hP                    = 4;

        coreFunction->MNNMatrixAdd          = _AVX_MNNMatrixAdd;
        coreFunction->MNNMatrixSub          = _AVX_MNNMatrixSub;
        coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemain;
        coreFunction->MNNPackC4ForMatMul_A  = _AVX_MNNPackC4ForMatMul_A;
        coreFunction->MNNConvRunForLineDepthwise = _AVX_MNNConvRunForLineDepthwise;
        coreFunction->MNNAxByClampBroadcastUnit = _AVX_MNNAxByClampBroadcastUnit;

        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit;
        gFunc.MNNExpC8 = _AVX_MNNExpC8;
        gFunc.MNNFloat2Int8 = _AVX_MNNFloat2Int8;
        gFunc.MNNInt8ScaleToFloat = _AVX_MNNInt8ScaleToFloat;
        gFunc.MNNLineDepthWiseInt8AddBiasScaleUnit = _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit;
        gFunc.MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit_FAST = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast;
        gFunc.MNNReluWithSlopeChannel = _AVX_MNNReluWithSlopeChannel;
        if (cpuFlags & libyuv::kCpuHasFMA3) {
            coreFunction->MNNPackedMatMul       = _AVX_MNNPackedMatMulFMA;
            coreFunction->MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemainFMA;
            gFunc.MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1FMA;
        }
    }
#ifdef MNN_AVX512
    if (cpuFlags & libyuv::kCpuHasAVX512VNNI) {
        coreFunction->MNNPackForMatMul_B    = _AVX512_MNNPackForMatMul_B;
        coreFunction->MNNPackC4ForMatMul_A  = _AVX512_MNNPackC4ForMatMul_A;
        coreFunction->MNNPackedMatMul = _AVX512_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _AVX512_MNNPackedMatMulRemain;
        gFunc.eP                    = 24;
        gFunc.hP                    = 4;
        gFunc.lP                    = 4;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit_FAST = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
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
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    return gFunc.MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realDst);
}

void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    gFunc.MNNLineDepthWiseInt8AddBiasScaleUnit(dst, src, weight, parameters, width, src_w_step, fw, fh, dilateX_step, dilateY_step);
}
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    _SSE_MNNInt8ToInt16(dest, source, count);
}

void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    gFunc.MNNComputeMatMulForE_1(A, B, C, biasPtr, param, tId);
}

void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    gFunc.MNNGemmInt8AddBiasScale_16x4_Unit_FAST(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realCount);
}
