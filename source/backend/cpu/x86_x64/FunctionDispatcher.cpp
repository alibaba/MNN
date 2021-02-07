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
    void (*MNNAddBias)(float* dst, const float* bias, size_t planeNumber, size_t biasNumber)     = _SSE_MNNAddBias;
    void (*MNNAddBiasRelu)(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) = _SSE_MNNAddBiasRelu;
    void (*MNNAddBiasRelu6)(float* dst, const float* bias, size_t planeNumber,
                            size_t biasNumber)                                                   = _SSE_MNNAddBiasRelu6;

    void (*MNNMatrixAdd)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                         size_t bStride, size_t height) = _SSE_MNNMatrixAdd;
    void (*MNNMatrixSub)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                         size_t bStride, size_t height) = _SSE_MNNMatrixSub;

    void (*MNNGemmFloatUnit_4)(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad,
                               size_t dst_step, size_t dst_depth_quad,
                               size_t weight_depth_offset)                  = _SSE_MNNGemmFloatUnit_4;
    void (*MNNGemmFloatCommon_4)(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                                 size_t dst_step, size_t dst_depth_quad, size_t width,
                                 size_t weight_depth_offset)                = _SSE_MNNGemmFloatCommon_4;
    void (*MNNPackC4ForMatMul_A)(float* dest, const float* source, size_t e, size_t l,
                                 size_t eReal)                              = _SSE_MNNPackC4ForMatMul_A;
    void (*MNNPackForMatMul_B)(float* dest, const float* source, size_t h, size_t l, bool transpose) = _SSE_MNNPackForMatMul_B;
    void (*MNNPackedMatMul)(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                            const float* postParameters, const float* bias) = _SSE_MNNPackedMatMul;
    void (*MNNPackedMatMulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                  float* cache, const float* postParameters,
                                  const float* bias)                        = _SSE_MNNPackedMatMulRemain;
    void (*MNNConvRunForLineDepthwise)(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                       size_t srcHStep, size_t dstHStep) = _SSE_MNNConvRunForLineDepthwise;
    void (*MNNGemmInt8AddBiasScale_16x4_Unit)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
    void (*MNNGemmInt8AddBiasScale_16x4_Unit_FAST)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
    void (*MNNExpC8)(float* dest, const float* source, const float* parameters, size_t countC8) = _SSE_MNNExpC8;
    void (*MNNFloat2Int8)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                       ssize_t maxValue, ssize_t zeroPoint) = _SSE_MNNFloat2Int8;
    void (*MNNInt8ScaleToFloat)(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) = _SSE_MNNInt8ScaleToFloat;
    void (*MNNLineDepthWiseInt8AddBiasScaleUnit)(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) = _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit;
    void (*MNNComputeMatMulForE_1)(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) = _SSE_MNNComputeMatMulForE_1;
};

static FunctionGroup gFunc;
void MNNFunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        gFunc.MNNAddBias            = _AVX_MNNAddBias;
        gFunc.MNNAddBiasRelu        = _AVX_MNNAddBiasRelu;
        gFunc.MNNAddBiasRelu6       = _AVX_MNNAddBiasRelu6;
        gFunc.MNNMatrixAdd          = _AVX_MNNMatrixAdd;
        gFunc.MNNMatrixSub          = _AVX_MNNMatrixSub;
        gFunc.MNNGemmFloatUnit_4    = _AVX_MNNGemmFloatUnit_4;
        gFunc.MNNGemmFloatCommon_4  = _AVX_MNNGemmFloatCommon_4;
        gFunc.MNNPackedMatMul       = _AVX_MNNPackedMatMul;
        gFunc.MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemain;
        gFunc.eP                    = 24;
        gFunc.MNNPackC4ForMatMul_A  = _AVX_MNNPackC4ForMatMul_A;
        gFunc.MNNConvRunForLineDepthwise = _AVX_MNNConvRunForLineDepthwise;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit;
        gFunc.MNNExpC8 = _AVX_MNNExpC8;
        gFunc.MNNFloat2Int8 = _AVX_MNNFloat2Int8;
        gFunc.MNNInt8ScaleToFloat = _AVX_MNNInt8ScaleToFloat;
        gFunc.MNNLineDepthWiseInt8AddBiasScaleUnit = _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit;
        gFunc.MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit_FAST = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast;
        if (cpuFlags & libyuv::kCpuHasFMA3) {
            gFunc.MNNGemmFloatUnit_4    = _AVX_MNNGemmFloatUnitFMA_4;
            gFunc.MNNGemmFloatCommon_4  = _AVX_MNNGemmFloatCommonFMA_4;
            gFunc.MNNPackedMatMul       = _AVX_MNNPackedMatMulFMA;
            gFunc.MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemainFMA;
            gFunc.MNNComputeMatMulForE_1 = _AVX_MNNComputeMatMulForE_1FMA;
        }
    }
#ifdef MNN_AVX512
    if (cpuFlags & libyuv::kCpuHasAVX512VNNI) {
//        gFunc.MNNPackForMatMul_B    = _AVX512_MNNPackForMatMul_B;
//        gFunc.MNNPackC4ForMatMul_A  = _AVX512_MNNPackC4ForMatMul_A;
//        gFunc.MNNPackedMatMul = _AVX512_MNNPackedMatMul;
//        gFunc.MNNPackedMatMulRemain = _AVX512_MNNPackedMatMulRemain;
//        gFunc.eP                    = 48;
//        gFunc.hP                    = 8;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
        gFunc.MNNGemmInt8AddBiasScale_16x4_Unit_FAST = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit;
    }
#endif
}

// ========= CommonOptFunction.cpp ===========
void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    return gFunc.MNNAddBias(dst, bias, planeNumber, biasNumber);
}

void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    return gFunc.MNNAddBiasRelu(dst, bias, planeNumber, biasNumber);
}

void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    return gFunc.MNNAddBiasRelu6(dst, bias, planeNumber, biasNumber);
}

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNCopyC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNAddC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset) {
    gFunc.MNNGemmFloatUnit_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, weight_depth_offset);
}

// ========= MNNGemmFloatCommon_4.cpp ===========
void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    gFunc.MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
}

// ========= MNNMatrixAdd.cpp ===========
void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    gFunc.MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
}

// ========= MNNMatrixSub.cpp ===========
void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    gFunc.MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return _SSE_MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
    return gFunc.MNNPackC4ForMatMul_A(dest, source, e, l, eReal);
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    gFunc.MNNPackForMatMul_B(dest, source, h, l, transpose);
}

void MNNGetMatMulPackMode(int* eP, int* lP, int* hP) {
    *eP = gFunc.eP;
    *lP = gFunc.lP;
    *hP = gFunc.hP;
}

int MNNGetConvolutionTileNumber() {
    return gFunc.tileNumber;
}
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint) {
    return gFunc.MNNFloat2Int8(src, dst, sizeQuad, scalep, minValue, maxValue, zeroPoint);
}
void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) {
    return gFunc.MNNInt8ScaleToFloat(dst, src, scale, size, zeroPoint);
}

void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                     const float* postParameters, const float* bias) {
    return gFunc.MNNPackedMatMul(C, A, B, parameter, cache, postParameters, bias);
}
void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                           float* cache, const float* postParameters, const float* bias) {
    return gFunc.MNNPackedMatMulRemain(C, A, B, eSize, parameter, cache, postParameters, bias);
}
void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    gFunc.MNNExpC8(dest, source, parameters, countC8);
}
void MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                size_t srcHStep, size_t dstHStep) {
    return gFunc.MNNConvRunForLineDepthwise(dst, src, weight, width, src_w_setup, fw, fh, dilateX_step, dilateY_step, height, srcHStep, dstHStep);
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
