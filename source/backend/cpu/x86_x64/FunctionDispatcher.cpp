//
//  FunctionDispatcher.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
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
    for (int i = 0; i < number; ++i) {
        auto addr = dst + 16 * i;
        auto s0   = _mm_loadu_ps(addr + 4 * 0);
        auto s1   = _mm_loadu_ps(addr + 4 * 1);
        auto s2   = _mm_loadu_ps(addr + 4 * 2);
        auto s3   = _mm_loadu_ps(addr + 4 * 3);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

        _mm_storeu_ps(addr + 4 * 0, s0);
        _mm_storeu_ps(addr + 4 * 1, s1);
        _mm_storeu_ps(addr + 4 * 2, s2);
        _mm_storeu_ps(addr + 4 * 3, s3);
    }
    return true;
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
    void (*MNNPackedMatMul)(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                            const float* postParameters, const float* bias) = _SSE_MNNPackedMatMul;
    void (*MNNPackedMatMulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                  float* cache, const float* postParameters,
                                  const float* bias)                        = _SSE_MNNPackedMatMulRemain;
    void (*MNNConvRunForLineDepthwise)(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                       size_t srcHStep, size_t dstHStep) = _SSE_MNNConvRunForLineDepthwise;
    void (*MNNGemmInt8AddBiasScale_16x4_Unit)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post) = _SSE_MNNGemmInt8AddBiasScale_16x4_Unit;
    void (*MNNExpC8)(float* dest, const float* source, const float* parameters, size_t countC8) = _SSE_MNNExpC8;
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
        if (cpuFlags & libyuv::kCpuHasFMA3) {
            gFunc.MNNGemmFloatUnit_4    = _AVX_MNNGemmFloatUnitFMA_4;
            gFunc.MNNGemmFloatCommon_4  = _AVX_MNNGemmFloatCommonFMA_4;
            gFunc.MNNPackedMatMul       = _AVX_MNNPackedMatMulFMA;
            gFunc.MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemainFMA;
        }
    }
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

#include <algorithm>
#include <cmath>

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return _SSE_MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * area * 4;
        auto srcPlane = src + z * area * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 4 * x;
            auto d  = dstPlane + 16 * x;
            auto s0 = _mm_loadu_ps(s + 0 * area);
            auto s1 = _mm_loadu_ps(s + 1 * area);
            auto s2 = _mm_loadu_ps(s + 2 * area);
            auto s3 = _mm_loadu_ps(s + 3 * area);

            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(d + 4 * 0, s0);
            _mm_storeu_ps(d + 4 * 1, s1);
            _mm_storeu_ps(d + 4 * 2, s2);
            _mm_storeu_ps(d + 4 * 3, s3);
        }
    }
    auto areaRemain  = areaC4 * 4;
    auto depthRemain = depthC4 * 4;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * area * 4 + dst;
        const float* srcPlane = src + depthC4 * area * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[4 * x + y] = srcPlane[y * area + x];
            }
            for (int y = remain; y < 4; y++) {
                dstPlane[4 * x + y] = 0;
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        float* dstPlane       = z * area * 4 + dst;
        const float* srcPlane = src + z * area * 4;
        for (int x = areaRemain; x < area; ++x) {
            float s0 = srcPlane[x];
            float s1 = srcPlane[x + area];
            float s2 = srcPlane[x + area * 2];
            float s3 = srcPlane[x + area * 3];
            _mm_store_ps(dstPlane + 4 * x, _mm_set_ps(s3, s2, s1, s0));
        }
    }
}
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w         = dim[0];
    int h         = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    auto wC4      = w / 4;
    auto hC4      = h / 4;
    for (int y = 0; y < hC4; ++y) {
        auto sy = (float*)srcO + 4 * y;
        auto dy = (float*)dstO + 4 * y * dstStride;
        for (int x = 0; x < wC4; ++x) {
            auto sx = sy + x * 4 * srcStride;
            auto dx = dy + 4 * x;
            auto s0 = _mm_loadu_ps(sx + srcStride * 0);
            auto s1 = _mm_loadu_ps(sx + srcStride * 1);
            auto s2 = _mm_loadu_ps(sx + srcStride * 2);
            auto s3 = _mm_loadu_ps(sx + srcStride * 3);
            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(dx + dstStride * 0, s0);
            _mm_storeu_ps(dx + dstStride * 1, s1);
            _mm_storeu_ps(dx + dstStride * 2, s2);
            _mm_storeu_ps(dx + dstStride * 3, s3);
        }
    }
    // Down
    for (int i = hC4 * 4; i < h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = 0; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj     = *sj;
        }
    }
    // Right
    for (int i = 0; i < hC4 * 4; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = wC4 * 4; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj     = *sj;
        }
    }
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * area * 4;
        auto srcPlane = src + z * area * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 16 * x;
            auto d  = dstPlane + 4 * x;
            auto s0 = _mm_loadu_ps(s + 0 * 4);
            auto s1 = _mm_loadu_ps(s + 1 * 4);
            auto s2 = _mm_loadu_ps(s + 2 * 4);
            auto s3 = _mm_loadu_ps(s + 3 * 4);

            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(d + 0 * area, s0);
            _mm_storeu_ps(d + 1 * area, s1);
            _mm_storeu_ps(d + 2 * area, s2);
            _mm_storeu_ps(d + 3 * area, s3);
        }
    }
    auto areaRemain  = areaC4 * 4;
    auto depthRemain = depthC4 * 4;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * area * 4 + dst;
        const float* srcPlane = src + depthC4 * area * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[y * area + x] = srcPlane[4 * x + y];
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        const float* srcPlane = z * area * 4 + src;
        float* dstPlane       = dst + z * area * 4;
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < 4; y++) {
                dstPlane[y * area + x] = srcPlane[4 * x + y];
            }
        }
    }
}
void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
    return gFunc.MNNPackC4ForMatMul_A(dest, source, e, l, eReal);
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    if (!transpose) {
        MNNUnpackTranspose(dest, source, l, h);
        return;
    }
    MNNPackC4(dest, source, l, h);
}

void MNNGetMatMulPackMode(int* eP, int* lP, int* hP) {
    *eP = gFunc.eP;
    *lP = gFunc.lP;
    *hP = gFunc.hP;
}

int MNNGetConvolutionTileNumber() {
    return gFunc.tileNumber;
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
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post) {
    return gFunc.MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post);
}
