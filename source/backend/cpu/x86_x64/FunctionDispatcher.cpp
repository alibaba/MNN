//
//  FunctionDispatcher.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include <limits>
#include "sse/FunctionSummary.hpp"
#include "avx/FunctionSummary.hpp"
#include "cpu_id.h"
// https://stackoverflow.com/a/11230437
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


bool MNNReorder4x4ByPlatform(float* dst, size_t number) {
    for (int i=0; i<number; ++i) {
        auto addr = dst + 16 * i;
        auto s0 = _mm_loadu_ps(addr + 4 * 0);
        auto s1 = _mm_loadu_ps(addr + 4 * 1);
        auto s2 = _mm_loadu_ps(addr + 4 * 2);
        auto s3 = _mm_loadu_ps(addr + 4 * 3);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        
        _mm_storeu_ps(addr + 4 * 0, s0);
        _mm_storeu_ps(addr + 4 * 1, s1);
        _mm_storeu_ps(addr + 4 * 2, s2);
        _mm_storeu_ps(addr + 4 * 3, s3);
    }
    return true;
}


struct FunctionGroup {
    int tileNumber = 8;
    void(*MNNAddBias)(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) = _SSE_MNNAddBias;
    void(*MNNAddBiasRelu)(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) = _SSE_MNNAddBiasRelu;
    void(*MNNAddBiasRelu6)(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) = _SSE_MNNAddBiasRelu6;
    
    void(*MNNMatrixAdd)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height) = _SSE_MNNMatrixAdd;
    void(*MNNMatrixSub)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height) = _SSE_MNNMatrixSub;

    void (*MNNGemmFloatUnit_4)(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                            size_t dst_depth_quad, size_t weight_depth_offset) = _SSE_MNNGemmFloatUnit_4;
    void (*MNNGemmFloatCommon_4)(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                              size_t dst_depth_quad, size_t width, size_t weight_depth_offset) = _SSE_MNNGemmFloatCommon_4;
    void (*MNNPackedMatMul)(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) = _SSE_MNNPackedMatMul;
    void (*MNNPackedMatMulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) = _SSE_MNNPackedMatMulRemain;
};

static FunctionGroup gFunc;
void MNNFunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        gFunc.MNNAddBias = _AVX_MNNAddBias;
        gFunc.MNNAddBiasRelu = _AVX_MNNAddBiasRelu;
        gFunc.MNNAddBiasRelu6 = _AVX_MNNAddBiasRelu6;
        gFunc.MNNMatrixAdd = _AVX_MNNMatrixAdd;
        gFunc.MNNMatrixSub = _AVX_MNNMatrixSub;
        gFunc.MNNGemmFloatUnit_4 = _AVX_MNNGemmFloatUnit_4;
        gFunc.MNNGemmFloatCommon_4 = _AVX_MNNGemmFloatCommon_4;
        gFunc.MNNPackedMatMul = _AVX_MNNPackedMatMul;
        gFunc.MNNPackedMatMulRemain = _AVX_MNNPackedMatMulRemain;
        if (cpuFlags & libyuv::kCpuHasFMA3) {
            gFunc.MNNGemmFloatUnit_4 = _AVX_MNNGemmFloatUnitFMA_4;
            gFunc.MNNGemmFloatCommon_4 = _AVX_MNNGemmFloatCommonFMA_4;
            gFunc.MNNPackedMatMul = _AVX_MNNPackedMatMulFMA;
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
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad) {
    {
        const auto dst_step_tmp = dst_step / sizeof(int8_t);
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            for (int w = 0; w < GEMM_INT8_DST_XUNIT; ++w) {
                const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
                auto dst_x         = dst_z + w * GEMM_INT8_UNIT;
                int32_t dstTemp[4] = {0, 0, 0, 0};

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                    const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;

                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                        for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                            dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                        }
                    }
                }

                for (int j = 0; j < 4; ++j) {
                    dst_x[j] = int32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j]);
                }
            }
        }
    }
 }

void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad){
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, bias, scale, src_depth_quad, dst_step, dst_depth_quad);
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return _SSE_MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
    const int mid = 1;
    auto ePack = e / 16;
    auto eDiv = UP_DIV(e, 16);
    auto lC4 = l / 4;
    auto lDiv = UP_DIV(l, 4);
    auto eRemain = ePack * 16;
    auto lRemain = lC4 * 4;
    if (eRemain != e) {
        ::memset(dest, 0, eDiv * l * 16 * mid * sizeof(float));
    }
    for (int y=0; y<ePack; ++y) {
        auto dstY = dest + y * l * 16;
        auto srcY = source + y * 64;
        for (int x=0; x<lC4 * mid; ++x) {
            auto srcX = srcY + x * 4 * eReal;
            auto dstX = dstY + x * 64;
            auto s00 = _mm_loadu_ps(srcX + 0 * 4);
            auto s01 = _mm_loadu_ps(srcX + 1 * 4);
            auto s02 = _mm_loadu_ps(srcX + 2 * 4);
            auto s03 = _mm_loadu_ps(srcX + 3 * 4);
            auto s10 = _mm_loadu_ps(srcX + 4 * 4);
            auto s11 = _mm_loadu_ps(srcX + 5 * 4);
            auto s12 = _mm_loadu_ps(srcX + 6 * 4);
            auto s13 = _mm_loadu_ps(srcX + 7 * 4);
            auto s20 = _mm_loadu_ps(srcX + 8 * 4);
            auto s21 = _mm_loadu_ps(srcX + 9 * 4);
            auto s22 = _mm_loadu_ps(srcX + 10 * 4);
            auto s23 = _mm_loadu_ps(srcX + 11 * 4);
            auto s30 = _mm_loadu_ps(srcX + 12 * 4);
            auto s31 = _mm_loadu_ps(srcX + 13 * 4);
            auto s32 = _mm_loadu_ps(srcX + 14 * 4);
            auto s33 = _mm_loadu_ps(srcX + 15 * 4);
            
            _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
            _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
            _MM_TRANSPOSE4_PS(s20, s21, s22, s23);
            _MM_TRANSPOSE4_PS(s30, s31, s32, s33);
            
            _mm_storeu_ps(dstX + 4 * 0, s00);
            _mm_storeu_ps(dstX + 4 * 4, s01);
            _mm_storeu_ps(dstX + 4 * 8, s02);
            _mm_storeu_ps(dstX + 4 * 12, s03);
            _mm_storeu_ps(dstX + 4 * 1, s10);
            _mm_storeu_ps(dstX + 4 * 5, s11);
            _mm_storeu_ps(dstX + 4 * 9, s12);
            _mm_storeu_ps(dstX + 4 * 13, s13);
            _mm_storeu_ps(dstX + 4 * 2, s20);
            _mm_storeu_ps(dstX + 4 * 6, s21);
            _mm_storeu_ps(dstX + 4 * 10, s22);
            _mm_storeu_ps(dstX + 4 * 14, s23);
            _mm_storeu_ps(dstX + 4 * 3, s30);
            _mm_storeu_ps(dstX + 4 * 7, s31);
            _mm_storeu_ps(dstX + 4 * 11, s32);
            _mm_storeu_ps(dstX + 4 * 15, s33);
        }
    }
    // Right
    for (int y=0; y<e; ++y) {
        auto yR = y % 16;
        auto yC = y / 16;
        for (int z=0; z<mid; ++z) {
            for (int x=lRemain; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x * mid + z) * 16 + yR + yC * 16 * l * mid] = source[(xC * mid + z) * eReal * 4 + y * 4 + xR];
            }
        }
    }
    // Down
    {
        auto yC = ePack;
        for (int y=eRemain; y<e; ++y) {
            auto yR = y - eRemain;
            for (int z=0; z<mid; ++z) {
                for (int x=0; x<lRemain; ++x) {
                    auto xR = x % 4;
                    auto xC = x / 4;
                    dest[(x * mid + z) * 16 + yR + yC * 16 * l * mid] = source[(xC + z * lDiv) * eReal * 4 + y * 4 + xR];
                }
            }
        }
    }
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = h / 6;
    auto hR = hP * 6;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 6)*6*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 6 * l;
            auto sourceY = source + y * 6;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 6 * x, sourceY + x * h, 6 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 6 * l;
            auto sourceY = source + hP * 6;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 6 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    // h, l -> h/6, l, 6, use 12 x 4 transpose
    hP = h / 12;
    hR = hP * 12;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    for (int y=0; y<hP; ++y) {
        auto dstY = dest + y * l * 12;
        auto srcY = source + y * l * 12;
        for (int x=0; x<lC4; ++x) {
            auto srcX = srcY + x * 4;
            auto dstX = dstY + x * 48;
            auto s00 = _mm_loadu_ps(srcX + 0 * l);
            auto s01 = _mm_loadu_ps(srcX + 1 * l);
            auto s02 = _mm_loadu_ps(srcX + 2 * l);
            auto s03 = _mm_loadu_ps(srcX + 3 * l);
            auto s04 = _mm_loadu_ps(srcX + 4 * l);
            auto s05 = _mm_loadu_ps(srcX + 5 * l);
            auto s06 = _mm_loadu_ps(srcX + 6 * l);
            auto s07 = _mm_loadu_ps(srcX + 7 * l);
            auto s08 = _mm_loadu_ps(srcX + 8 * l);
            auto s09 = _mm_loadu_ps(srcX + 9 * l);
            auto s10 = _mm_loadu_ps(srcX + 10 * l);
            auto s11 = _mm_loadu_ps(srcX + 11 * l);
            
            _MM_TRANSPOSE4_PS(s00, s03, s06, s09);
            _MM_TRANSPOSE4_PS(s01, s04, s07, s10);
            _MM_TRANSPOSE4_PS(s02, s05, s08, s11);
            
            _mm_storeu_ps(dstX + 4 * 0, s00);
            _mm_storeu_ps(dstX + 4 * 1, s01);
            _mm_storeu_ps(dstX + 4 * 2, s02);
            _mm_storeu_ps(dstX + 4 * 3, s03);
            _mm_storeu_ps(dstX + 4 * 4, s04);
            _mm_storeu_ps(dstX + 4 * 5, s05);
            _mm_storeu_ps(dstX + 4 * 6, s06);
            _mm_storeu_ps(dstX + 4 * 7, s07);
            _mm_storeu_ps(dstX + 4 * 8, s08);
            _mm_storeu_ps(dstX + 4 * 9, s09);
            _mm_storeu_ps(dstX + 4 * 10, s10);
            _mm_storeu_ps(dstX + 4 * 11, s11);
        }
    }
    
    // Right
    for (int y=0; y<h; ++y) {
        auto yR = y % 6;
        auto yC = y / 6;
        for (int x=lR; x<l; ++x) {
            dest[x * 6 + yR + yC * 6 * l] = source[x + y * l];
        }
    }
    // Down
    for (int y=0; y<h; ++y) {
        auto yR = y % 6;
        auto yC = y / 6;
        for (int x=0; x<l; ++x) {
            dest[x * 6 + yR + yC * 6 * l] = source[x + y * l];
        }
    }
}

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 6;
}

int MNNGetConvolutionTileNumber() {
    return gFunc.tileNumber;
}
void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    return gFunc.MNNPackedMatMul(C, A, B, parameter, cache, postParameters, bias);
}
void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    return gFunc.MNNPackedMatMulRemain(C, A, B, eSize, parameter, cache, postParameters, bias);
}
