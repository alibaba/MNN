//
//  FunctionDispatcher.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "DispatchHelper.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "sse/FunctionSummary.hpp"
#include "avx/FunctionSummary.hpp"

// https://stackoverflow.com/a/11230437
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)
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

// ========= CommonOptFunction.cpp ===========
void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBias(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBias(dst, bias, planeNumber, biasNumber);
    }
}

void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBiasRelu(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBiasRelu(dst, bias, planeNumber, biasNumber);
    }
}

void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBiasRelu6(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBiasRelu6(dst, bias, planeNumber, biasNumber);
    }
}

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNCopyC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNAddC4WithStride(source, dest, srcStride, dstStride, count);
}

// ========= MNNConvSlideWindowBorder.cpp ===========
void MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                              size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                              size_t dilateX_step, size_t dilateY_step, float* alpha) {
    _SSE_MNNConvSlideWindowBorder(dst, src, weight, src_depth_quad, src_depth_step, fw, fh,
                                  weight_y_step, weight_z_step, dilateX_step, dilateY_step, alpha);
}

// ========= MNNConvSlideWindowMiddle.cpp ===========
void MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                              size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                              size_t dilateY_step, float* alpha) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
    } else {
        _SSE_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
    }
}

void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNGemmFloatUnit_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, weight_depth_offset);
    } else {
        _SSE_MNNGemmFloatUnit_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, weight_depth_offset);
    }
}

// ========= MNNGemmFloatCommon_4.cpp ===========
void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    } else {
        _SSE_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    }
}

// ========= MNNMatrixAdd.cpp ===========
void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    } else {
        _SSE_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    }
}

// ========= MNNMatrixSub.cpp ===========
void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
    } else {
        _SSE_MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
    }
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
#ifdef MNN_OPTIMIZE_INT8_SSE
    if (cpu_feature_available(AVX)) {
        return _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, bias, scale, src_depth_quad, dst_step, dst_depth_quad);
    } else
#endif
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

void MNNPackForMatMul_A(float* dest, const float* source, size_t e, size_t l, bool transpose) {
    auto ePack = e / 16;
    auto lC4 = l / 4;
    auto eRemain = ePack * 16;
    auto lRemain = lC4 * 4;
    if (eRemain != e) {
        ::memset(dest, 0, UP_DIV(e, 16) * l * 16 * sizeof(float));
    }
    if (transpose) {
        for (int y=0; y<ePack; ++y) {
            auto dstY = dest + y * l * 16;
            auto srcY = source + y * 16;
            for (int x=0; x<l; ++x) {
                auto srcX = srcY + x * e;
                auto dstX = dstY + x * 16;
                ::memcpy(dstX, srcX, 16 * sizeof(float));
            }
        }
        if (eRemain != e) {
            auto remain = e - eRemain;
            auto dstY = dest + ePack * l * 16;
            auto srcY = source + ePack * 16;
            for (int x=0; x<l; ++x) {
                auto srcX = srcY + x * e;
                auto dstX = dstY + x * 16;
                ::memcpy(dstX, srcX, remain * sizeof(float));
            }
        }
        return;
    }
    for (int y=0; y<ePack; ++y) {
        auto dstY = dest + y * l * 16;
        auto srcY = source + y * l * 16;
        for (int x=0; x<lC4; ++x) {
            auto srcX = srcY + x * 4;
            auto dstX = dstY + x * 64;
            auto s00 = _mm_loadu_ps(srcX + 0 * l);
            auto s01 = _mm_loadu_ps(srcX + 1 * l);
            auto s02 = _mm_loadu_ps(srcX + 2 * l);
            auto s03 = _mm_loadu_ps(srcX + 3 * l);
            auto s10 = _mm_loadu_ps(srcX + 4 * l);
            auto s11 = _mm_loadu_ps(srcX + 5 * l);
            auto s12 = _mm_loadu_ps(srcX + 6 * l);
            auto s13 = _mm_loadu_ps(srcX + 7 * l);
            auto s20 = _mm_loadu_ps(srcX + 8 * l);
            auto s21 = _mm_loadu_ps(srcX + 9 * l);
            auto s22 = _mm_loadu_ps(srcX + 10 * l);
            auto s23 = _mm_loadu_ps(srcX + 11 * l);
            auto s30 = _mm_loadu_ps(srcX + 12 * l);
            auto s31 = _mm_loadu_ps(srcX + 13 * l);
            auto s32 = _mm_loadu_ps(srcX + 14 * l);
            auto s33 = _mm_loadu_ps(srcX + 15 * l);
            
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
        for (int x=lRemain; x<l; ++x) {
            dest[x * 16 + yR + yC * 16 * l] = source[x + y * l];
        }
    }
    // Down
    for (int y=eRemain; y<e; ++y) {
        auto yR = y % 16;
        auto yC = y / 16;
        for (int x=0; x<lRemain; ++x) {
            dest[x * 16 + yR + yC * 16 * l] = source[x + y * l];
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



void MNNUnpackForMatMul_C(float* dest, const float* source, size_t e, size_t h) {
    auto ePack = e / 16;
    auto hPack = h / 12;
    auto hP = UP_DIV(h, 6);
    auto eRemain = ePack * 16;
    auto hRemain = hPack * 12;

    for (int yC=0; yC<ePack; ++yC) {
        for (int xC=0; xC<hPack; ++xC) {
            auto dstX = dest + yC * 16 * h + xC * 12;
            auto srcX = source + yC * hP * 96 + xC * 192;
//            for (int u=0; u<16; ++u) {
//                for (int v=0; v<12; ++v) {
//                    dstX[u*h+v] = srcX[v*16+u];
//                }
//            }
            // 16 x 12 -> 1x3 - 16x4 transpose
            for (int v=0; v<3; ++v) {
                auto dstV = dstX + 4 * v;
                auto srcV = srcX + 64 * v;
                auto s00 = _mm_loadu_ps(srcV + 4 * 0);
                auto s01 = _mm_loadu_ps(srcV + 4 * 1);
                auto s02 = _mm_loadu_ps(srcV + 4 * 2);
                auto s03 = _mm_loadu_ps(srcV + 4 * 3);
                auto s10 = _mm_loadu_ps(srcV + 4 * 4);
                auto s11 = _mm_loadu_ps(srcV + 4 * 5);
                auto s12 = _mm_loadu_ps(srcV + 4 * 6);
                auto s13 = _mm_loadu_ps(srcV + 4 * 7);
                auto s20 = _mm_loadu_ps(srcV + 4 * 8);
                auto s21 = _mm_loadu_ps(srcV + 4 * 9);
                auto s22 = _mm_loadu_ps(srcV + 4 * 10);
                auto s23 = _mm_loadu_ps(srcV + 4 * 11);
                auto s30 = _mm_loadu_ps(srcV + 4 * 12);
                auto s31 = _mm_loadu_ps(srcV + 4 * 13);
                auto s32 = _mm_loadu_ps(srcV + 4 * 14);
                auto s33 = _mm_loadu_ps(srcV + 4 * 15);

                _MM_TRANSPOSE4_PS(s00, s10, s20, s30);
                _MM_TRANSPOSE4_PS(s01, s11, s21, s31);
                _MM_TRANSPOSE4_PS(s02, s12, s22, s32);
                _MM_TRANSPOSE4_PS(s03, s13, s23, s33);

                _mm_storeu_ps(dstV + h * 0, s00);
                _mm_storeu_ps(dstV + h * 1, s10);
                _mm_storeu_ps(dstV + h * 2, s20);
                _mm_storeu_ps(dstV + h * 3, s30);
                _mm_storeu_ps(dstV + h * 4, s01);
                _mm_storeu_ps(dstV + h * 5, s11);
                _mm_storeu_ps(dstV + h * 6, s21);
                _mm_storeu_ps(dstV + h * 7, s31);
                _mm_storeu_ps(dstV + h * 8, s02);
                _mm_storeu_ps(dstV + h * 9, s12);
                _mm_storeu_ps(dstV + h * 10, s22);
                _mm_storeu_ps(dstV + h * 11, s32);
                _mm_storeu_ps(dstV + h * 12, s03);
                _mm_storeu_ps(dstV + h * 13, s13);
                _mm_storeu_ps(dstV + h * 14, s23);
                _mm_storeu_ps(dstV + h * 15, s33);
            }
        }
    }
    for (int y=eRemain; y<e; ++y) {
        auto yR = y % 16;
        auto yC = y / 16;
        for (int x=0; x<h; ++x) {
            auto xR = x % 6;
            auto xC = x / 6;
            dest[y * h + x] = source[yC * hP * 96 + xC * 96 + xR * 16 + yR];
        }
    }
    for (int y=0; y<eRemain; ++y) {
        auto yR = y % 16;
        auto yC = y / 16;
        for (int x=hRemain; x<h; ++x) {
            auto xR = x % 6;
            auto xC = x / 6;
            dest[y * h + x] = source[yC * hP * 96 + xC * 96 + xR * 16 + yR];
        }
    }
}

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 6;
}
void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter) {
    return _AVX_MNNPackedMatMul(C, A, B, parameter);
}

extern "C" {
void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t length, size_t hSub) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNStrassenMergeCFunction(c11, c12, c21, c22, xAddr, cStride, length, hSub);
    } else {
        _SSE_MNNStrassenMergeCFunction(c11, c12, c21, c22, xAddr, cStride, length, hSub);
    }
}
}
