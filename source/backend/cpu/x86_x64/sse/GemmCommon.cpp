//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <algorithm>
#include <cmath>

bool _SSE_MNNReorder4x4ByPlatform(float* dst, size_t number) {
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

void _SSE_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int xStride = info[3];
    int xS4 = xStride * 4;
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto source = sourceGroup[n];
        auto dest = destOrigin + eOffset + lOffset * eDest;
        const int pack   = 12;
        const int mid    = 1; // Deprecate
        const int packC4 = pack / 4;
        auto ePack       = e / pack;
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto eRemain     = ePack * pack;
        auto lRemain     = lC4 * 4;
        auto lRes        = l - lRemain;
        for (int y = 0; y < ePack; ++y) {
            auto dstY = dest + y * l * pack;
            auto srcY = source + y * pack * 4;
            for (int x = 0; x < lC4; ++x) {
                auto srcX = srcY + x * 4 * eReal;
                auto dstX = dstY + x * pack * 4;
                auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
                auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
                auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
                auto s03  = _mm_loadu_ps(srcX + 3 * xS4);
                auto s10  = _mm_loadu_ps(srcX + 4 * xS4);
                auto s11  = _mm_loadu_ps(srcX + 5 * xS4);
                auto s12  = _mm_loadu_ps(srcX + 6 * xS4);
                auto s13  = _mm_loadu_ps(srcX + 7 * xS4);
                auto s20  = _mm_loadu_ps(srcX + 8 * xS4);
                auto s21  = _mm_loadu_ps(srcX + 9 * xS4);
                auto s22  = _mm_loadu_ps(srcX + 10 * xS4);
                auto s23  = _mm_loadu_ps(srcX + 11 * xS4);

                _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
                _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
                _MM_TRANSPOSE4_PS(s20, s21, s22, s23);

    #define STORE_TEMP(i)                               \
        _mm_storeu_ps(dstX + 4 * (3 * i + 0), s##0##i); \
        _mm_storeu_ps(dstX + 4 * (3 * i + 1), s##1##i); \
        _mm_storeu_ps(dstX + 4 * (3 * i + 2), s##2##i);

                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
                STORE_TEMP(3);
            }
            if (lRes == 0) {
                continue;
            }
            auto srcX = srcY + lC4 * 4 * eReal;
            auto dstX = dstY + lC4 * eDest * 4;
            auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
            auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
            auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
            auto s03  = _mm_loadu_ps(srcX + 3 * xS4);
            auto s10  = _mm_loadu_ps(srcX + 4 * xS4);
            auto s11  = _mm_loadu_ps(srcX + 5 * xS4);
            auto s12  = _mm_loadu_ps(srcX + 6 * xS4);
            auto s13  = _mm_loadu_ps(srcX + 7 * xS4);
            auto s20  = _mm_loadu_ps(srcX + 8 * xS4);
            auto s21  = _mm_loadu_ps(srcX + 9 * xS4);
            auto s22  = _mm_loadu_ps(srcX + 10 * xS4);
            auto s23  = _mm_loadu_ps(srcX + 11 * xS4);

            _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
            _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
            _MM_TRANSPOSE4_PS(s20, s21, s22, s23);
            if (lRes == 3) {
                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
            } else if (lRes == 2) {
                STORE_TEMP(0);
                STORE_TEMP(1);
            } else {
                STORE_TEMP(0);
            }
        }
        // Down
        {
            auto eLast    = e - eRemain;
            auto lastDest = dest + ePack * pack * l;
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    lastDest[x * eDest + yR] = source[xC * eReal * 4 + y * 4 * xStride + xR];
                }
            }
        }
    }
}

void _SSE_GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                        const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto minValue     = _mm_set1_ps(postParameters[2]);
    auto maxValue     = _mm_set1_ps(postParameters[3]);
    if (nullptr != bias) {
        for (int y = 0; y < hC4; ++y) {
            auto biasValue = _mm_loadu_ps(bias + 4 * y);
            auto dst       = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst + 4 * x));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    } else {
        for (int y = 0; y < hC4; ++y) {
            auto dst = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_loadu_ps(dst + 4 * x);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    }
}

void _SSE_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    auto oneValue = _mm_set1_epi16(1);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
        const float* scale_dz = nullptr;
        if (post->scale != nullptr) {
            scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        }
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m128i d0 = _mm_set1_epi32(0);
        __m128i d1 = _mm_set1_epi32(0);
        __m128i d2 = _mm_set1_epi32(0);
        __m128i d3 = _mm_set1_epi32(0);

        __m128i e0 = _mm_set1_epi32(0);
        __m128i e1 = _mm_set1_epi32(0);
        __m128i e2 = _mm_set1_epi32(0);
        __m128i e3 = _mm_set1_epi32(0);

        __m128i D0 = _mm_set1_epi32(0);
        __m128i D1 = _mm_set1_epi32(0);
        __m128i D2 = _mm_set1_epi32(0);
        __m128i D3 = _mm_set1_epi32(0);

        __m128i E0 = _mm_set1_epi32(0);
        __m128i E1 = _mm_set1_epi32(0);
        __m128i E2 = _mm_set1_epi32(0);
        __m128i E3 = _mm_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
            const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
            auto w0 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
            auto w1 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 1));
            auto w2 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));
            auto w3 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 3));

            auto s0 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 0));
            auto s1 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 1));
            auto s2 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 2));
            auto s3 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 3));

//#define COMPUTE(i, j)\
//auto d##i##j = _mm_maddubs_epi16(s##i, w##j);\
//d##i##j = _mm_madd_epi16(d##i##j, oneValue);\

#define COMPUTE(i, j)\
auto W##i##j##0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, w##j), 8);\
auto W##i##j##1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, w##j), 8);\
auto S##i##j##0 = _mm_unpacklo_epi8(s##i, zero);\
auto S##i##j##1 = _mm_unpackhi_epi8(s##i, zero);\
auto d##i##j = _mm_add_epi32(_mm_madd_epi16(S##i##j##0, W##i##j##0), _mm_madd_epi16(S##i##j##1, W##i##j##1));\

            COMPUTE(0, 0);
            COMPUTE(0, 1);
            COMPUTE(0, 2);
            COMPUTE(0, 3);
            COMPUTE(1, 0);
            COMPUTE(1, 1);
            COMPUTE(1, 2);
            COMPUTE(1, 3);
            COMPUTE(2, 0);
            COMPUTE(2, 1);
            COMPUTE(2, 2);
            COMPUTE(2, 3);
            COMPUTE(3, 0);
            COMPUTE(3, 1);
            COMPUTE(3, 2);
            COMPUTE(3, 3);

            d0 = _mm_add_epi32(d0, d00);
            d1 = _mm_add_epi32(d1, d01);
            d2 = _mm_add_epi32(d2, d02);
            d3 = _mm_add_epi32(d3, d03);

            e0 = _mm_add_epi32(e0, d10);
            e1 = _mm_add_epi32(e1, d11);
            e2 = _mm_add_epi32(e2, d12);
            e3 = _mm_add_epi32(e3, d13);

            D0 = _mm_add_epi32(D0, d20);
            D1 = _mm_add_epi32(D1, d21);
            D2 = _mm_add_epi32(D2, d22);
            D3 = _mm_add_epi32(D3, d23);

            E0 = _mm_add_epi32(E0, d30);
            E1 = _mm_add_epi32(E1, d31);
            E2 = _mm_add_epi32(E2, d32);
            E3 = _mm_add_epi32(E3, d33);
        }
        d0 = _mm_hadd_epi32(d0, d1);
        d1 = _mm_hadd_epi32(d2, d3);
        d0 = _mm_hadd_epi32(d0, d1);

        e0 = _mm_hadd_epi32(e0, e1);
        e1 = _mm_hadd_epi32(e2, e3);
        d1 = _mm_hadd_epi32(e0, e1);

        D0 = _mm_hadd_epi32(D0, D1);
        D1 = _mm_hadd_epi32(D2, D3);
        d2 = _mm_hadd_epi32(D0, D1);

        E0 = _mm_hadd_epi32(E0, E1);
        E1 = _mm_hadd_epi32(E2, E3);
        d3 = _mm_hadd_epi32(E0, E1);
        
        if (post->scale != nullptr) {
            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            auto scaleValue = _mm_loadu_ps(scale_dz);
            d0 = _mm_add_epi32(d0, biasValue);
            d1 = _mm_add_epi32(d1, biasValue);
            d2 = _mm_add_epi32(d2, biasValue);
            d3 = _mm_add_epi32(d3, biasValue);
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
            auto m1 = _mm_cmplt_ps(f1, _mm_castsi128_ps(zero));
            auto m2 = _mm_cmplt_ps(f2, _mm_castsi128_ps(zero));
            auto m3 = _mm_cmplt_ps(f3, _mm_castsi128_ps(zero));
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            if (GEMM_INT8_DST_XUNIT == realDst) {
                _mm_storeu_ps((float*)dst_x, _mm_castsi128_ps(d0));
            } else {
                int32_t tempV[4];
                _mm_storeu_si128((__m128i*)tempV, d0);
                for (int j=0; j<realDst; ++j) {
                    ((int32_t*)dst_x)[j] = tempV[j];
                }
            }
        } else {
            auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
            __m128 f[4] = {
                _mm_cvtepi32_ps(_mm_add_epi32(d0, biasValue)),
                _mm_cvtepi32_ps(_mm_add_epi32(d1, biasValue)),
                _mm_cvtepi32_ps(_mm_add_epi32(d2, biasValue)),
                _mm_cvtepi32_ps(_mm_add_epi32(d3, biasValue)),
            };
            for (int j = 0; j < realDst; ++j) {
                _mm_storeu_ps(((float*)dst_x) + j * 4, f[j]);
            }
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * 4;
        auto srcPlane = src + z * srcAreaOffset * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 4 * x;
            auto d  = dstPlane + 16 * x;
            auto s0 = _mm_loadu_ps(s + 0 * srcAreaOffset);
            auto s1 = _mm_loadu_ps(s + 1 * srcAreaOffset);
            auto s2 = _mm_loadu_ps(s + 2 * srcAreaOffset);
            auto s3 = _mm_loadu_ps(s + 3 * srcAreaOffset);

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
        float* dstPlane       = depthC4 * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[4 * x + y] = srcPlane[y * srcAreaOffset + x];
            }
            for (int y = remain; y < 4; y++) {
                dstPlane[4 * x + y] = 0;
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        float* dstPlane       = z * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + z * srcAreaOffset * 4;
        for (int x = areaRemain; x < area; ++x) {
            float s0 = srcPlane[x];
            float s1 = srcPlane[x + srcAreaOffset];
            float s2 = srcPlane[x + srcAreaOffset * 2];
            float s3 = srcPlane[x + srcAreaOffset * 3];
            _mm_storeu_ps(dstPlane + 4 * x, _mm_set_ps(s3, s2, s1, s0));
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

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    auto areaC4  = area / 4;
    auto depthC4 = depth / 4;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    for (int z = 0; z < depthC4; ++z) {
        auto dstPlane = dst + z * dstAreaOffset * 4;
        auto srcPlane = src + z * srcAreaOffset * 4;
        for (int x = 0; x < areaC4; ++x) {
            auto s  = srcPlane + 16 * x;
            auto d  = dstPlane + 4 * x;
            auto s0 = _mm_loadu_ps(s + 0 * 4);
            auto s1 = _mm_loadu_ps(s + 1 * 4);
            auto s2 = _mm_loadu_ps(s + 2 * 4);
            auto s3 = _mm_loadu_ps(s + 3 * 4);

            _MM_TRANSPOSE4_PS(s0, s1, s2, s3);

            _mm_storeu_ps(d + 0 * dstAreaOffset, s0);
            _mm_storeu_ps(d + 1 * dstAreaOffset, s1);
            _mm_storeu_ps(d + 2 * dstAreaOffset, s2);
            _mm_storeu_ps(d + 3 * dstAreaOffset, s3);
        }
    }
    auto areaRemain  = areaC4 * 4;
    auto depthRemain = depthC4 * 4;
    // Down
    int remain = depth - depthRemain;
    if (remain > 0) {
        float* dstPlane       = depthC4 * dstAreaOffset * 4 + dst;
        const float* srcPlane = src + depthC4 * srcAreaOffset * 4;
        for (int x = 0; x < area; ++x) {
            for (int y = 0; y < remain; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[4 * x + y];
            }
        }
    }
    // Right
    for (int z = 0; z < depthC4; ++z) {
        const float* srcPlane = z * srcAreaOffset * 4 + src;
        float* dstPlane       = dst + z * dstAreaOffset * 4;
        for (int x = areaRemain; x < area; ++x) {
            for (int y = 0; y < 4; y++) {
                dstPlane[y * dstAreaOffset + x] = srcPlane[4 * x + y];
            }
        }
    }
}

void _SSE_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[2] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        MNNUnpackTranspose(dest, source, l, h, offset);
        return;
    }
    MNNPackC4(dest, source, l, h, offset);
}

void _SSE_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        MNNUnpackTransposeInt16((int16_t*)dest, (const int16_t*)source, l, h, offset);
        return;
    }
    MNNPackC4Int16((int16_t*)dest, (const int16_t*)source, l, h, offset);
}

void _SSE_MNNPackedSparseMatMul(float* C, const float* A, const float* B, unsigned int* NNZMap, int* dataOffsetMap, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    // sse version
    MNN_ASSERT(false);
    return;
}
