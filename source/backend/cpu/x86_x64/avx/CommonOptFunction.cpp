//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
void _AVX_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128*)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128*)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_max_ps(dstV, _mm_set1_ps(0.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    auto minV = _mm256_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128*)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            dstV      = _mm256_min_ps(dstV, minV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_min_ps(_mm_max_ps(dstV, _mm_set1_ps(0.0f)), _mm_set1_ps(6.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
            minV = _mm256_set1_ps(6.0f);
        }
    }
    _mm256_zeroall();
}

static void _postTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
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
                _mm_store_ps(dst + 4 * x, sum);
            }
        }
    } else {
        for (int y = 0; y < hC4; ++y) {
            auto dst = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_loadu_ps(dst + 4 * x);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_store_ps(dst + 4 * x, sum);
            }
        }
    }
}

void _AVX_MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8;
    auto p0    = _mm256_set1_ps(parameters[0]);
    auto p1    = _mm256_set1_ps(parameters[1]);
    auto p2    = _mm256_set1_ps(parameters[2]);
    auto p3    = _mm256_set1_ps(parameters[3]);
    auto p4    = _mm256_set1_ps(parameters[4]);
    auto p5    = _mm256_set1_ps(parameters[5]);
    auto p6    = _mm256_set1_ps(parameters[6]);
    auto p7    = _mm256_set1_ps(parameters[7]);
    auto xMax  = _mm256_set1_ps(87);
    auto xMin  = _mm256_set1_ps(-87);
    auto basic = _mm256_set1_epi32(1 << 23);
    auto temp127 = _mm256_set1_epi32(127);
    auto negZero = _mm256_set1_ps(-0.f);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm256_xor_ps(_mm256_loadu_ps(source + i * 8), negZero);
        x                 = _mm256_max_ps(x, xMin);
        x                 = _mm256_min_ps(x, xMax);
        auto div          = _mm256_mul_ps(x, p1);
        auto divInt       = _mm256_cvtps_epi32(div);
        div               = _mm256_cvtepi32_ps(divInt);
        auto div2         = _mm256_add_epi32(divInt, temp127);
        div2 = _mm256_mullo_epi32(div2, basic);
        auto expBasic  = _mm256_castsi256_ps(div2);
        auto xReamin   = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
        auto t         = xReamin;
        auto c0        = _mm256_mul_ps(p7, t);
        auto c1        = _mm256_add_ps(c0, p6);
        auto c2        = _mm256_mul_ps(c1, t);
        auto c3        = _mm256_add_ps(c2, p5);
        auto c4        = _mm256_mul_ps(c3, t);
        auto c5        = _mm256_add_ps(c4, p4);
        auto c6        = _mm256_mul_ps(c5, t);
        auto c7        = _mm256_add_ps(c6, p3);
        auto c8        = _mm256_mul_ps(c7, t);
        auto c9        = _mm256_add_ps(c8, p2);
        auto expRemain = c9;
        _mm256_storeu_ps(dest + 8 * i, _mm256_mul_ps(expBasic, expRemain));
    }
}

void _AVX_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 8;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    bool need4 = widthRemain >= 4;
    if (need4) {
        widthRemain-=4;
    }
    if (src_w_setup == 4) {
        for (int y = 0; y < height; ++y) {
            auto srcY = src + y * srcHStep;
            auto dstY = dst + y * dstHStep;
            for (dx = 0; dx < widthUnit; ++dx) {
                auto dstValue0 = _mm256_set1_ps(0.0f);
                auto dstValue1 = _mm256_set1_ps(0.0f);
                auto dstValue2 = _mm256_set1_ps(0.0f);
                auto dstValue3 = _mm256_set1_ps(0.0f);
                for (fy = 0; fy < fh; ++fy) {
                    const float* src_y    = srcY + fy * dilateY_step;
                    const float* weight_y = weight_z + fy * fw * 4;
                    for (fx = 0; fx < fw; ++fx) {
                        const float* src_x    = src_y + fx * dilateX_step;
                        const float* weight_x = weight_y + 4 * fx;
                        auto weightValue = _mm256_broadcast_ps((__m128*)weight_x);
                        dstValue0 = _mm256_add_ps(dstValue0, _mm256_mul_ps(_mm256_loadu_ps(src_x + 0 * 8), weightValue));
                        dstValue1 = _mm256_add_ps(dstValue1, _mm256_mul_ps(_mm256_loadu_ps(src_x + 1 * 8), weightValue));
                        dstValue2 = _mm256_add_ps(dstValue2, _mm256_mul_ps(_mm256_loadu_ps(src_x + 2 * 8), weightValue));
                        dstValue3 = _mm256_add_ps(dstValue3, _mm256_mul_ps(_mm256_loadu_ps(src_x + 3 * 8), weightValue));
                    }
                }
                _mm256_storeu_ps(dstY + 8 * 0, dstValue0);
                _mm256_storeu_ps(dstY + 8 * 1, dstValue1);
                _mm256_storeu_ps(dstY + 8 * 2, dstValue2);
                _mm256_storeu_ps(dstY + 8 * 3, dstValue3);
                dstY += 4 * unit;
                srcY += unit * src_w_setup;
            }
            if (need4) {
                auto dstValue0 = _mm256_set1_ps(0.0f);
                auto dstValue1 = _mm256_set1_ps(0.0f);
                for (fy = 0; fy < fh; ++fy) {
                    const float* src_y    = srcY + fy * dilateY_step;
                    const float* weight_y = weight_z + fy * fw * 4;
                    for (fx = 0; fx < fw; ++fx) {
                        const float* src_x    = src_y + fx * dilateX_step;
                        const float* weight_x = weight_y + 4 * fx;
                        auto weightValue = _mm256_broadcast_ps((__m128*)weight_x);
                        dstValue0 = _mm256_add_ps(dstValue0, _mm256_mul_ps(_mm256_loadu_ps(src_x + 0 * 8), weightValue));
                        dstValue1 = _mm256_add_ps(dstValue1, _mm256_mul_ps(_mm256_loadu_ps(src_x + 1 * 8), weightValue));
                    }
                }
                _mm256_storeu_ps(dstY + 8 * 0, dstValue0);
                _mm256_storeu_ps(dstY + 8 * 1, dstValue1);
                dstY += 4 * 4;
                srcY += 4 * src_w_setup;
            }
            for (dx = 0; dx < widthRemain; ++dx) {
                float* dst_x          = dstY + dx * 4;
                auto dstValue = _mm_set1_ps(0.0f);
                const float* src_z    = srcY + src_w_setup * dx;
                const float* weight_z = weight;
                for (fy = 0; fy < fh; ++fy) {
                    const float* src_y    = src_z + fy * dilateY_step;
                    const float* weight_y = weight_z + fy * fw * 4;
                    for (fx = 0; fx < fw; ++fx) {
                        const float* weight_x = weight_y + 4 * fx;
                        const float* src_x    = src_y + fx * dilateX_step;
                        dstValue = _mm_add_ps(dstValue, _mm_mul_ps(_mm_loadu_ps(src_x), _mm_loadu_ps(weight_x)));
                    }
                }
                _mm_storeu_ps(dst_x, dstValue);
            }
        }
        return;
    }
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            auto dstValue4 = _mm_set1_ps(0.0f);
            auto dstValue5 = _mm_set1_ps(0.0f);
            auto dstValue6 = _mm_set1_ps(0.0f);
            auto dstValue7 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                    dstValue4 = _mm_add_ps(dstValue4, _mm_mul_ps(_mm_loadu_ps(src_x + 4 * src_w_setup), weightValue));
                    dstValue5 = _mm_add_ps(dstValue5, _mm_mul_ps(_mm_loadu_ps(src_x + 5 * src_w_setup), weightValue));
                    dstValue6 = _mm_add_ps(dstValue6, _mm_mul_ps(_mm_loadu_ps(src_x + 6 * src_w_setup), weightValue));
                    dstValue7 = _mm_add_ps(dstValue7, _mm_mul_ps(_mm_loadu_ps(src_x + 7 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            _mm_storeu_ps(dstY + 4 * 4, dstValue4);
            _mm_storeu_ps(dstY + 4 * 5, dstValue5);
            _mm_storeu_ps(dstY + 4 * 6, dstValue6);
            _mm_storeu_ps(dstY + 4 * 7, dstValue7);
            dstY += 4 * unit;
            srcY += unit * src_w_setup;
        }
        if (need4) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            dstY += 4 * 4;
            srcY += 4 * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * 4;
            auto dstValue = _mm_set1_ps(0.0f);
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 4 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm_add_ps(dstValue, _mm_mul_ps(_mm_loadu_ps(src_x), _mm_loadu_ps(weight_x)));
                }
            }
            _mm_storeu_ps(dst_x, dstValue);
        }
    }
}

void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto sizeC2 = sizeQuad / 2;
    __m128 scaleValue = _mm_loadu_ps(scalep);
    {
        __m256 zero = _mm256_set1_ps(0);
        __m256 minValue = _mm256_set1_ps(minV);
        __m256 maxValue = _mm256_set1_ps(maxV);
        __m256 plus = _mm256_set1_ps(0.5f);
        __m256 minus = _mm256_set1_ps(-0.5f);
        __m256 scaleValue2 = _mm256_insertf128_ps(_mm256_castps128_ps256(scaleValue), scaleValue, 1);
        alignas(32) int32_t temp[8];
        for (int i = 0; i < sizeC2; ++i) {
            auto f0 = _mm256_loadu_ps(src);
            f0 = _mm256_mul_ps(f0, scaleValue2);
            f0 = _mm256_min_ps(f0, maxValue);
            f0 = _mm256_max_ps(f0, minValue);
            // 1: _CMP_LT_OS
            auto m0 = _mm256_cmp_ps(f0, zero, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            f0 = _mm256_add_ps(f0, m0);
            // 3: _MM_FROUND_TO_ZERO
            auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            *(__m256i*)temp = d0;
            for (int j=0; j<8; ++j) {
                dst[j] = temp[j];
            }

            src += 8;
            dst += 8;
        }
    }

    if (sizeQuad % 2 != 0) {
        __m128i zero = _mm_set1_epi32(0);
        __m128 minValue = _mm_set1_ps(minV);
        __m128 maxValue = _mm_set1_ps(maxV);
        __m128 plus = _mm_set1_ps(0.5f);
        __m128 minus = _mm_set1_ps(-0.5f);
        alignas(16) int32_t temp[4];
        __m128 f0 = _mm_loadu_ps(src);
        f0 = _mm_mul_ps(f0, scaleValue);
        f0 = _mm_min_ps(f0, maxValue);
        f0 = _mm_max_ps(f0, minValue);
        auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
        m0 = _mm_blendv_ps(plus, minus, m0);
        f0 = _mm_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
        *(__m128i*)temp = d0;
        for (int j=0; j<4; ++j) {
            dst[j] = temp[j];
        }
    }
}

void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    __m128i zero = _mm_set1_epi32(0);
    __m128 scaleValue = _mm_loadu_ps(scale);
    __m256 scaleValue2 = _mm256_insertf128_ps(_mm256_castps128_ps256(scaleValue), scaleValue, 1);
    for (int i = 0; i < sizeC4; ++i) {
        auto s0 = _mm_castps_si128(_mm_loadu_ps((const float*)src));
        auto s1 = _mm_unpackhi_epi64(s0, zero);
        auto Sf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s0));
        auto Sf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s1));
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(Sf0, scaleValue2));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(Sf1, scaleValue2));
        src += 16;
        dst += 16;
    }
    if (sizeRemain > 0) {
        alignas(16) int8_t srcTemp[16];
        ::memcpy(srcTemp, src, sizeRemain * 4);
        auto s0 = *(__m128i*)srcTemp;
        auto s1 = _mm_unpackhi_epi64(s0, zero);
        auto Sf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s0));
        auto Sf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s1));
        switch (sizeRemain) {
            case 3:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(Sf0, scaleValue2));
                _mm_storeu_ps(dst + 8 * 1, _mm_mul_ps(_mm256_extractf128_ps(Sf1, 0), scaleValue));
                break;
            case 2:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(Sf0, scaleValue2));
                break;
            case 1:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(_mm256_extractf128_ps(Sf0, 0), scaleValue));
                break;
            default:
                break;
        }
    }
}
void _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 4;
    int widthRemain = width % 4;
    auto weight = (const int16_t*)weightO;
    auto biasValue = _mm256_broadcastsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias)));
    //auto biasValue = *(__m128i*)parameters->bias;
    auto scaleValue = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)parameters->scale))));
    __m256i d0, d1;
    int dx, fx, fy;
    __m256i srcValue0;
    auto srcTemp0 = (int64_t*)(&srcValue0);
    __m256i weightValue;
    auto weightTemp = (int64_t*)(&weightValue);
    __m256i zero = _mm256_xor_si256(srcValue0, srcValue0);
    __m256 zero128 = _mm256_set1_ps(0.0f);
    __m128i minValue = _mm_set1_epi8(parameters->minValue);
    __m128i maxValue = _mm_set1_epi8(parameters->maxValue);
    __m256 plus = _mm256_set1_ps(0.5f);
    __m256 minus = _mm256_set1_ps(-0.5f);
    if (4 == src_w_step) {
        // Stride = 1
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto s0_16 = _mm256_castps_si256(_mm256_loadu_ps((float*)src_x));
                    auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
                    auto s1_32 = _mm256_unpackhi_epi16(s0_16, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightTemp[2] = *(int64_t*)weight_x;
                    weightValue = _mm256_unpacklo_epi16(weightValue, zero);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(weightValue, s0_32));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(weightValue, s1_32));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            // Int32 -> Int8
            d0 = _mm256_packs_epi32(d0, d1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);
            auto D = _mm_packs_epi16(D0, D1);
            D = _mm_min_epi8(D, maxValue);
            D = _mm_max_epi8(D, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(D));
            dst += 16;
            src += src_w_step * 4;
        }
    } else {
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + 1 * src_w_step);
                    srcTemp0[2] = *(int64_t*)(src_x + 2 * src_w_step);
                    srcTemp0[3] = *(int64_t*)(src_x + 3 * src_w_step);
                    auto s0_32 = _mm256_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm256_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightTemp[2] = *(int64_t*)weight_x;
                    weightValue = _mm256_unpacklo_epi16(weightValue, zero);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(weightValue, s0_32));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(weightValue, s1_32));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            // Int32 -> Int8
            d0 = _mm256_packs_epi32(d0, d1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);
            auto D = _mm_packs_epi16(D0, D1);
            D = _mm_min_epi8(D, maxValue);
            D = _mm_max_epi8(D, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(D));
            dst += 16;
            src += src_w_step * 4;
        }
    }
    switch (widthRemain) {
        case 3:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + 1 * src_w_step);
                    srcTemp0[2] = *(int64_t*)(src_x + 2 * src_w_step);
                    auto s0_32 = _mm256_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm256_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightTemp[2] = *(int64_t*)weight_x;
                    weightValue = _mm256_unpacklo_epi16(weightValue, zero);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(weightValue, s0_32));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(weightValue, s1_32));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            // Int32 -> Int8
            d0 = _mm256_packs_epi32(d0, d1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);
            auto D = _mm_packs_epi16(D0, D1);
            D = _mm_min_epi8(D, maxValue);
            D = _mm_max_epi8(D, minValue);
            int8_t temp[16];

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(D));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 2:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + 1 * src_w_step);
                    auto s0_32 = _mm256_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm256_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightTemp[2] = *(int64_t*)weight_x;
                    weightValue = _mm256_unpacklo_epi16(weightValue, zero);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(weightValue, s0_32));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(weightValue, s1_32));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            // Int32 -> Int8
            d0 = _mm256_packs_epi32(d0, d1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);
            auto D = _mm_packs_epi16(D0, D1);
            D = _mm_min_epi8(D, maxValue);
            D = _mm_max_epi8(D, minValue);
            int8_t temp[16];

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(D));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 1:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    auto s0_32 = _mm256_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm256_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightTemp[2] = *(int64_t*)weight_x;
                    weightValue = _mm256_unpacklo_epi16(weightValue, zero);
                    d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(weightValue, s0_32));
                    d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(weightValue, s1_32));
                }
            }
            __m256 f0 = _mm256_cvtepi32_ps(d0);
            __m256 f1 = _mm256_cvtepi32_ps(d1);
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            auto m0 = _mm256_cmp_ps(f0, zero128, 1);
            auto m1 = _mm256_cmp_ps(f1, zero128, 1);
            m0 = _mm256_blendv_ps(plus, minus, m0);
            m1 = _mm256_blendv_ps(plus, minus, m1);
            f0 = _mm256_add_ps(f0, m0);
            f1 = _mm256_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
            d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

            // Int32 -> Int8
            d0 = _mm256_packs_epi32(d0, d1);
            auto D0 = _mm256_extracti128_si256(d0, 0);
            auto D1 = _mm256_extracti128_si256(d0, 1);
            auto D = _mm_packs_epi16(D0, D1);
            D = _mm_min_epi8(D, maxValue);
            D = _mm_max_epi8(D, minValue);
            int8_t temp[16];

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(D));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }

        default:
            break;
    }
}

void _AVX_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 8;
    auto lR = lC4 * 8;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm256_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(A + x * 8), _mm256_loadu_ps(by + x * 8)));
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 1));
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 8;
        auto hR = hC4 * 8;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            auto sumValue = _mm256_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm256_loadu_ps(biasPtr + 8 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_broadcast_ss(A + x), _mm256_loadu_ps(bs + h * x)));
            }
            _mm256_storeu_ps(C + 8 * y, sumValue);
        }
        for (int y=hR; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}

void _AVX_MNNComputeMatMulForE_1FMA(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 8;
    auto lR = lC4 * 8;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm256_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm256_fmadd_ps(_mm256_loadu_ps(A + x * 8), _mm256_loadu_ps(by + x * 8), sumValue);
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 1));
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 8;
        auto hR = hC4 * 8;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            auto sumValue = _mm256_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm256_loadu_ps(biasPtr + 8 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_fmadd_ps(_mm256_broadcast_ss(A + x), _mm256_loadu_ps(bs + h * x), sumValue);
            }
            _mm256_storeu_ps(C + 8 * y, sumValue);
        }
        for (int y=hR; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}
