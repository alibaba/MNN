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
