//
//  MathFunctions.cpp
//  MNN
//
//  Created by MNN on b'2021/07/09'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <emmintrin.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include "core/Macro.h"
#include "FunctionSummary.hpp"

void _SSE_MNNExpC8(float* dest, const float* source, const float* offset, const float* parameters, size_t countC8) {
    auto count = countC8 * 2;
    auto A     = _mm_set1_ps(offset[0]);
    auto B    = _mm_set1_ps(offset[1]);
    auto p0    = _mm_set1_ps(parameters[0]);
    auto p1    = _mm_set1_ps(parameters[1]);
    auto p2    = _mm_set1_ps(parameters[2]);
    auto p3    = _mm_set1_ps(parameters[3]);
    auto p4    = _mm_set1_ps(parameters[4]);
    auto p5    = _mm_set1_ps(parameters[5]);
    auto p6    = _mm_set1_ps(parameters[6]);
    auto p7    = _mm_set1_ps(parameters[7]);
    auto xMax  = _mm_set1_ps(87);
    auto xMin  = _mm_set1_ps(-87);
    // auto basic = _mm_set1_epi32(1 << 23);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm_mul_ps(_mm_loadu_ps(source + i * 4), A);
        x                 = _mm_max_ps(x, xMin);
        x                 = _mm_min_ps(x, xMax);
        auto div          = _mm_mul_ps(x, p1);
        auto divInt       = _mm_cvtps_epi32(div);
        div               = _mm_cvtepi32_ps(divInt);
        auto div2         = _mm_add_epi32(divInt, _mm_set1_epi32(127));
        // div2 = _mm_mullo_epi32(div2, basic);
        div2 = _mm_slli_epi32(div2, 23);
        auto expBasic  = _mm_castsi128_ps(div2);
        auto xReamin   = _mm_sub_ps(x, _mm_mul_ps(div, p0));
        auto t         = xReamin;
        auto c0        = _mm_mul_ps(p7, t);
        auto c1        = _mm_add_ps(c0, p6);
        auto c2        = _mm_mul_ps(c1, t);
        auto c3        = _mm_add_ps(c2, p5);
        auto c4        = _mm_mul_ps(c3, t);
        auto c5        = _mm_add_ps(c4, p4);
        auto c6        = _mm_mul_ps(c5, t);
        auto c7        = _mm_add_ps(c6, p3);
        auto c8        = _mm_mul_ps(c7, t);
        auto c9        = _mm_add_ps(c8, p2);
        auto expRemain = c9;
        _mm_storeu_ps(dest + 4 * i, _mm_add_ps(_mm_mul_ps(expBasic, expRemain), B));
    }
}

void _SSE_MNNSoftmax(float* dest, const float* source, size_t size) {
    float tmpfloat4[4];
    int count  = size / 4;
    int remain = count * 4;
    // step 1: get maxValue
    float maxValue = source[0];
    if (count > 0) {
        auto maxVal = _mm_loadu_ps(source);
        for (int i = 1; i < count; i++) {
            maxVal = _mm_max_ps(maxVal, _mm_loadu_ps(source + i * 4));
        }
        _mm_storeu_ps(tmpfloat4, maxVal);
        maxValue = tmpfloat4[0] > tmpfloat4[1] ? tmpfloat4[0] : tmpfloat4[1];
        maxValue = maxValue > tmpfloat4[2] ? maxValue : tmpfloat4[2];
        maxValue = maxValue > tmpfloat4[3] ? maxValue : tmpfloat4[3];
    }
    for (int i = remain; i < size; i++) {
        maxValue = maxValue > source[i] ? maxValue : source[i];
    }

    // step 2: get exp(x - maxValue) and sum(exp(x - maxValue))
    float sumValue = 0.f;
    if (count > 0) {
        auto sumVal = _mm_set1_ps(0.f);
        auto p0    = _mm_set1_ps(0.6931471805599453);
        auto p1    = _mm_set1_ps(1.4426950408889634);
        auto p2    = _mm_set1_ps(1.f);
        auto p3    = _mm_set1_ps(1.f);
        auto p4    = _mm_set1_ps(0.5);
        auto p5    = _mm_set1_ps(0.1666666666666666);
        auto p6    = _mm_set1_ps(0.041666666666666664);
        auto p7    = _mm_set1_ps(0.008333333333333333);
        auto xMax  = _mm_set1_ps(87);
        auto xMin  = _mm_set1_ps(-87);
        // auto basic = _mm_set1_epi32(1 << 23);
        for (int i = 0; i < count; ++i) {
            auto x            = _mm_sub_ps(_mm_loadu_ps(source + i * 4), _mm_set1_ps(maxValue));
            x                 = _mm_max_ps(x, xMin);
            x                 = _mm_min_ps(x, xMax);
            auto div          = _mm_mul_ps(x, p1);
            auto divInt       = _mm_cvtps_epi32(div);
            div               = _mm_cvtepi32_ps(divInt);
            auto div2         = _mm_add_epi32(divInt, _mm_set1_epi32(127));
            // div2 = _mm_mullo_epi32(div2, basic);
            div2 = _mm_slli_epi32(div2, 23);
            auto expBasic  = _mm_castsi128_ps(div2);
            auto xReamin   = _mm_sub_ps(x, _mm_mul_ps(div, p0));
            auto t         = xReamin;
            auto c0        = _mm_mul_ps(p7, t);
            auto c1        = _mm_add_ps(c0, p6);
            auto c2        = _mm_mul_ps(c1, t);
            auto c3        = _mm_add_ps(c2, p5);
            auto c4        = _mm_mul_ps(c3, t);
            auto c5        = _mm_add_ps(c4, p4);
            auto c6        = _mm_mul_ps(c5, t);
            auto c7        = _mm_add_ps(c6, p3);
            auto c8        = _mm_mul_ps(c7, t);
            auto c9        = _mm_add_ps(c8, p2);
            auto expRemain = c9;
            auto expRes    = _mm_mul_ps(expBasic, expRemain);
            sumVal         = _mm_add_ps(expRes, sumVal);
            _mm_storeu_ps(dest + 4 * i, expRes);
        }
        _mm_storeu_ps(tmpfloat4, sumVal);
        sumValue = tmpfloat4[0] + tmpfloat4[1] + tmpfloat4[2] + tmpfloat4[3];
    }
    auto param = 0.6931471805599453;
    float xLimit = 87;
    for (int i = remain; i < size; i++) {
        auto x         = source[i] - maxValue;
        x = x > -xLimit ? x : -xLimit;
        x = x < xLimit ? x : xLimit;

        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);

        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dest[i]  = expBasic * expRemain;
        sumValue += dest[i];
    }
    // step 3: get x / sum and store
    for (int i = 0; i < count; ++i) {
        // using  1 / ((1 / x) * sum) instead x * (1 / sum) or x / sum for some bugs in intel cpu
        auto x = _mm_rcp_ps(_mm_loadu_ps(dest + 4 * i));
        auto y = _mm_set1_ps(sumValue);
        auto z = _mm_rcp_ps(_mm_mul_ps(x, y));
        _mm_storeu_ps(dest + 4 * i, z);
    }
    sumValue = 1.f / sumValue;
    for (int i = remain; i < size; i++) {
        dest[i] *= sumValue;
    }
}

void _SSE_MNNGelu(float* dst, const float* src, size_t size) {
    auto var1 = _mm_set1_ps(0.044715f);
    auto var2 = _mm_set1_ps(0.79788458f);
    auto var3 = _mm_set1_ps(378.f);
    auto var4 = _mm_set1_ps(17325.f);
    auto var5 = _mm_set1_ps(135135.f);
    auto var6 = _mm_set1_ps(28.f);
    auto var7 = _mm_set1_ps(3150.f);
    auto var8 = _mm_set1_ps(62370.f);
    auto var9 = _mm_set1_ps(135135.f);
    auto var10 = _mm_set1_ps(0.5);
    auto varOne = _mm_set1_ps(1.f);
    auto varNegOne = _mm_set1_ps(-1.f);
    for (int i = 0; i < size * 2; i++) {
        auto x = _mm_loadu_ps(src + i * 4);
        auto y = _mm_mul_ps(x, x);
        y = _mm_mul_ps(y, x);
        y = _mm_mul_ps(y, var1);
        y = _mm_add_ps(y, x);
        y = _mm_mul_ps(y, var2);
        // y = tanh(y)
        {
            auto y2 = _mm_mul_ps(y, y);
            auto w = _mm_add_ps(y2, var3);
            w = _mm_mul_ps(w, y2);
            w = _mm_add_ps(w, var4);
            w = _mm_mul_ps(w, y2);
            w = _mm_add_ps(w, var5);
            w = _mm_mul_ps(w, y);
            auto z = _mm_mul_ps(y2, var6);
            z = _mm_add_ps(z, var7);
            z = _mm_mul_ps(z, y2);
            z = _mm_add_ps(z, var8);
            z = _mm_mul_ps(z, y2);
            z = _mm_add_ps(z, var9);
            z = _mm_div_ps(w, z);
            z = _mm_max_ps(z, varNegOne);
            y = _mm_min_ps(z, varOne);
        }
        y = _mm_add_ps(y, varOne);
        y = _mm_mul_ps(y, x);
        y = _mm_mul_ps(y, var10);
        _mm_storeu_ps(dst + i * 4, y);
    }
}

void _SSE_MNNHardSwish(float* dst, const float* src, size_t size) {
    auto zero = _mm_set1_ps(0.f);
    auto three = _mm_set1_ps(3.f);
    auto six = _mm_set1_ps(6.f);
    for (int i = 0; i < size; i++) {
        auto x = _mm_loadu_ps(src + 4 * i);
        _mm_storeu_ps(dst + 4 * i, _mm_div_ps(_mm_mul_ps(x, _mm_min_ps(_mm_max_ps(_mm_add_ps(x, three), zero), six)), six));
    }
}

void _SSE_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size) {
    float tmpfloat4[4];
    int count  = size / 4;
    int remain = count * 4;
    // step 1: get sum
    float sum = 0.f;
    if (count > 0) {
        auto sumVal = _mm_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            sumVal = _mm_add_ps(sumVal, _mm_loadu_ps(src + i * 4));
        }
        _mm_storeu_ps(tmpfloat4, sumVal);
        sum += (tmpfloat4[0] + tmpfloat4[1] + tmpfloat4[2] + tmpfloat4[3]);
    }
    for (int i = remain; i < size; i++) {
        sum += src[i];
    }
    // step 2: get square_sum
    float mean = sum / size;
    float square_sum = 0.f;
    auto meanVal = _mm_set1_ps(mean);
    if (count > 0) {
        auto sumVal = _mm_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            auto x = _mm_sub_ps(_mm_loadu_ps(src + i * 4), meanVal);
            sumVal = _mm_add_ps(sumVal, _mm_mul_ps(x, x));
        }
        _mm_storeu_ps(tmpfloat4, sumVal);
        square_sum += (tmpfloat4[0] + tmpfloat4[1] + tmpfloat4[2] + tmpfloat4[3]);
    }
    for (int i = remain; i < size; i++) {
        float x = (src[i] - mean);
        square_sum += x * x;
    }
    // step 3: get result
    float variable = square_sum / size;
    variable = 1.f / sqrt(variable + epsilon);
    auto variableVal = _mm_set1_ps(variable);
    if (gamma && beta) {
        for (int i = 0; i < count; i++) {
            auto x = _mm_sub_ps(_mm_loadu_ps(src + i * 4), meanVal);
            auto g = _mm_loadu_ps(gamma + i * 4);
            auto b = _mm_loadu_ps(beta + i * 4);
            auto y = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(x, g), variableVal), b);
            _mm_storeu_ps(dst + i * 4, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * gamma[i] * variable + beta[i] ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            auto x = _mm_sub_ps(_mm_loadu_ps(src + i * 4), meanVal);
            auto y = _mm_mul_ps(x, variableVal);
            _mm_storeu_ps(dst + i * 4, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * variable;
        }
    }
}
