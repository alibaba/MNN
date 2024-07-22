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

void _SSE_MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8 * 2;
    auto A     = _mm_set1_ps(offset[0]);
    auto B    = _mm_set1_ps(offset[1]);
    auto C    = _mm_set1_ps(offset[2]);
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
    auto summer = _mm_setzero_ps();
    // auto basic = _mm_set1_epi32(1 << 23);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm_mul_ps(_mm_loadu_ps(source + i * 4), A);
        x = _mm_add_ps(x, C);
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
        auto res = _mm_add_ps(_mm_mul_ps(expBasic, expRemain), B);
        _mm_storeu_ps(dest + 4 * i, res);
        summer = _mm_add_ps(summer, res);
    }
    float tmp[4];
    _mm_storeu_ps(tmp, summer);
    float total = offset[3];
    for (int i=0; i<4; ++i) {
        total+=tmp[i];
    }
    offset[3] = total;
}

void _SSE_MNNSoftmax(float* dest, const float* source, size_t size) {
    float tmpfloat4[4];
    int count  = static_cast<int32_t>(size / 4);
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

void _SSE_MNNGelu(float* dst, const float* src, size_t size, float* parameters) {
    // parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    auto var1 = _mm_set1_ps(parameters[0]);
    auto var2 = _mm_set1_ps(parameters[1]);
    auto var3 = _mm_set1_ps(parameters[2]);
    auto var4 = _mm_set1_ps(parameters[3]);
    auto var5 = _mm_set1_ps(parameters[4]);
    auto var6 = _mm_set1_ps(parameters[5]);
    auto var7 = _mm_set1_ps(parameters[6]);
    auto var8 = _mm_set1_ps(parameters[7]);
    auto var9 = _mm_set1_ps(parameters[4]);
    auto var10 = _mm_set1_ps(0.5);
    auto varOne = _mm_set1_ps(1.f);
    auto varNegOne = _mm_set1_ps(-1.f);
    auto clamp_min = _mm_set1_ps(-5.0f);
    auto clamp_max = _mm_set1_ps(5.0f);
    for (int i = 0; i < size * 2; i++) {
        auto x = _mm_loadu_ps(src + i * 4);
        auto y = _mm_mul_ps(x, x);
        y = _mm_mul_ps(y, x);
        y = _mm_mul_ps(y, var1);
        y = _mm_add_ps(y, x);
        y = _mm_mul_ps(y, var2);
        y = _mm_max_ps(y, clamp_min);
        y = _mm_min_ps(y, clamp_max);
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

void _SSE_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    float tmpfloat4[4];
    int count  = static_cast<int32_t>(size / 4);
    int remain = count * 4;
    float mean = 0;
    if(!RMSNorm){
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
        mean = sum / size;
    }
    // step 2: get square_sum
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

void _SSE_MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, QuanPrePostParameters *params) {
    uint8_t* dstO = (uint8_t*)dst;
    uint8_t* srcO = (uint8_t*)src;
    auto outputZero = _mm_set1_ps(static_cast<float>(params->outputZeroPoint[0]));
    __m128 maxValue = _mm_set1_ps(params->maxValue);
    __m128 minValue = _mm_set1_ps(params->minValue);
    auto offset = _mm_set1_epi32(128);
    auto zero = _mm_set1_epi32(0);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    __m128i zeroPointValue = _mm_set1_epi32(static_cast<int32_t>(params->inputZeroPoint[0]) + 128);
    for (int j = 0;j < depthQuad; ++j) {
        auto slopeZ = _mm_loadu_ps(slope + 4 * j);
        const uint8_t* srcZ = srcO + 4 * j * planeNumber;
        uint8_t* dstZ = dstO + 4 * j * planeNumber;
        int32_t srcZ_ext[4] = {*(int32_t*)srcZ, 0, 0, 0};
        for (int i = 0; i < planeNumber; ++i) {
            // auto srcData8 = _mm_loadu_si32(srcZ);
            auto srcData8 = _mm_castps_si128(_mm_loadu_ps((float*)srcZ_ext));
            auto srcData16 = _mm_unpacklo_epi8(srcData8, zero);
            auto srcData32 = _mm_unpacklo_epi16(srcData16, zero);
            srcData32 = _mm_sub_epi32(srcData32, zeroPointValue);
            auto srcDataf  = _mm_cvtepi32_ps(srcData32);
            auto mask1 = _mm_cmplt_ps(srcDataf, _mm_castsi128_ps(zero));
            auto mask0 = _mm_cmpge_ps(srcDataf, _mm_castsi128_ps(zero));
            auto f = _mm_mul_ps(srcDataf, slopeZ);
            f = _mm_add_ps(f, outputZero);
            f = _mm_min_ps(f, maxValue);
            f = _mm_max_ps(f, minValue);
            auto r = _mm_add_ps(_mm_and_ps(srcDataf, mask0), _mm_and_ps(f, mask1));
            auto m0 = _mm_cmplt_ps(r, _mm_castsi128_ps(zero));
            m0 = _mm_blendv_ps(plus, minus, m0);
            r = _mm_add_ps(r, m0);
            // Round to zero
            auto d0 = _mm_cvtps_epi32(_mm_round_ps(r, 3));
            d0 = _mm_add_epi32(d0, offset);
            d0 = _mm_packs_epi32(d0, d0);
            d0 = _mm_packus_epi16(d0, d0);
            *((int*)dstZ + i) = _mm_cvtsi128_si32(d0);
        }
    }
}

