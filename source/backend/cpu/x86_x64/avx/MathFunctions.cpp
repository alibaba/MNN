//
//  MathFunctions.cpp
//  MNN
//
//  Created by MNN on b'2021/07/05'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include <math.h>

void _AVX_MNNGelu(float *dst, const float *src, size_t size, float* parameters) {
    // parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    auto var1 = _mm256_set1_ps(parameters[0]);
    auto var2 = _mm256_set1_ps(parameters[1]);
    auto var3 = _mm256_set1_ps(parameters[2]);
    auto var4 = _mm256_set1_ps(parameters[3]);
    auto var5 = _mm256_set1_ps(parameters[4]);
    auto var6 = _mm256_set1_ps(parameters[5]);
    auto var7 = _mm256_set1_ps(parameters[6]);
    auto var8 = _mm256_set1_ps(parameters[7]);
    auto var9 = _mm256_set1_ps(parameters[4]);
    auto var10 = _mm256_set1_ps(0.5);
    auto varOne = _mm256_set1_ps(1.f);
    auto varNegOne = _mm256_set1_ps(-1.f);
    auto clamp_min = _mm256_set1_ps(-5.0f);
    auto clamp_max = _mm256_set1_ps(5.0f);
    for (int i = 0; i < size; i++) {
        auto x = _mm256_loadu_ps(src + i * 8);
        auto y = _mm256_mul_ps(x, x);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var1);
        y = _mm256_add_ps(y, x);
        y = _mm256_mul_ps(y, var2);
        y = _mm256_max_ps(y, clamp_min);
        y = _mm256_min_ps(y, clamp_max);
        // y = tanh(y)
        {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
        }
        y = _mm256_add_ps(y, varOne);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var10);
        _mm256_storeu_ps(dst + i * 8, y);
    }
}
void _AVX_MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8;
    auto A     = _mm256_broadcast_ss(offset + 0);
    auto B     = _mm256_broadcast_ss(offset + 1);
    auto C     = _mm256_broadcast_ss(offset + 2);
    auto p0    = _mm256_set1_ps(parameters[0]);
    auto p1    = _mm256_set1_ps(parameters[1]);
    auto p2    = _mm256_set1_ps(0.25f);
    auto p3    = _mm256_set1_ps(1.0f);
    auto p4    = _mm256_set1_ps(parameters[4]);
    auto p5    = _mm256_set1_ps(parameters[5]);
    auto p6    = _mm256_set1_ps(parameters[6]);
    auto p7    = _mm256_set1_ps(parameters[7]);
    auto xMax  = _mm256_set1_ps(87);
    auto xMin  = _mm256_set1_ps(-87);
    auto basic = _mm256_set1_epi32(1 << 23);
    auto temp127 = _mm256_set1_epi32(127);
    auto negZero = _mm256_set1_ps(-0.f);
    auto summer = _mm256_setzero_ps();
    for (int i = 0; i < count; ++i) {
        auto x            = _mm256_mul_ps(_mm256_loadu_ps(source + i * 8), A);
        x = _mm256_add_ps(x, C);
        x                 = _mm256_max_ps(x, xMin);
        x                 = _mm256_min_ps(x, xMax);
        auto div          = _mm256_mul_ps(x, p1);
        auto divInt       = _mm256_cvtps_epi32(div);
        div               = _mm256_cvtepi32_ps(divInt);
        auto div2         = _mm256_add_epi32(divInt, temp127);
        div2 = _mm256_mullo_epi32(div2, basic);
        auto expBasic  = _mm256_castsi256_ps(div2);
        auto xReamin   = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
        auto t         = _mm256_mul_ps(xReamin, p2);
        auto c0        = _mm256_mul_ps(p7, t);
        auto c1        = _mm256_add_ps(c0, p6);
        auto c2        = _mm256_mul_ps(c1, t);
        auto c3        = _mm256_add_ps(c2, p5);
        auto c4        = _mm256_mul_ps(c3, t);
        auto c5        = _mm256_add_ps(c4, p4);
        auto c6        = _mm256_mul_ps(c5, t);
        auto c7        = _mm256_add_ps(c6, p3);
        auto c8        = _mm256_mul_ps(c7, t);
        auto c9        = _mm256_add_ps(c8, p3);
        auto expRemain = _mm256_mul_ps(c9, c9);
        expRemain = _mm256_mul_ps(expRemain, expRemain);
        auto res =  _mm256_add_ps(_mm256_mul_ps(expBasic, expRemain), B);
        _mm256_storeu_ps(dest + 8 * i, res);
        summer = _mm256_add_ps(summer, res);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, summer);
    float total = offset[3];
    for (int i=0; i<8; ++i) {
        total+=tmp[i];
    }
    offset[3] = total;
}


void _AVX_MNNSoftmax(float* softmaxDst, float* input, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize) {
    const float xLimit = 87.0f;
    const float param = 0.6931471805599453f; // ln(2)
    const float inv_param = 1.0f / param;
    const int32_t exp_offset = 127;
    const float exp_scale = 8388608.0f; // 2^23

    for (int k = 0; k < outside; ++k) {
        float* source = input + k * reduceSize;
        float* dest = softmaxDst + k * reduceSize;

        float tmpfloat8[8];
        int count  = reduceSize/ 8;
        int remain = count * 8;
        // step 1: get maxValue
        float maxValue = source[0];

        float oldMax = maxValue;
        if (runningMax) {
            oldMax = runningMax[k];
        }

        if (count > 0) {
            auto maxVal = _mm256_loadu_ps(source);
            for (int i = 1; i < count; i++) {
                maxVal = _mm256_max_ps(maxVal, _mm256_loadu_ps(source + i * 8));
            }
            _mm256_storeu_ps(tmpfloat8, maxVal);
            maxValue = tmpfloat8[0] > tmpfloat8[1] ? tmpfloat8[0] : tmpfloat8[1];
            for (int i = 2; i < 8; i++) {
                maxValue = maxValue > tmpfloat8[i] ? maxValue : tmpfloat8[i];
            }
        }
        for (int i = remain; i < reduceSize; i++) {
            maxValue = maxValue > source[i] ? maxValue : source[i];
        }

        float newMax = ALIMAX(oldMax, maxValue);

        // step 2: get exp(x - newMax) and sum(exp(x - newMax))
        float exprOffset[4] = {1.0f, 0.0f, 0.0f, 0.0f };
        exprOffset[2] = -newMax;
        MNNExp(dest, source, exprOffset, reduceSize);
        float sumValue = exprOffset[3];

        if (runningMax != nullptr && runningSum != nullptr && updateScale != nullptr) {
            // === Step 3: Update running variables ===
            float scale = expf(oldMax - newMax);
            runningSum[k] = runningSum[k] * scale + sumValue;
            runningMax[k] = newMax;
            updateScale[k] = scale;
        } else {
            // step 3: get x / sum and store
            for (int i = 0; i < count; ++i) {
                // using  1 / ((1 / x) * sum) instead x * (1 / sum) or x / sum for some bugs in intel cpu
                auto x = _mm256_rcp_ps(_mm256_loadu_ps(dest + 8 * i));
                auto y = _mm256_set1_ps(sumValue);
                auto z = _mm256_rcp_ps(_mm256_mul_ps(x, y));
                _mm256_storeu_ps(dest + 8 * i, z);
            }
            auto scale = 1.f / sumValue;
            for (int i = remain; i < reduceSize; i++) {
                dest[i] *= scale;
            }
        }
    }
}

void _AVX_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    float tmpfloat8[8];
    int count  = static_cast<int32_t>(size / 8);
    int remain = count * 8;
    // step 1: get sum
    float mean = 0;
    if(!RMSNorm){
        float sum = 0.f;
        if (count > 0) {
            auto sumVal = _mm256_set1_ps(0.f);
            for (int i = 0; i < count; i++) {
                sumVal = _mm256_add_ps(sumVal, _mm256_loadu_ps(src + i * 8));
            }
            _mm256_storeu_ps(tmpfloat8, sumVal);
            for (int i = 0; i < 8; i++) {
                sum += tmpfloat8[i];
            }
        }
        for (int i = remain; i < size; i++) {
            sum += src[i];
        }
        mean = sum / size;
    }
    // step 2: get square_sum
    float square_sum = 0.f;
    auto meanVal = _mm256_set1_ps(mean);
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            sumVal = _mm256_add_ps(sumVal, _mm256_mul_ps(x, x));
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            square_sum += tmpfloat8[i];
        }
    }
    for (int i = remain; i < size; i++) {
        float x = (src[i] - mean);
        square_sum += x * x;
    }
    // step 3: get result
    float variable = square_sum / size;
    variable = 1.f / sqrt(variable + epsilon);
    auto variableVal = _mm256_set1_ps(variable);
    if (gamma && beta) {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto g = _mm256_loadu_ps(gamma + i * 8);
            auto b = _mm256_loadu_ps(beta + i * 8);
            auto y = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(x, g), variableVal), b);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * gamma[i] * variable + beta[i] ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto y = _mm256_mul_ps(x, variableVal);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (int i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * variable;
        }
    }
}