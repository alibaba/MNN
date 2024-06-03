//
//  MathFunctions.cpp
//  MNN
//
//  Created by MNN on b'2021/07/05'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include <math.h>

void _AVX_MNNGeluFMA(float *dst, const float *src, size_t size, float* parameters) {
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
        y = _mm256_fmadd_ps(y, var1, x);
        y = _mm256_mul_ps(y, var2);
        y = _mm256_max_ps(y, clamp_min);
        y = _mm256_min_ps(y, clamp_max);
        // y = tanh(y)
        {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_fmadd_ps(w, y2, var4);
            w = _mm256_fmadd_ps(w, y2, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_fmadd_ps(z, y2, var8);
            z = _mm256_fmadd_ps(z, y2, var9);
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

void _AVX_MNNExpC8FMA(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8;
    auto A     = _mm256_broadcast_ss(offset + 0);
    auto B     = _mm256_broadcast_ss(offset + 1);
    auto C     = _mm256_broadcast_ss(offset + 2);
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
        auto t         = xReamin;
        auto c1        = _mm256_fmadd_ps(p7, t, p6);
        auto c3        = _mm256_fmadd_ps(c1, t, p5);
        auto c5        = _mm256_fmadd_ps(c3, t, p4);
        auto c7        = _mm256_fmadd_ps(c5, t, p3);
        auto c9        = _mm256_fmadd_ps(c7, t, p2);
        auto expRemain = c9;
        auto res = _mm256_fmadd_ps(expBasic, expRemain, B);
        summer = _mm256_add_ps(summer, res);
        _mm256_storeu_ps(dest + 8 * i, res);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, summer);
    float total = offset[3];
    for (int i=0; i<8; ++i) {
        total+=tmp[i];
    }
    offset[3] = total;
}
