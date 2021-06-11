//
//  ResizeFunction.cpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ResizeFunction.h"
#include <math.h>
#include "core/AutoStorage.h"
#include "core/Macro.h"
#include "math/Vec.hpp"

using namespace MNN::Math;
using Vec4 = MNN::Math::Vec<float, 4>;
// F = -0.5
static Vec4 CubicInterpolation(Vec4& A, Vec4& B, Vec4& C, Vec4& D, float t) {
    Vec4 a = (B - C) + (B - A) * 0.5f  + (D - C) * 0.5f;
    Vec4 b = C - ((B - A) + (B - C)) - (B + D) * 0.5f;

    Vec4 c = (C - A) * 0.5f;
    Vec4 d = B;
    return ((a * t + b) * t + c) * t + d;
}

// F = -0.75
static Vec4 CubicInterpolation2(Vec4& A, Vec4& B, Vec4& C, Vec4& D, float t) {
    float b0 = 1.0f - 2.25f * t * t + 1.25f * t * t * t;
    float c0 = 1.0f - 2.25f * (1.0f - t) * (1.0f - t) + 1.25 * (1.0f - t) * (1.0f - t) * (1.0f - t);
    auto t_a = 1.0f + t;
    auto t_d = 2.0f - t;
    auto a0 = 3.0f - 6.0f * (t_a) + 5.0f * 0.75 * t_a * t_a - 0.75f * t_a * t_a * t_a;
    auto d0 = 3.0f - 6.0f * (t_d) + 5.0f * 0.75 * t_d * t_d - 0.75f * t_d * t_d * t_d;
    
    return A * a0 + B * b0 + C * c0 + D * d0;
}

void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        auto A        = Vec4::load(src + 4 * position[4 * i + 0]);
        auto B        = Vec4::load(src + 4 * position[4 * i + 1]);
        auto C        = Vec4::load(src + 4 * position[4 * i + 2]);
        auto D        = Vec4::load(src + 4 * position[4 * i + 3]);
        Vec4::save(dst + 4 * i, CubicInterpolation2(A, B, C, D, f));
    }
}

void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t,
                    size_t number) {
    float f = *t;
    for (int i = 0; i < number; ++i) {
        auto a = Vec4::load(A + 4 * i);
        auto b = Vec4::load(B + 4 * i);
        auto c = Vec4::load(C + 4 * i);
        auto d = Vec4::load(D + 4 * i);
        Vec4::save(dst + 4 * i, CubicInterpolation2(a, b, c, d, f));
    }
}
