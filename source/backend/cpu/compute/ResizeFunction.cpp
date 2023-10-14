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
using Vec4 = Vec<float, 4>;
using Vec16 = Vec<float, 16>;
using Vec8 = Vec<float, 8>;
// F = -0.5
static Vec4 CubicInterpolation(Vec4& A, Vec4& B, Vec4& C, Vec4& D, float t) {
    Vec4 a = (B - C) + (B - A) * 0.5f  + (D - C) * 0.5f;
    Vec4 b = C - ((B - A) + (B - C)) - (B + D) * 0.5f;

    Vec4 c = (C - A) * 0.5f;
    Vec4 d = B;
    return ((a * t + b) * t + c) * t + d;
}

// F = -0.75
template<typename T, int pack>
static Vec<T, pack> CubicInterpolation2(Vec<T, pack>& A, Vec<T, pack>& B, Vec<T, pack>& C, Vec<T, pack>& D, float t) {
    float b0 = 1.0f - 2.25f * t * t + 1.25f * t * t * t;
    float c0 = 1.0f - 2.25f * (1.0f - t) * (1.0f - t) + 1.25 * (1.0f - t) * (1.0f - t) * (1.0f - t);
    auto t_a = 1.0f + t;
    auto t_d = 2.0f - t;
    auto a0 = 3.0f - 6.0f * (t_a) + 5.0f * 0.75 * t_a * t_a - 0.75f * t_a * t_a * t_a;
    auto d0 = 3.0f - 6.0f * (t_d) + 5.0f * 0.75 * t_d * t_d - 0.75f * t_d * t_d * t_d;
    
    return A * a0 + B * b0 + C * c0 + D * d0;
}

void CPUBilinearSampleC4(const float* src, float* dst, const int32_t* position, const float* factor, int8_t* zeroPoint,
                                size_t number) {
    int pack = 4;
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        Vec4 df(f);
        Vec4 sf(1.0f - f);
        Vec4 A = Vec4::load(src + position[2 * i] * pack);
        Vec4 B = Vec4::load(src + position[2 * i + 1] * pack);
        Vec4 Result = B * df + A * sf;
        Vec4::save(dst + pack * i, B * df + A * sf);
    }
}

void CPUBilinearLineC4(float* dst, const float* A, const float* B, const float* t, int8_t* zeroPoint, size_t number) {
    int pack = 4;
    Vec4 df(*t);
    Vec4 sf(1.0f - *t);
    for (int i = 0; i < number; ++i) {
        Vec4 value = Vec4::load(A + pack * i) * sf + Vec4::load(B + pack * i) * df;
        Vec4::save(dst + pack * i, value);
    }
}

void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        auto A        = Vec4::load(src + 4 * position[4 * i + 0]);
        auto B        = Vec4::load(src + 4 * position[4 * i + 1]);
        auto C        = Vec4::load(src + 4 * position[4 * i + 2]);
        auto D        = Vec4::load(src + 4 * position[4 * i + 3]);
        Vec4::save(dst + 4 * i, CubicInterpolation2(A, B, C, D, f));
    }
}

void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
                    size_t number, ssize_t minValue, ssize_t maxValue) {
    float f = *t;
    for (int i = 0; i < number; ++i) {
        auto a = Vec4::load(A + 4 * i);
        auto b = Vec4::load(B + 4 * i);
        auto c = Vec4::load(C + 4 * i);
        auto d = Vec4::load(D + 4 * i);
        Vec4::save(dst + 4 * i, CubicInterpolation2<float, 4>(a, b, c, d, f));
    }
}

#ifndef MNN_USE_NEON
void MNNCubicSampleC16(const int8_t* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number) {
    int pack = 16;
    using  Vec16 = Vec<float, 16>;
#ifdef MNN_USE_SSE
    Vec16 zeroPointV(128 + (*zeroPoint));
    const uint8_t* srcPtr = (uint8_t*)src;
#else
    Vec16 zeroPointV(*zeroPoint);
    const int8_t* srcPtr = src;
#endif
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        auto A        = Vec16::load(srcPtr + pack * position[4 * i + 0]) - zeroPointV;
        auto B        = Vec16::load(srcPtr + pack * position[4 * i + 1]) - zeroPointV;
        auto C        = Vec16::load(srcPtr + pack * position[4 * i + 2]) - zeroPointV;
        auto D        = Vec16::load(srcPtr + pack * position[4 * i + 3]) - zeroPointV;
        auto val16 = CubicInterpolation2<float, 16>(A, B, C, D, f);
        Vec16::save(dst + pack * i, CubicInterpolation2<float, 16>(A, B, C, D, f));
    }
}

void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
                    size_t number, ssize_t minValue, ssize_t maxValue) {
    int pack = 16;
    using  Vec16 = Vec<float, 16>;
#ifdef MNN_USE_SSE
    uint8_t* dstPtr = (uint8_t*)dst;
    int offset = 128 + (*zeroPoint);
    int minVal = 128 + minValue;
    int maxVal = 128 + maxValue;
#else
    int8_t* dstPtr = dst;
    int offset = *zeroPoint;
    int minVal = (int)minValue;
    int maxVal = (int)maxValue;
#endif
    float f = *t;
    for (int i = 0; i < number; ++i) {
        auto a = Vec16::load(A + pack * i);
        auto b = Vec16::load(B + pack * i);
        auto c = Vec16::load(C + pack * i);
        auto d = Vec16::load(D + pack * i);
        auto val16 = CubicInterpolation2<float, 16>(a, b, c, d, f);
        for (int j = 0; j < pack; ++j) {
            int val = (int)roundf(val16[j]) + offset;
            if (val > maxVal) {
                val = maxVal;
            }
            if (val < minVal) {
                val = minVal;
            }
            *(dstPtr + pack * i + j) = val;
        }
    }
}

void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, const int32_t* position, const float* factor, int8_t* zeroPoint,
                                size_t number) {
#ifdef MNN_USE_SSE
    int offset = 128 + *zeroPoint;
    const uint8_t* srcPtr = (uint8_t*)src;
#else
    int offset = *zeroPoint;
    const int8_t* srcPtr = src;
#endif
    int pack = 8;
    for (int i = 0; i < number; ++i) {
        int16_t df = factor[i] * 128;
        int16_t sf = (1 - factor[i]) * 128;
        auto aPtr = srcPtr + position[2 * i] * pack;
        auto bPtr = srcPtr + position[2 * i + 1] * pack;
        for (int j = 0; j < pack; ++j) {
            int a = static_cast<int32_t>(*(aPtr + j) - offset);
            int b = static_cast<int32_t>(*(bPtr + j) - offset);
            int16_t val = static_cast<int16_t>(a * sf + b * df);
            *(dst + pack * i + j) = val;
        }
    }
}

void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, const float* t, int8_t* zeroPoint, size_t number) {
#ifdef MNN_USE_SSE
    int offset = 128 + (*zeroPoint);
    uint8_t* dstPtr = (uint8_t*)dst;
#else
    int offset = *zeroPoint;
    int8_t* dstPtr = dst;
#endif
    int pack = 8;
    int16_t df = (*t) * 128;
    int16_t sf = (1 - *t) * 128;
    for (int i = 0; i < number; ++i) {
        auto aPtr = A + pack * i;
        auto bPtr = B + pack * i;
        for (int j = 0; j < pack; ++j) {
            int32_t val = *(aPtr + j) * sf + *(bPtr + j) * df;
            int8_t valOut = (val + (1<<13)) / (1 << 14);
            if (val < 0) {
                valOut = (val - (1 << 13)) / (1 << 14);
            }
            *(dstPtr+ pack * i + j) = valOut+ offset;
        }
    }
}

#endif
