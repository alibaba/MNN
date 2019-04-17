//
//  ResizeFunction.cpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ResizeFunction.h"
#include <math.h>
#include "AutoStorage.h"
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#ifndef MNN_USE_NEON
static float CubicInterpolation(float A, float B, float C, float D, float t) {
    float a = (B - C) + 0.5f * (B - A) + (D - C) * 0.5f;
    float b = C - ((B - A) + (B - C)) - (B + D) * 0.5f;

    float c = 0.5f * (C - A);
    float d = B;
    return ((a * t + b) * t + c) * t + d;
}

void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
        for (int k = 0; k < 4; ++k) {
            float A        = src[4 * position[4 * i + 0] + k];
            float B        = src[4 * position[4 * i + 1] + k];
            float C        = src[4 * position[4 * i + 2] + k];
            float D        = src[4 * position[4 * i + 3] + k];
            dst[4 * i + k] = CubicInterpolation(A, B, C, D, f);
        }
    }
}

void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t,
                    size_t number) {
    float f = *t;
    for (int i = 0; i < number; ++i) {
        for (int j = 0; j < 4; ++j) {
            int k  = i * 4 + j;
            dst[k] = CubicInterpolation(A[k], B[k], C[k], D[k], f);
        }
    }
}

#endif // MNN_USE_NEON

void MNNBilinearSampleC4(const float* src, float* dst, int32_t* position, const float* factor, size_t number) {
    for (int i = 0; i < number; ++i) {
        float f = factor[i];
#ifdef MNN_USE_NEON
        float32x4_t df = vdupq_n_f32(f);
        float32x4_t sf = vdupq_n_f32(1.0f - f);
        float32x4_t A  = vld1q_f32(src + position[2 * i] * 4);
        float32x4_t B  = vld1q_f32(src + position[2 * i + 1] * 4);
        vst1q_f32(dst + 4 * i, B * df + A * sf);
#else
        for (int k = 0; k < 4; ++k) {
            float A        = src[4 * position[2 * i + 0] + k];
            float B        = src[4 * position[2 * i + 1] + k];
            dst[4 * i + k] = B * f + A * (1 - f);
        }
#endif
    }
}

void MNNBilinearLineC4(float* dst, const float* A, const float* B, float* t, size_t number) {
#ifdef MNN_USE_NEON
    float32x4_t df = vdupq_n_f32(*t);
    float32x4_t sf = vdupq_n_f32(1.0f) - df;
    for (int i = 0; i < number; ++i) {
        float32x4_t value = vld1q_f32(A + 4 * i) * sf + vld1q_f32(B + 4 * i) * df;
        vst1q_f32(dst + 4 * i, value);
    }
#else
    float f = *t;
    for (int i = 0; i < number; ++i) {
        for (int j = 0; j < 4; ++j) {
            int k = i * 4 + j;
            dst[k] = A[k] * (1 - f) + B[k] * f;
        }
    }
#endif
}
