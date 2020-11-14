//
//  WinogradOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/WinogradOptFunction.hpp"
#include <cstring>
#include <memory>
#include "core/Macro.h"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

#define DEFAULT_UNIT 8
extern "C" {
void MNNWinogradMatrixProductLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                  size_t length);
void MNNWinogradMatrixProductRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                   size_t length);
}

#ifndef MNN_USE_NEON

// M = BT * S , M = w*h * l, S = w*k * l, B = h*k
void MNNWinogradMatrixProductLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                  size_t length) {
    auto unitStep = 4 * length;
    for (int y = 0; y < h; ++y) {
        auto dstY = M + y * w * unitStep;
        for (int x = 0; x < w; ++x) {
            auto dstX = dstY + x * unitStep;
            auto srcX = S + x * unitStep;
            ::memset(dstX, 0, unitStep * sizeof(float));
            for (int i = 0; i < k; ++i) {
                auto b    = B[i * h + y];
                auto srcY = srcX + i * w * unitStep;
                if (0.0f == b) {
                    continue;
                }
                for (int j = 0; j < unitStep; ++j) {
                    dstX[j] += srcY[j] * b;
                }
            }
        }
    }
}

// M = S * B , M = w*h * l, S = k*h * l, B = w*k
void MNNWinogradMatrixProductRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                   size_t length) {
    auto unitStep = 4 * length;
    for (int y = 0; y < h; ++y) {
        auto dstY = M + y * w * unitStep;
        auto srcY = S + y * k * unitStep;

        for (int x = 0; x < w; ++x) {
            auto dstX = dstY + x * unitStep;
            ::memset(dstX, 0, unitStep * sizeof(float));
            for (int i = 0; i < k; ++i) {
                auto srcX = srcY + i * unitStep;
                auto b    = B[i * h + x];
                if (0.0f == b) {
                    continue;
                }
                for (int j = 0; j < unitStep; ++j) {
                    dstX[j] += srcX[j] * b;
                }
            }
        }
    }
}
#endif

namespace MNN {
void WinogradFunction::productLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                   size_t length) {
    MNNWinogradMatrixProductLeft(S, B, M, w, h, k, length);
}

void WinogradFunction::productRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k,
                                    size_t length) {
    MNNWinogradMatrixProductRight(S, B, M, w, h, k, length);
}
int WinogradFunction::getPreferNumber() {
    return DEFAULT_UNIT;
}

static void _sourceTransformUnit4x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit4x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
}

#define LOAD8                                     \
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep); \
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep); \
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep); \
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep); \
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep); \
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep); \
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep); \
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

static void _sourceTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    Vec4 m0 = s0 * 36.f - s2 * 49.f + s4 * 14.f - s6;

    Vec4 m1 = (s1 + s2) * 36.f - (s3 + s4) * 13.f + (s5 + s6);
    Vec4 m2 = (s2 - s1) * 36.f + (s3 - s4) * 13.f + (s6 - s5);

    Vec4 m3 = s1 * 18.f + s2 * 9.f - s3 * 20.f - s4 * 10.f + s5 * 2.f + s6;
    Vec4 m4 = s2 * 9.f - s1 * 18.f + s3 * 20.f - s4 * 10.f - s5 * 2.f + s6;

    Vec4 m5 = s1 * 12.f + s2 * 4.f - s3 * 15.f - s4 * 5.f + s5 * 3.f + s6;
    Vec4 m6 = s2 * 4.f - s1 * 12.f + s3 * 15.f - s4 * 5.f - s5 * 3.f + s6;

    Vec4 m7 = s3 * 49.f - s1 * 36.f - s5 * 14.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
    Vec4::save(dstStart + 6 * dstStep, m6);
    Vec4::save(dstStart + 7 * dstStep, m7);
}

static void _destTransformUnit8x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
}

static void _destTransformUnit8x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
}

static void _destTransformUnit8x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
}

static void _destTransformUnit8x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
}

static void _destTransformUnit8x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit8x7(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f;
    auto m6 = (s1 + s2) + (s3 + s4) * 64.f + (s5 + s6) * 729.f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
    Vec4::save(dstStart + 6 * dstStep, m6);
}

static WinogradFunction::TransformFunc gProcUnit8[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit8x2,
    _destTransformUnit8x3,
    _destTransformUnit8x4,
    _destTransformUnit8x5,
    _destTransformUnit8x6,
    _destTransformUnit8x7,
};


#define LOAD6                                     \
Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep); \
Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep); \
Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep); \
Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep); \
Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep); \
Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);

static void _sourceTransformUnit6x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;
    Vec4 m0 = s0 * 4.f - s2 * 5.f + s4;

    Vec4 m1 = (s1 + s2) * (-4.f) + (s3 + s4);
    Vec4 m2 = (s1 - s2) * (4.f) + (s4 - s3);

    Vec4 m3 = s1 * -2.f - s2 + s3 * 2.f + s4;
    Vec4 m4 = s1 * 2.f - s2 - s3 * 2.f + s4;

    Vec4 m5 = s1 * 4.f - s3 * 5.f + s5;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit6x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
}
static void _destTransformUnit6x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit6x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
}
static void _destTransformUnit6x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
}

static WinogradFunction::TransformFunc gProcUnit6[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit6x2,
    _destTransformUnit6x3,
    _destTransformUnit6x4,
    _destTransformUnit6x5,
};


WinogradFunction::TransformFunc WinogradFunction::chooseSourceTransform(int k, int w) {
    if (8 == k && 8 == w) {
        return _sourceTransformUnit8x8;
    }
    if (6 == k && 6 == w) {
        return _sourceTransformUnit6x6;
    }
    if (4 == k && 4 == w) {
        return _sourceTransformUnit4x4;
    }
    MNN_ASSERT(false);
    return nullptr;
}

WinogradFunction::TransformFunc WinogradFunction::chooseDestTransform(int k, int h) {
    if (8 == k) {
        if (h <= 1 || h > 7) {
            return nullptr;
        }
        return gProcUnit8[h];
    }
    if (6 == k) {
        if (h <= 1 || h > 5) {
            return nullptr;
        }
        return gProcUnit6[h];
    }
    if (2 == h && 4 == k) {
        return _destTransformUnit4x2;
    }
    if (3 == h && 4 == k) {
        return _destTransformUnit4x3;
    }
    return nullptr;
}

} // namespace MNN
