//
//  WinogradOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "WinogradOptFunction.hpp"
#include <cstring>
#include <memory>
#include "Macro.h"
#include "Vec4.hpp"
using namespace MNN::Math;

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

    auto m0 = s0 - s2 * 4.f;
    auto m1 = s1 + s2 * 2.f;
    auto m2 = s2 * 2.f - s1;
    auto m3 = s3 - s1 * 0.25f;

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
    auto m1 = (s1 - s2) * 0.5f + s3;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) * 0.5f;
    auto m2 = (s1 + s2) * 0.25f + s3;

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
    Vec4 m0 = s0 - s2 * 5.4444446563720703f + s4 * 6.2222223281860352f - s6 * 1.7777777910232544f;

    Vec4 m1 = s1 * 1.5000000f + s2 * 3.0000000f - s3 * 2.1666667461395264f - s4 * 4.3333334922790527f +
              s5 * 0.6666666865348816f + s6 * 1.3333333730697632f;
    Vec4 m2 = s2 * 3.0000000f - s1 * 1.5000000f + s3 * 2.1666667461395264f - s4 * 4.3333334922790527f -
              s5 * 0.6666666865348816f + s6 * 1.3333333730697632f;

    Vec4 m3 = (s3 + s4) * 1.3333333730697632f - (s1 + s2) * 0.3000000f - (s5 + s6) * 0.5333333611488342f;
    Vec4 m4 = (s4 - s3) * 1.3333333730697632f + (s1 - s2) * 0.3000000f + (s5 - s6) * 0.5333333611488342f;

    Vec4 m5 = s1 * 0.0333333350718021f + s2 * 0.0222222227603197f - s3 * 0.1666666716337204f -
              s4 * 0.1111111119389534f + s5 * 0.1333333402872086f + s6 * 0.0888888910412788f;
    Vec4 m6 = s2 * 0.0222222227603197f - s1 * 0.0333333350718021f + s3 * 0.1666666716337204f -
              s4 * 0.1111111119389534f - s5 * 0.1333333402872086f + s6 * 0.0888888910412788f;

    Vec4 m7 = s3 * 3.0625000f - s1 * 0.5625000f - s5 * 3.5f + s7;

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
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
}

static void _destTransformUnit8x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f;
    auto m2 = (s1 + s2) * 0.25f + s3 + s4 + (s5 + s6) * 2.25f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
}

static void _destTransformUnit8x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f;
    auto m2 = (s1 + s2) * 0.25f + s3 + s4 + (s5 + s6) * 2.25f;
    auto m3 = (s1 - s2) * 0.125f + (s3 - s4) + (s5 - s6) * 3.375f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
}

static void _destTransformUnit8x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f;
    auto m2 = (s1 + s2) * 0.25f + s3 + s4 + (s5 + s6) * 2.25f;
    auto m3 = (s1 - s2) * 0.125f + (s3 - s4) + (s5 - s6) * 3.375f;
    auto m4 = (s1 + s2) * 0.0625f + (s3 + s4) + (s5 + s6) * 5.0625f + s7;

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
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f;
    auto m2 = (s1 + s2) * 0.25f + s3 + s4 + (s5 + s6) * 2.25f;
    auto m3 = (s1 - s2) * 0.125f + (s3 - s4) + (s5 - s6) * 3.375f;
    auto m4 = (s1 + s2) * 0.0625f + (s3 + s4) + (s5 + s6) * 5.0625f;
    auto m5 = (s1 - s2) * 0.03125f + (s3 - s4) + (s5 - s6) * 7.59375f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
}

static WinogradFunction::TransformFunc gProcUnit8[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit8x2,
    _destTransformUnit8x3,
    _destTransformUnit8x4,
    _destTransformUnit8x5,
    _destTransformUnit8x6,
    nullptr, // 7
};
WinogradFunction::TransformFunc WinogradFunction::chooseSourceTransform(int k, int w) {
    if (8 == k && 8 == w) {
        return _sourceTransformUnit8x8;
    }
    if (4 == k && 4 == w) {
        return _sourceTransformUnit4x4;
    }
    MNN_ASSERT(false);
    return nullptr;
}

WinogradFunction::TransformFunc WinogradFunction::chooseDestTransform(int k, int h) {
    if (8 == k) {
        if (h <= 1 || h >= 7) {
            return nullptr;
        }
        return gProcUnit8[h];
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
