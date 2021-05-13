//
//  WinogradOptFunctionHalf.cpp
//  MNN
//
//  Created by MNN on 2021/03/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "WinogradOptFunctionHalf.hpp"
#include <cstring>
#include <memory>
#include "core/Macro.h"
#include "VecHalf.hpp"
using BFVec4 = MNN::Math::VecHalf<4>;

namespace MNN {
static void _sourceTransformUnit4x4(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
    BFVec4::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit4x2(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
}


#define LOAD6                                     \
BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep); \
BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep); \
BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep); \
BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep); \
BFVec4 s4 = BFVec4::load(srcBlock + 4 * srcStep); \
BFVec4 s5 = BFVec4::load(srcBlock + 5 * srcStep);

static void _sourceTransformUnit6x6(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;
    BFVec4 m0 = s0 * 4.f - s2 * 5.f + s4;

    BFVec4 m1 = (s1 + s2) * (-4.f) + (s3 + s4);
    BFVec4 m2 = (s1 - s2) * (4.f) + (s4 - s3);

    BFVec4 m3 = s1 * -2.f - s2 + s3 * 2.f + s4;
    BFVec4 m4 = s1 * 2.f - s2 - s3 * 2.f + s4;

    BFVec4 m5 = s1 * 4.f - s3 * 5.f + s5;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
    BFVec4::save(dstStart + 3 * dstStep, m3);
    BFVec4::save(dstStart + 4 * dstStep, m4);
    BFVec4::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit6x5(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);
    BFVec4 s4 = BFVec4::load(srcBlock + 4 * srcStep);
    BFVec4 s5 = BFVec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
    BFVec4::save(dstStart + 3 * dstStep, m3);
    BFVec4::save(dstStart + 4 * dstStep, m4);
}
static void _destTransformUnit6x4(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);
    BFVec4 s4 = BFVec4::load(srcBlock + 4 * srcStep);
    BFVec4 s5 = BFVec4::load(srcBlock + 5 * srcStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
    BFVec4::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit6x3(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);
    BFVec4 s4 = BFVec4::load(srcBlock + 4 * srcStep);
    BFVec4 s5 = BFVec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
    BFVec4::save(dstStart + 2 * dstStep, m2);
}
static void _destTransformUnit6x2(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep) {
    BFVec4 s0 = BFVec4::load(srcBlock + 0 * srcStep);
    BFVec4 s1 = BFVec4::load(srcBlock + 1 * srcStep);
    BFVec4 s2 = BFVec4::load(srcBlock + 2 * srcStep);
    BFVec4 s3 = BFVec4::load(srcBlock + 3 * srcStep);
    BFVec4 s4 = BFVec4::load(srcBlock + 4 * srcStep);
    BFVec4 s5 = BFVec4::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;

    BFVec4::save(dstStart + 0 * dstStep, m0);
    BFVec4::save(dstStart + 1 * dstStep, m1);
}

static WinogradFunctionHalf::TransformFunc gProcUnit6[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit6x2,
    _destTransformUnit6x3,
    _destTransformUnit6x4,
    _destTransformUnit6x5,
};


WinogradFunctionHalf::TransformFunc WinogradFunctionHalf::chooseSourceTransform(int k, int w) {
    if (6 == k && 6 == w) {
        return _sourceTransformUnit6x6;
    }
    if (4 == k && 4 == w) {
        return _sourceTransformUnit4x4;
    }
    MNN_ASSERT(false);
    return nullptr;
}

WinogradFunctionHalf::TransformFunc WinogradFunctionHalf::chooseDestTransform(int k, int h) {
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
