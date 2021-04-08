//
//  Arm82WinogradOptFunc.cpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82WinogradOptFunc.hpp"
#include "Arm82Vec.hpp"
#include "Arm82OptFunc.hpp"
#include <cstring>
#include <memory>
#include "core/Macro.h"
#include "math/Vec.hpp"
using Vec = MNN::Math::Vec<FLOAT16, 8>;

namespace MNN {

static void _sourceTransformUnit4x4(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);

    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
    Vec::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit4x2(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
}

#define LOAD6                                     \
Vec s0 = Vec::load(srcBlock + 0 * srcStep); \
Vec s1 = Vec::load(srcBlock + 1 * srcStep); \
Vec s2 = Vec::load(srcBlock + 2 * srcStep); \
Vec s3 = Vec::load(srcBlock + 3 * srcStep); \
Vec s4 = Vec::load(srcBlock + 4 * srcStep); \
Vec s5 = Vec::load(srcBlock + 5 * srcStep);

static void _sourceTransformUnit6x6(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;
    Vec m0 = s0 * (FLOAT16)4 - s2 * (FLOAT16)5 + s4;

    Vec m1 = (s1 + s2) * (-(FLOAT16)4) + (s3 + s4);
    Vec m2 = (s1 - s2) * ((FLOAT16)4) + (s4 - s3);

    Vec m3 = s1 * -(FLOAT16)2 - s2 + s3 * (FLOAT16)2 + s4;
    Vec m4 = s1 * (FLOAT16)2 - s2 - s3 * (FLOAT16)2 + s4;

    Vec m5 = s1 * (FLOAT16)4 - s3 * (FLOAT16)5 + s5;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
    Vec::save(dstStart + 3 * dstStep, m3);
    Vec::save(dstStart + 4 * dstStep, m4);
    Vec::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit6x5(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);
    Vec s4 = Vec::load(srcBlock + 4 * srcStep);
    Vec s5 = Vec::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * (FLOAT16)2;
    auto m2 = (s1 + s2) + (s3 + s4) * (FLOAT16)4;
    auto m3 = (s1 - s2) + (s3 - s4) * (FLOAT16)8;
    auto m4 = (s1 + s2) + (s3 + s4) * (FLOAT16)16 + s5;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
    Vec::save(dstStart + 3 * dstStep, m3);
    Vec::save(dstStart + 4 * dstStep, m4);
}
static void _destTransformUnit6x4(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);
    Vec s4 = Vec::load(srcBlock + 4 * srcStep);
    Vec s5 = Vec::load(srcBlock + 5 * srcStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * (FLOAT16)4;
    auto m3 = v3 + v1 * (FLOAT16)8 + s5;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
    Vec::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit6x3(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);
    Vec s4 = Vec::load(srcBlock + 4 * srcStep);
    Vec s5 = Vec::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * (FLOAT16)2;
    auto m2 = (s1 + s2) + (s3 + s4) * (FLOAT16)4 + s5;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
    Vec::save(dstStart + 2 * dstStep, m2);
}
static void _destTransformUnit6x2(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    Vec s0 = Vec::load(srcBlock + 0 * srcStep);
    Vec s1 = Vec::load(srcBlock + 1 * srcStep);
    Vec s2 = Vec::load(srcBlock + 2 * srcStep);
    Vec s3 = Vec::load(srcBlock + 3 * srcStep);
    Vec s4 = Vec::load(srcBlock + 4 * srcStep);
    Vec s5 = Vec::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * (FLOAT16)2 + s5;

    Vec::save(dstStart + 0 * dstStep, m0);
    Vec::save(dstStart + 1 * dstStep, m1);
}

static Arm82WinogradFunction::TransformFunc gProcUnit6[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit6x2,
    _destTransformUnit6x3,
    _destTransformUnit6x4,
    _destTransformUnit6x5,
};


Arm82WinogradFunction::TransformFunc Arm82WinogradFunction::chooseSourceTransform(int k, int w) {
    if (6 == k && 6 == w) {
        return _sourceTransformUnit6x6;
    }
    if (4 == k && 4 == w) {
        return _sourceTransformUnit4x4;
    }
    MNN_ASSERT(false);
    return nullptr;
}

Arm82WinogradFunction::TransformFunc Arm82WinogradFunction::chooseDestTransform(int k, int h) {
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

int Arm82MNNGetConvTileNumber() {
    int eP, lP, hP;
    Arm82MNNGetMatMulPackMode(&eP, &lP, &hP);
    return eP; // 8
}

} // namespace MNN
#endif
