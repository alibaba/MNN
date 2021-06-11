//
//  WinogradAVX2.cpp
//  MNN
//
//  Created by MNN on b'2021/05/16'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Vec8.hpp"
#include "FunctionSummary.hpp"
namespace MNN {
static void _sourceTransformUnit4x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);

    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit4x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
}

#define LOAD8                                     \
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep); \
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep); \
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep); \
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep); \
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep); \
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep); \
    Vec8 s6 = Vec8::load(srcBlock + 6 * srcStep); \
    Vec8 s7 = Vec8::load(srcBlock + 7 * srcStep);

static void _sourceTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    Vec8 m0 = s0 * 36.f - s2 * 49.f + s4 * 14.f - s6;

    Vec8 m1 = (s1 + s2) * 36.f - (s3 + s4) * 13.f + (s5 + s6);
    Vec8 m2 = (s2 - s1) * 36.f + (s3 - s4) * 13.f + (s6 - s5);

    Vec8 m3 = s1 * 18.f + s2 * 9.f - s3 * 20.f - s4 * 10.f + s5 * 2.f + s6;
    Vec8 m4 = s2 * 9.f - s1 * 18.f + s3 * 20.f - s4 * 10.f - s5 * 2.f + s6;

    Vec8 m5 = s1 * 12.f + s2 * 4.f - s3 * 15.f - s4 * 5.f + s5 * 3.f + s6;
    Vec8 m6 = s2 * 4.f - s1 * 12.f + s3 * 15.f - s4 * 5.f - s5 * 3.f + s6;

    Vec8 m7 = s3 * 49.f - s1 * 36.f - s5 * 14.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
    Vec8::save(dstStart + 5 * dstStep, m5);
    Vec8::save(dstStart + 6 * dstStep, m6);
    Vec8::save(dstStart + 7 * dstStep, m7);
}

static void _destTransformUnit8x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
}

static void _destTransformUnit8x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
}

static void _destTransformUnit8x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
}

static void _destTransformUnit8x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
}

static void _destTransformUnit8x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);
    Vec8 s6 = Vec8::load(srcBlock + 6 * srcStep);
    Vec8 s7 = Vec8::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
    Vec8::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit8x7(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);
    Vec8 s6 = Vec8::load(srcBlock + 6 * srcStep);
    Vec8 s7 = Vec8::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f;
    auto m6 = (s1 + s2) + (s3 + s4) * 64.f + (s5 + s6) * 729.f + s7;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
    Vec8::save(dstStart + 5 * dstStep, m5);
    Vec8::save(dstStart + 6 * dstStep, m6);
}

static CoreFunctions::WinoTransFunc gProcUnit8[] = {
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
Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep); \
Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep); \
Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep); \
Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep); \
Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep); \
Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);

static void _sourceTransformUnit6x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;
    Vec8 m0 = s0 * 4.f - s2 * 5.f + s4;

    Vec8 m1 = (s1 + s2) * (-4.f) + (s3 + s4);
    Vec8 m2 = (s1 - s2) * (4.f) + (s4 - s3);

    Vec8 m3 = s1 * -2.f - s2 + s3 * 2.f + s4;
    Vec8 m4 = s1 * 2.f - s2 - s3 * 2.f + s4;

    Vec8 m5 = s1 * 4.f - s3 * 5.f + s5;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
    Vec8::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit6x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
    Vec8::save(dstStart + 4 * dstStep, m4);
}
static void _destTransformUnit6x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
    Vec8::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit6x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
    Vec8::save(dstStart + 2 * dstStep, m2);
}
static void _destTransformUnit6x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec8 s0 = Vec8::load(srcBlock + 0 * srcStep);
    Vec8 s1 = Vec8::load(srcBlock + 1 * srcStep);
    Vec8 s2 = Vec8::load(srcBlock + 2 * srcStep);
    Vec8 s3 = Vec8::load(srcBlock + 3 * srcStep);
    Vec8 s4 = Vec8::load(srcBlock + 4 * srcStep);
    Vec8 s5 = Vec8::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;

    Vec8::save(dstStart + 0 * dstStep, m0);
    Vec8::save(dstStart + 1 * dstStep, m1);
}

static CoreFunctions::WinoTransFunc gProcUnit6[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit6x2,
    _destTransformUnit6x3,
    _destTransformUnit6x4,
    _destTransformUnit6x5,
};


static CoreFunctions::WinoTransFunc _AVX2_chooseSourceTransform(int k, int w) {
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

static CoreFunctions::WinoTransFunc _AVX2_chooseDestTransform(int k, int h) {
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
};
void _AVX_WinogradInit(void* functions) {
    auto core = reinterpret_cast<MNN::CoreFunctions*>(functions);
    core->chooseWinoDestTransform = MNN::_AVX2_chooseDestTransform;
    core->chooseWinoSourceTransform = MNN::_AVX2_chooseSourceTransform;
}
