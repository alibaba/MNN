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
#include <map>
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/MemoryFormater.h"

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

static void _sourceTransformUnit4x4Pack12(float* srcBlock, float* dstStart, size_t dstStep) {

    // register number: (srcUnit + 1) * EPack/4 = 15
    constexpr int Nh = 4; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // transpose 12x4 to 4x12
        // register number : EPack
        Vec4 s0 = Vec4::load(srcPtr + 0 * packCUnit);
        Vec4 s3 = Vec4::load(srcPtr + 1 * packCUnit);
        Vec4 s6 = Vec4::load(srcPtr + 2 * packCUnit);
        Vec4 s9 = Vec4::load(srcPtr + 3 * packCUnit);
        Vec4 s1 = Vec4::load(srcPtr + 4 * packCUnit);
        Vec4 s4 = Vec4::load(srcPtr + 5 * packCUnit);
        Vec4 s7 = Vec4::load(srcPtr + 6 * packCUnit);
        Vec4 s10 = Vec4::load(srcPtr + 7 * packCUnit);
        Vec4 s2 = Vec4::load(srcPtr + 8 * packCUnit);
        Vec4 s5 = Vec4::load(srcPtr + 9 * packCUnit);
        Vec4 s8 = Vec4::load(srcPtr + 10 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 11 * packCUnit);
        Vec4::transpose4(s0, s3, s6, s9);
        Vec4::transpose4(s1, s4, s7, s10);
        Vec4::transpose4(s2, s5, s8, s11);

        // to-optimize: interleave load and save in loop
        // deal with pack when packCUnit is 8
        Vec4::save(srcPtr + 0 * packCUnit, s0);
        Vec4::save(srcPtr + 1 * packCUnit, s1);
        Vec4::save(srcPtr + 2 * packCUnit, s2);
        Vec4::save(srcPtr + 3 * packCUnit, s3);
        Vec4::save(srcPtr + 4 * packCUnit, s4);
        Vec4::save(srcPtr + 5 * packCUnit, s5);
        Vec4::save(srcPtr + 6 * packCUnit, s6);
        Vec4::save(srcPtr + 7 * packCUnit, s7);
        Vec4::save(srcPtr + 8 * packCUnit, s8);
        Vec4::save(srcPtr + 9 * packCUnit, s9);
        Vec4::save(srcPtr + 10 * packCUnit, s10);
        Vec4::save(srcPtr + 11 * packCUnit, s11);
        srcPtr += loadTransposeStride;
    }

    // MNN_PRINT("winograd in BT*D*B, transpose, loadTransposeStride:%zu, dstStep:%zu\n", loadTransposeStride, dstStep);
    // formatMatrix((const float*)srcBlock, {Nh, static_cast<int>(packCUnit), ePack});

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        // source transform D * B. register number : srcUnit * (EPack/4 + 1)
        Vec4 s00 = Vec4::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        Vec4 s01 = Vec4::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        Vec4 s02 = Vec4::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        Vec4 s10 = Vec4::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        Vec4 s12 = Vec4::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        Vec4 s20 = Vec4::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        Vec4 s21 = Vec4::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        Vec4 s22 = Vec4::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        Vec4 s30 = Vec4::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        Vec4 s31 = Vec4::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        Vec4 s32 = Vec4::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        // dstStep =  ePack * pack * ic_4
        auto ep0 = s00 - s20;
        auto ep1 = s01 - s21;
        auto ep2 = s02 - s22;
        Vec4::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 + s20;
        ep1 = s11 + s21;
        ep2 = s12 + s22;
        Vec4::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 - s10;
        ep1 = s21 - s11;
        ep2 = s22 - s12;
        Vec4::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s30 - s10;
        ep1 = s31 - s11;
        ep2 = s32 - s12;
        Vec4::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        // MNN_PRINT("\nwinograd in BT*D*B, iNh:0-3, i4c:%d\n", i4c);
        // formatMatrix(dstPtr + 0 * dstStep , {ePack});
        // formatMatrix(dstPtr + 1 * dstStep , {ePack});
        // formatMatrix(dstPtr + 2 * dstStep , {ePack});
        // formatMatrix(dstPtr + 3 * dstStep , {ePack});

        srcPtr += ePack;
        dstPtr += ePack;
    }
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

    // LOAD8;
    // Vec4::save(dstStart + 0 * dstStep, s0);
    // Vec4::save(dstStart + 1 * dstStep, s1);
    // Vec4::save(dstStart + 2 * dstStep, s2);
    // Vec4::save(dstStart + 3 * dstStep, s3);
    // Vec4::save(dstStart + 4 * dstStep, s4);
    // Vec4::save(dstStart + 5 * dstStep, s5);
    // Vec4::save(dstStart + 6 * dstStep, s6);
    // Vec4::save(dstStart + 7 * dstStep, s7);
}

static void _sourceTransformUnit8x8Pack12(float* srcBlock, float* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/4 = 27
    // todo: impliment
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // transpose 12x4 to 4x12
        // register number : EPack
        Vec4 s0 = Vec4::load(srcPtr + 0 * packCUnit);
        Vec4 s3 = Vec4::load(srcPtr + 1 * packCUnit);
        Vec4 s6 = Vec4::load(srcPtr + 2 * packCUnit);
        Vec4 s9 = Vec4::load(srcPtr + 3 * packCUnit);
        Vec4 s1 = Vec4::load(srcPtr + 4 * packCUnit);
        Vec4 s4 = Vec4::load(srcPtr + 5 * packCUnit);
        Vec4 s7 = Vec4::load(srcPtr + 6 * packCUnit);
        Vec4 s10 = Vec4::load(srcPtr + 7 * packCUnit);
        Vec4 s2 = Vec4::load(srcPtr + 8 * packCUnit);
        Vec4 s5 = Vec4::load(srcPtr + 9 * packCUnit);
        Vec4 s8 = Vec4::load(srcPtr + 10 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 11 * packCUnit);
        Vec4::transpose4(s0, s3, s6, s9);
        Vec4::transpose4(s1, s4, s7, s10);
        Vec4::transpose4(s2, s5, s8, s11);

        // to-optimize: interleave load and save in loop
        // deal with pack when packCUnit is 8
        Vec4::save(srcPtr + 0 * packCUnit, s0);
        Vec4::save(srcPtr + 1 * packCUnit, s1);
        Vec4::save(srcPtr + 2 * packCUnit, s2);
        Vec4::save(srcPtr + 3 * packCUnit, s3);
        Vec4::save(srcPtr + 4 * packCUnit, s4);
        Vec4::save(srcPtr + 5 * packCUnit, s5);
        Vec4::save(srcPtr + 6 * packCUnit, s6);
        Vec4::save(srcPtr + 7 * packCUnit, s7);
        Vec4::save(srcPtr + 8 * packCUnit, s8);
        Vec4::save(srcPtr + 9 * packCUnit, s9);
        Vec4::save(srcPtr + 10 * packCUnit, s10);
        Vec4::save(srcPtr + 11 * packCUnit, s11);
        srcPtr += loadTransposeStride;
    }

    //     MNN_PRINT("winograd in BT*D*B, transpose, loadTransposeStride:%zu, dstStep:%zu\n", loadTransposeStride, dstStep);
    // formatMatrix((const float*)srcBlock, {Nh, static_cast<int>(packCUnit), ePack});

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        Vec4 s00 = Vec4::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        Vec4 s01 = Vec4::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        Vec4 s02 = Vec4::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        Vec4 s10 = Vec4::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        Vec4 s12 = Vec4::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        Vec4 s20 = Vec4::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        Vec4 s21 = Vec4::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        Vec4 s22 = Vec4::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        Vec4 s30 = Vec4::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        Vec4 s31 = Vec4::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        Vec4 s32 = Vec4::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        Vec4 s40 = Vec4::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        Vec4 s41 = Vec4::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        Vec4 s42 = Vec4::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        Vec4 s50 = Vec4::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        Vec4 s51 = Vec4::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        Vec4 s52 = Vec4::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        Vec4 s60 = Vec4::load(srcPtr + 6 * loadTransposeStride + 0 * packCUnit);
        Vec4 s61 = Vec4::load(srcPtr + 6 * loadTransposeStride + 1 * packCUnit);
        Vec4 s62 = Vec4::load(srcPtr + 6 * loadTransposeStride + 2 * packCUnit);

        Vec4 s70 = Vec4::load(srcPtr + 7 * loadTransposeStride + 0 * packCUnit);
        Vec4 s71 = Vec4::load(srcPtr + 7 * loadTransposeStride + 1 * packCUnit);
        Vec4 s72 = Vec4::load(srcPtr + 7 * loadTransposeStride + 2 * packCUnit);


        // to-try: reorder complicated commpute of 8x8
        auto ep0 = s00 * 36.f - s20 * 49.f + s40 * 14.f - s60;
        auto ep1 = s01 * 36.f - s21 * 49.f + s41 * 14.f - s61;
        auto ep2 = s02 * 36.f - s22 * 49.f + s42 * 14.f - s62;
        Vec4::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 + s20) * 36.f - (s30 + s40) * 13.f + (s50 + s60);
        ep1 = (s11 + s21) * 36.f - (s31 + s41) * 13.f + (s51 + s61);
        ep2 = (s12 + s22) * 36.f - (s32 + s42) * 13.f + (s52 + s62);
        Vec4::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s20 - s10) * 36.f + (s30 - s40) * 13.f + (s60 - s50);
        ep1 = (s21 - s11) * 36.f + (s31 - s41) * 13.f + (s61 - s51);
        ep2 = (s22 - s12) * 36.f + (s32 - s42) * 13.f + (s62 - s52);
        Vec4::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 18.f + s20 * 9.f - s30 * 20.f - s40 * 10.f + s50 * 2.f + s60;
        ep1 = s11 * 18.f + s21 * 9.f - s31 * 20.f - s41 * 10.f + s51 * 2.f + s61;
        ep2 = s12 * 18.f + s22 * 9.f - s32 * 20.f - s42 * 10.f + s52 * 2.f + s62;
        Vec4::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 9.f - s10 * 18.f + s30 * 20.f - s40 * 10.f - s50 * 2.f + s60;
        ep1 = s21 * 9.f - s11 * 18.f + s31 * 20.f - s41 * 10.f - s51 * 2.f + s61;
        ep2 = s22 * 9.f - s12 * 18.f + s32 * 20.f - s42 * 10.f - s52 * 2.f + s62;
        Vec4::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 12.f + s20 * 4.f - s30 * 15.f - s40 * 5.f + s50 * 3.f + s60;
        ep1 = s11 * 12.f + s21 * 4.f - s31 * 15.f - s41 * 5.f + s51 * 3.f + s61;
        ep2 = s12 * 12.f + s22 * 4.f - s32 * 15.f - s42 * 5.f + s52 * 3.f + s62;
        Vec4::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 4.f - s10 * 12.f + s30 * 15.f - s40 * 5.f - s50 * 3.f + s60;
        ep1 = s21 * 4.f - s11 * 12.f + s31 * 15.f - s41 * 5.f - s51 * 3.f + s61;
        ep2 = s22 * 4.f - s12 * 12.f + s32 * 15.f - s42 * 5.f - s52 * 3.f + s62;
        Vec4::save(dstPtr + 6 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 6 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 6 * dstStep + 2 * packCUnit, ep2);

        ep0 = s30 * 49.f - s10 * 36.f - s50 * 14.f + s70;
        ep1 = s31 * 49.f - s11 * 36.f - s51 * 14.f + s71;
        ep2 = s32 * 49.f - s12 * 36.f - s52 * 14.f + s72;
        Vec4::save(dstPtr + 7 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 7 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 7 * dstStep + 2 * packCUnit, ep2);
        srcPtr += ePack;
        dstPtr += ePack;
    }
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
static void _sourceTransformUnit6x6Pack12(float* srcBlock, float* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/4 = 21
    constexpr int Nh = 6; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // transpose 12x4 to 4x12
        // register number : EPack
        Vec4 s0 = Vec4::load(srcPtr + 0 * packCUnit);
        Vec4 s3 = Vec4::load(srcPtr + 1 * packCUnit);
        Vec4 s6 = Vec4::load(srcPtr + 2 * packCUnit);
        Vec4 s9 = Vec4::load(srcPtr + 3 * packCUnit);
        Vec4 s1 = Vec4::load(srcPtr + 4 * packCUnit);
        Vec4 s4 = Vec4::load(srcPtr + 5 * packCUnit);
        Vec4 s7 = Vec4::load(srcPtr + 6 * packCUnit);
        Vec4 s10 = Vec4::load(srcPtr + 7 * packCUnit);
        Vec4 s2 = Vec4::load(srcPtr + 8 * packCUnit);
        Vec4 s5 = Vec4::load(srcPtr + 9 * packCUnit);
        Vec4 s8 = Vec4::load(srcPtr + 10 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 11 * packCUnit);
        Vec4::transpose4(s0, s3, s6, s9);
        Vec4::transpose4(s1, s4, s7, s10);
        Vec4::transpose4(s2, s5, s8, s11);

        // to-optimize: interleave load and save in loop
        // deal with pack when packCUnit is 8
        Vec4::save(srcPtr + 0 * packCUnit, s0);
        Vec4::save(srcPtr + 1 * packCUnit, s1);
        Vec4::save(srcPtr + 2 * packCUnit, s2);
        Vec4::save(srcPtr + 3 * packCUnit, s3);
        Vec4::save(srcPtr + 4 * packCUnit, s4);
        Vec4::save(srcPtr + 5 * packCUnit, s5);
        Vec4::save(srcPtr + 6 * packCUnit, s6);
        Vec4::save(srcPtr + 7 * packCUnit, s7);
        Vec4::save(srcPtr + 8 * packCUnit, s8);
        Vec4::save(srcPtr + 9 * packCUnit, s9);
        Vec4::save(srcPtr + 10 * packCUnit, s10);
        Vec4::save(srcPtr + 11 * packCUnit, s11);
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {

        Vec4 s00 = Vec4::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        Vec4 s01 = Vec4::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        Vec4 s02 = Vec4::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        Vec4 s10 = Vec4::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        Vec4 s11 = Vec4::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        Vec4 s12 = Vec4::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        Vec4 s20 = Vec4::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        Vec4 s21 = Vec4::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        Vec4 s22 = Vec4::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        Vec4 s30 = Vec4::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        Vec4 s31 = Vec4::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        Vec4 s32 = Vec4::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        Vec4 s40 = Vec4::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        Vec4 s41 = Vec4::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        Vec4 s42 = Vec4::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        Vec4 s50 = Vec4::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        Vec4 s51 = Vec4::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        Vec4 s52 = Vec4::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        // to-try: reorder
        auto ep0 = s00 * 4.f - s20 * 5.f + s40;
        auto ep1 = s01 * 4.f - s21 * 5.f + s41;
        auto ep2 = s02 * 4.f - s22 * 5.f + s42;
        Vec4::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 + s20) * (-4.f) + s30 + s40;
        ep1 = (s11 + s21) * (-4.f) + s31 + s41;
        ep2 = (s12 + s22) * (-4.f) + s32 + s42;
        Vec4::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 - s20) * (4.f) + s40 - s30;
        ep1 = (s11 - s21) * (4.f) + s41 - s31;
        ep2 = (s12 - s22) * (4.f) + s42 - s32;
        Vec4::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * (-2.f) - s20 + s30 * 2.f + s40;
        ep1 = s11 * (-2.f) - s21 + s31 * 2.f + s41;
        ep2 = s12 * (-2.f) - s22 + s32 * 2.f + s42;
        Vec4::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 2.f - s20 - s30 * 2.f + s40;
        ep1 = s11 * 2.f - s21 - s31 * 2.f + s41;
        ep2 = s12 * 2.f - s22 - s32 * 2.f + s42;
        Vec4::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 4.f - s30 * 5.f + s50;
        ep1 = s11 * 4.f - s31 * 5.f + s51;
        ep2 = s12 * 4.f - s32 * 5.f + s52;
        Vec4::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        Vec4::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        Vec4::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        srcPtr += ePack;
        dstPtr += ePack;
    }
}

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


static void _sourceUnrollTransformUnit4x4(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    constexpr size_t srcUnit = 4; // srcUnit

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 - s2;
        auto m1 = s1 + s2;
        auto m2 = s2 - s1;
        auto m3 = s3 - s1;

        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    }
    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
}

static void _sourceUnrollTransformUnit6x6(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 two(2.f);
    Vec4 four(4.f);
    Vec4 five(5.f);
    constexpr size_t srcUnit = 6; // srcUnit
    Vec4 buf0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 buf1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 buf2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 buf3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 buf4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 buf5 = Vec4::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto mid0 = Vec4::fma(buf4, buf2, Vec4(-4));
        auto mid1 = Vec4::fma(buf3, buf1, Vec4(-4));
        auto mid2 = Vec4::fma(buf2, buf0, Vec4(-4));
        auto mid3 = Vec4::fma(buf5, buf3, Vec4(-4));
        auto mid4 = buf4 - buf2;
        auto mid5 = (buf3 - buf1) * Vec4(2);
        Vec4 m0 = mid0 - mid2;
        Vec4 m1 = mid0 + mid1;
        Vec4 m2 = mid0 - mid1;
        Vec4 m3 = mid4 + mid5;
        Vec4 m4 = mid4 - mid5;
        Vec4 m5 = mid3 - mid1;

        buf0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto mid0 = Vec4::fma(buf4, buf2, Vec4(-4));
    auto mid1 = Vec4::fma(buf3, buf1, Vec4(-4));
    auto mid2 = Vec4::fma(buf2, buf0, Vec4(-4));
    auto mid3 = Vec4::fma(buf5, buf3, Vec4(-4));
    auto mid4 = buf4 - buf2;
    auto mid5 = (buf3 - buf1) * Vec4(2);
    Vec4 m0 = mid0 - mid2;
    Vec4 m1 = mid0 + mid1;
    Vec4 m2 = mid0 - mid1;
    Vec4 m3 = mid4 + mid5;
    Vec4 m4 = mid4 - mid5;
    Vec4 m5 = mid3 - mid1;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    Vec4::save(dstFloatPtr + 4 * dstStep, m4);
    Vec4::save(dstFloatPtr + 5 * dstStep, m5);

 }


static void _sourceUnrollTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    constexpr size_t srcUnit = 8; // srcUnit

    Vec4 buf0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 buf1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 buf2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 buf3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 buf4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 buf5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 buf6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 buf7 = Vec4::load(srcBlock + 7 * srcStep);
// #pragma unroll(srcUnit - 1)
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec4 mid0, mid1, mid2;
        mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(36)), buf4, Vec4(-13));
        mid1     = Vec4::fma(Vec4::fma(buf4, buf0, Vec4(36)), buf2, Vec4(-13));
        Vec4 m0 = mid1 - mid0;

        mid2     = Vec4::fma(Vec4::fma(buf5, buf1, Vec4(36)), buf3, Vec4(-13));
        Vec4 m1 = mid0 + mid2;
        Vec4 m2 = mid0 - mid2;
        mid1     = Vec4::fma(Vec4::fma(buf7, buf3, Vec4(36)), buf5, Vec4(-13));
        Vec4 m7 = mid1 - mid2;

        mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(9)), buf4, Vec4(-10));
        mid1     = Vec4::fma(buf5, buf1, Vec4(18)) + Vec4::fma(buf5, buf3, Vec4(-20));
        mid2     = Vec4::fma(buf5 * 3, buf1, Vec4(12));
        Vec4 m3 = mid0 + mid1;
        Vec4 m4 = mid0 - mid1;

        mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(4)), buf4, Vec4(-5));
        mid1     = Vec4::fma(mid2, buf3, Vec4(-15));
        Vec4 m5 = mid0 + mid1;
        Vec4 m6 = mid0 - mid1;

        buf0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 5 * dstStep, m5);
        buf6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 6 * dstStep, m6);
        buf7 = Vec4::load(srcFloatPtr + 7 * srcStep);
        Vec4::save(dstFloatPtr + 7 * dstStep, m7);
    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    Vec4 mid0, mid1, mid2;
    mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(36)), buf4, Vec4(-13));
    mid1     = Vec4::fma(Vec4::fma(buf4, buf0, Vec4(36)), buf2, Vec4(-13));
    Vec4 m0 = mid1 - mid0;

    mid2     = Vec4::fma(Vec4::fma(buf5, buf1, Vec4(36)), buf3, Vec4(-13));
    Vec4 m1 = mid0 + mid2;
    Vec4 m2 = mid0 - mid2;
    mid1     = Vec4::fma(Vec4::fma(buf7, buf3, Vec4(36)), buf5, Vec4(-13));
    Vec4 m7 = mid1 - mid2;

    mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(9)), buf4, Vec4(-10));
    mid1     = Vec4::fma(buf5, buf1, Vec4(18)) + Vec4::fma(buf5, buf3, Vec4(-20));
    mid2     = Vec4::fma(buf5 * 3, buf1, Vec4(12));
    Vec4 m3 = mid0 + mid1;
    Vec4 m4 = mid0 - mid1;

    mid0     = Vec4::fma(Vec4::fma(buf6, buf2, Vec4(4)), buf4, Vec4(-5));
    mid1     = Vec4::fma(mid2, buf3, Vec4(-15));
    Vec4 m5 = mid0 + mid1;
    Vec4 m6 = mid0 - mid1;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    Vec4::save(dstFloatPtr + 4 * dstStep, m4);
    Vec4::save(dstFloatPtr + 5 * dstStep, m5);
    Vec4::save(dstFloatPtr + 6 * dstStep, m6);
    Vec4::save(dstFloatPtr + 7 * dstStep, m7);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit4x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + s3;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;
    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);

}
template<size_t IterLoop>
static void _destUnrollTransformUnit4x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        auto m1 = (s1 - s2);
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + s3;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
}


template<size_t IterLoop>
static void _destUnrollTransformUnit6x5(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    Vec4::save(dstFloatPtr + 4 * dstStep, m4);


}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x4(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {


    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto v0 = s3 + s4;
        auto v1 = s3 - s4;
        auto v2 = s1 + s2;
        auto v3 = s1 - s2;

        auto m0 = s0 + v2 + v0;
        auto m1 = v3 + v1 + v1;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m2 = v2 + v0 * 4.f;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        auto m3 = v3 + v1 * 8.f + s5;
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);

    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);

}
template<size_t IterLoop>
static void _destUnrollTransformUnit6x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
        Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
        Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
        Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
        Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
        Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;
    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);

}


template<size_t IterLoop>
static void _destUnrollTransformUnit8x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    for (int i = 0; i < IterLoop; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + i * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        Vec4 s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        Vec4 s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        Vec4 s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4 s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4 s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4 s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4 s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4 s7 = Vec4::load(srcFloatPtr + 7 * srcStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    }
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        m2 += (s5 + s6) * 9.f + s7;
        m1 += (s5 - s6) * 3.f;
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s7 = Vec4::load(srcFloatPtr + 7 * srcStep);

    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x4(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f + s7;
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = Vec4::load(srcFloatPtr + 7 * srcStep);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop  - 1) * dstRowStep);
    Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
    mid0 = s1 + s2;
    mid1 = s1 - s2;
    mid2 = s3 + s4;
    mid3 = s3 - s4;
    mid4 = s5 + s6;
    mid5 = s5 - s6;
    auto m0 = s0 + mid0 + mid2 + mid4;
    auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
    auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
    auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f + s7;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);


}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x5(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f + s7;
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = Vec4::load(srcFloatPtr + 7 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
    mid0 = s1 + s2;
    mid1 = s1 - s2;
    mid2 = s3 + s4;
    mid3 = s3 - s4;
    mid4 = s5 + s6;
    mid5 = s5 - s6;
    auto m0 = s0 + mid0 + mid2 + mid4;
    auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
    auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
    auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
    auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f + s7;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    Vec4::save(dstFloatPtr + 4 * dstStep, m4);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x6(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {


    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
        s7 = Vec4::load(srcFloatPtr + 7 * srcStep);
        Vec4::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);

    Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
    mid0 = s1 + s2;
    mid1 = s1 - s2;
    auto m0 = s0 + mid0;
    mid2 = s3 + s4;
    mid3 = s3 - s4;
    m0 = m0 + mid2;
    mid4 = s5 + s6;
    mid5 = s5 - s6;
    m0 = m0 + mid4;

    auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
    auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
    auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
    auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
    auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;

    Vec4::save(dstFloatPtr + 0 * dstStep, m0);
    Vec4::save(dstFloatPtr + 1 * dstStep, m1);
    Vec4::save(dstFloatPtr + 2 * dstStep, m2);
    Vec4::save(dstFloatPtr + 3 * dstStep, m3);
    Vec4::save(dstFloatPtr + 4 * dstStep, m4);
    Vec4::save(dstFloatPtr + 5 * dstStep, m5);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x7(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
        Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
        Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
        Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
        Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
        Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
        Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
        Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f;
        s0 = Vec4::load(srcFloatPtr + 0 * srcStep);
        auto m6 = mid0 + mid2 * 64.f + mid4 * 729.f + s7;
        s1 = Vec4::load(srcFloatPtr + 1 * srcStep);

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec4::load(srcFloatPtr + 2 * srcStep);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec4::load(srcFloatPtr + 3 * srcStep);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec4::load(srcFloatPtr + 4 * srcStep);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = Vec4::load(srcFloatPtr + 5 * srcStep);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
        s6 = Vec4::load(srcFloatPtr + 6 * srcStep);
        Vec4::save(dstFloatPtr + 5 * dstStep, m5);
        s7 = Vec4::load(srcFloatPtr + 7 * srcStep);
        Vec4::save(dstFloatPtr + 6 * dstStep, m6);
    }

    auto dstFloatPtr = (float*)(dstStart +(IterLoop  - 1) * dstRowStep);

        Vec4 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f;
        auto m6 = mid0 + mid2 * 64.f + mid4 * 729.f + s7;

        Vec4::save(dstFloatPtr + 0 * dstStep, m0);
        Vec4::save(dstFloatPtr + 1 * dstStep, m1);
        Vec4::save(dstFloatPtr + 2 * dstStep, m2);
        Vec4::save(dstFloatPtr + 3 * dstStep, m3);
        Vec4::save(dstFloatPtr + 4 * dstStep, m4);
        Vec4::save(dstFloatPtr + 5 * dstStep, m5);
        Vec4::save(dstFloatPtr + 6 * dstStep, m6);

}

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

WinogradFunction::TransformPackFunc WinogradFunction::chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {
    if (ePack == 12 && lPack == 1 && packCUnit == 4) {
        if (k == 4 && w == 4) {
            return _sourceTransformUnit4x4Pack12;
        }
        if (k == 6 && w == 6) {
            return _sourceTransformUnit6x6Pack12;
        }
        if (k == 8 && w == 8) {
            return _sourceTransformUnit8x8Pack12;
        }
        // other packing size
    }
    return nullptr;
}

#define SELECT_KH(K, H) if (h == H) return _destTransformUnit##K##x##H
WinogradFunction::TransformFunc WinogradFunction::chooseDestTransform(int k, int h) {
    if (8 == k) {
        SELECT_KH(8, 7);
        SELECT_KH(8, 6);
        SELECT_KH(8, 5);
        SELECT_KH(8, 4);
        SELECT_KH(8, 3);
        SELECT_KH(8, 2);
        return nullptr;
    }
    if (6 == k) {
        SELECT_KH(6, 5);
        SELECT_KH(6, 4);
        SELECT_KH(6, 3);
        SELECT_KH(6, 2);
        return nullptr;
    }
    if (2 == h && 4 == k) {
        return _destTransformUnit4x2;
    }
    if (3 == h && 4 == k) {
        return _destTransformUnit4x3;
    }
    return nullptr;
}


WinogradFunction::WinoUnrollTransFunc WinogradFunction::chooseSourceUnrollTransform(int k, int w) {
    if (8 == k && 8 == w) {
        return _sourceUnrollTransformUnit8x8;
    }
    if (6 == k && 6 == w) {
        return _sourceUnrollTransformUnit6x6;
    }
    if (4 == k && 4 == w) {
        return _sourceUnrollTransformUnit4x4;
    }
    MNN_ASSERT(false);
    return nullptr;
}


void WinogradFunction::chooseWinoDestUnrollTransform(WinogradFunction::WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h) {

    static WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit4[][5] = {
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 0
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 1
        {
            nullptr,
            _destUnrollTransformUnit4x2<1>,
            _destUnrollTransformUnit4x2<2>,
            _destUnrollTransformUnit4x2<3>,
            _destUnrollTransformUnit4x2<4>
        },
        {
            nullptr,
            _destUnrollTransformUnit4x3<1>,
            _destUnrollTransformUnit4x3<2>,
            _destUnrollTransformUnit4x3<3>,
            _destUnrollTransformUnit4x3<4>
        }
    };

    static WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit6[][7] = {
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 0
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 1
        {
            nullptr,
            _destUnrollTransformUnit6x2<1>,
            _destUnrollTransformUnit6x2<2>,
            _destUnrollTransformUnit6x2<3>,
            _destUnrollTransformUnit6x2<4>,
            _destUnrollTransformUnit6x2<5>,
            _destUnrollTransformUnit6x2<6>
        },
        {
            nullptr,
            _destUnrollTransformUnit6x3<1>,
            _destUnrollTransformUnit6x3<2>,
            _destUnrollTransformUnit6x3<3>,
            _destUnrollTransformUnit6x3<4>,
            _destUnrollTransformUnit6x3<5>,
            _destUnrollTransformUnit6x3<6>
        },
        {
            nullptr,
            _destUnrollTransformUnit6x4<1>,
            _destUnrollTransformUnit6x4<2>,
            _destUnrollTransformUnit6x4<3>,
            _destUnrollTransformUnit6x4<4>,
            _destUnrollTransformUnit6x4<5>,
            _destUnrollTransformUnit6x4<6>
        },
        {
            nullptr,
            _destUnrollTransformUnit6x5<1>,
            _destUnrollTransformUnit6x5<2>,
            _destUnrollTransformUnit6x5<3>,
            _destUnrollTransformUnit6x5<4>,
            _destUnrollTransformUnit6x5<5>,
            _destUnrollTransformUnit6x5<6>
        }
    };

    static WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit8[][9] = {
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 0
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        }, // 1
        {
            nullptr,
            _destUnrollTransformUnit8x2<1>,
            _destUnrollTransformUnit8x2<2>,
            _destUnrollTransformUnit8x2<3>,
            _destUnrollTransformUnit8x2<4>,
            _destUnrollTransformUnit8x2<5>,
            _destUnrollTransformUnit8x2<6>,
            _destUnrollTransformUnit8x2<7>,
            _destUnrollTransformUnit8x2<8>
        },
        {
            nullptr,
            _destUnrollTransformUnit8x3<1>,
            _destUnrollTransformUnit8x3<2>,
            _destUnrollTransformUnit8x3<3>,
            _destUnrollTransformUnit8x3<4>,
            _destUnrollTransformUnit8x3<5>,
            _destUnrollTransformUnit8x3<6>,
            _destUnrollTransformUnit8x3<7>,
            _destUnrollTransformUnit8x3<8>
        },
        {
            nullptr,
            _destUnrollTransformUnit8x4<1>,
            _destUnrollTransformUnit8x4<2>,
            _destUnrollTransformUnit8x4<3>,
            _destUnrollTransformUnit8x4<4>,
            _destUnrollTransformUnit8x4<5>,
            _destUnrollTransformUnit8x4<6>,
            _destUnrollTransformUnit8x4<7>,
            _destUnrollTransformUnit8x4<8>
        },
        {
            nullptr,
            _destUnrollTransformUnit8x5<1>,
            _destUnrollTransformUnit8x5<2>,
            _destUnrollTransformUnit8x5<3>,
            _destUnrollTransformUnit8x5<4>,
            _destUnrollTransformUnit8x5<5>,
            _destUnrollTransformUnit8x5<6>,
            _destUnrollTransformUnit8x5<7>,
            _destUnrollTransformUnit8x5<8>
        },
        {
            nullptr,
            _destUnrollTransformUnit8x6<1>,
            _destUnrollTransformUnit8x6<2>,
            _destUnrollTransformUnit8x6<3>,
            _destUnrollTransformUnit8x6<4>,
            _destUnrollTransformUnit8x6<5>,
            _destUnrollTransformUnit8x6<6>,
            _destUnrollTransformUnit8x6<7>,
            _destUnrollTransformUnit8x6<8>
        },
        {
            nullptr,
            _destUnrollTransformUnit8x7<1>,
            _destUnrollTransformUnit8x7<2>,
            _destUnrollTransformUnit8x7<3>,
            _destUnrollTransformUnit8x7<4>,
            _destUnrollTransformUnit8x7<5>,
            _destUnrollTransformUnit8x7<6>,
            _destUnrollTransformUnit8x7<7>,
            _destUnrollTransformUnit8x7<8>
        }
    };

    ::memset((void*)destFunctions, 0, maxUnit * sizeof(WinogradFunction::WinoUnrollDestTransFunc));
    if (8 == k && h > 1 && h < 8) {
        memcpy((void*)destFunctions, gDestTransUnit8[h], (8 + 1) * sizeof(WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
    if (6 == k && h > 1 && h < 6) {
        ::memcpy((void*)destFunctions, gDestTransUnit6[h], (6 + 1) * sizeof(WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
    if (4 == k && h > 1 && h < 4) {
        memcpy((void*)destFunctions, gDestTransUnit4[h], (4 + 1) * sizeof(WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
    MNN_ASSERT(false);
    MNN_ERROR("Can not find function for chooseWinoDestUnrollTransform: k:%d, h:%d\n", k, h);
    return;
}

} // namespace MNN
