//
//  WinogradFunctions.cpp
//  MNN
//
//  Created by MNN on 2021/05/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include "Vec16.hpp"
#include "FunctionSummary.hpp"
#define PACK_UNIT 16

using VecType = Vec16;


#define LOAD_16Ix16(i)                                                          \
    VecType s0_##i  = VecType::load(srcPtr + (i * packCUnit + 0) * packCUnit);  \
    VecType s1_##i  = VecType::load(srcPtr + (i * packCUnit + 1) * packCUnit);  \
    VecType s2_##i  = VecType::load(srcPtr + (i * packCUnit + 2) * packCUnit);  \
    VecType s3_##i  = VecType::load(srcPtr + (i * packCUnit + 3) * packCUnit);  \
    VecType s4_##i  = VecType::load(srcPtr + (i * packCUnit + 4) * packCUnit);  \
    VecType s5_##i  = VecType::load(srcPtr + (i * packCUnit + 5) * packCUnit);  \
    VecType s6_##i  = VecType::load(srcPtr + (i * packCUnit + 6) * packCUnit);  \
    VecType s7_##i  = VecType::load(srcPtr + (i * packCUnit + 7) * packCUnit);  \
    VecType s8_##i  = VecType::load(srcPtr + (i * packCUnit + 8) * packCUnit);  \
    VecType s9_##i  = VecType::load(srcPtr + (i * packCUnit + 9) * packCUnit);  \
    VecType s10_##i = VecType::load(srcPtr + (i * packCUnit + 10) * packCUnit); \
    VecType s11_##i = VecType::load(srcPtr + (i * packCUnit + 11) * packCUnit); \
    VecType s12_##i = VecType::load(srcPtr + (i * packCUnit + 12) * packCUnit); \
    VecType s13_##i = VecType::load(srcPtr + (i * packCUnit + 13) * packCUnit); \
    VecType s14_##i = VecType::load(srcPtr + (i * packCUnit + 14) * packCUnit); \
    VecType s15_##i = VecType::load(srcPtr + (i * packCUnit + 15) * packCUnit);

#define SAVE_3Ix16(i)                                          \
    VecType::save(srcPtr + (i * 3 + 0) * packCUnit, s##i##_0); \
    VecType::save(srcPtr + (i * 3 + 1) * packCUnit, s##i##_1); \
    VecType::save(srcPtr + (i * 3 + 2) * packCUnit, s##i##_2);

#define TRANSPOSE_48X16_SAVE()                                                                                        \
    LOAD_16Ix16(0);                                                                                                  \
    LOAD_16Ix16(1);                                                                                                  \
    LOAD_16Ix16(2);                                                                                                  \
    VecType::transpose16(s0_0, s1_0, s2_0, s3_0, s4_0, s5_0, s6_0, s7_0, s8_0, s9_0, s10_0, s11_0, s12_0, s13_0, \
                             s14_0, s15_0);                                                                          \
    VecType::transpose16(s0_1, s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1, s8_1, s9_1, s10_1, s11_1, s12_1, s13_1, \
                             s14_1, s15_1);                                                                          \
    VecType::transpose16(s0_2, s1_2, s2_2, s3_2, s4_2, s5_2, s6_2, s7_2, s8_2, s9_2, s10_2, s11_2, s12_2, s13_2, \
                             s14_2, s15_2);                                                                          \
    /* to-optimize: interleave load and save in loop*/                                                               \
    SAVE_3Ix16(0);                                                                                                   \
    SAVE_3Ix16(1);                                                                                                   \
    SAVE_3Ix16(2);                                                                                                   \
    SAVE_3Ix16(3);                                                                                                   \
    SAVE_3Ix16(4);                                                                                                   \
    SAVE_3Ix16(5);                                                                                                   \
    SAVE_3Ix16(6);                                                                                                   \
    SAVE_3Ix16(7);                                                                                                   \
    SAVE_3Ix16(8);                                                                                                   \
    SAVE_3Ix16(9);                                                                                                   \
    SAVE_3Ix16(10);                                                                                                  \
    SAVE_3Ix16(11);                                                                                                  \
    SAVE_3Ix16(12);                                                                                                  \
    SAVE_3Ix16(13);                                                                                                  \
    SAVE_3Ix16(14);                                                                                                  \
    SAVE_3Ix16(15);

#define LOAD_1x48(i)                                                                   \
    VecType s##i##0 = VecType::load(srcPtr + (i)*loadTransposeStride + 0 * packCUnit); \
    VecType s##i##1 = VecType::load(srcPtr + (i)*loadTransposeStride + 1 * packCUnit); \
    VecType s##i##2 = VecType::load(srcPtr + (i)*loadTransposeStride + 2 * packCUnit);

#define SAVE_1x48(i)                                          \
    VecType::save(dstPtr + (i)*dstStep + 0 * packCUnit, ep0); \
    VecType::save(dstPtr + (i)*dstStep + 1 * packCUnit, ep1); \
    VecType::save(dstPtr + (i)*dstStep + 2 * packCUnit, ep2);

namespace MNN {

static void _sourceTransformUnit4x4Pack48(float* srcBlock, float* dstStart, size_t dstStep) {

    // register number: (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 4; // srcUnit
    constexpr int ePack = 48;
    constexpr size_t packCUnit = 16;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_48X16_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        LOAD_1x48(0);
        LOAD_1x48(1);
        LOAD_1x48(2);
        LOAD_1x48(3);
        // dstStep =  ePack * pack * ic_4
        auto ep0 = s00 - s20;
        auto ep1 = s01 - s21;
        auto ep2 = s02 - s22;
        SAVE_1x48(0);

        ep0 = s10 + s20;
        ep1 = s11 + s21;
        ep2 = s12 + s22;
        SAVE_1x48(1);

        ep0 = s20 - s10;
        ep1 = s21 - s11;
        ep2 = s22 - s12;
        SAVE_1x48(2);

        ep0 = s30 - s10;
        ep1 = s31 - s11;
        ep2 = s32 - s12;
        SAVE_1x48(3);

        srcPtr += ePack;
        dstPtr += ePack;
    }
}


static void _sourceTransformUnit6x6Pack48(float* srcBlock, float* dstStart, size_t dstStep) {

    // register number: (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 6; // srcUnit
    constexpr int ePack = 48;
    constexpr size_t packCUnit = 16;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_48X16_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        LOAD_1x48(0);
        LOAD_1x48(1);
        LOAD_1x48(2);
        LOAD_1x48(3);
        LOAD_1x48(4);
        LOAD_1x48(5);
        // dstStep =  ePack * pack * ic_4
        auto ep0 = s00 * 4.f - s20 * 5.f + s40;
        auto ep1 = s01 * 4.f - s21 * 5.f + s41;
        auto ep2 = s02 * 4.f - s22 * 5.f + s42;
        SAVE_1x48(0);

        ep0 = (s10 + s20) * (-4.f) + s30 + s40;
        ep1 = (s11 + s21) * (-4.f) + s31 + s41;
        ep2 = (s12 + s22) * (-4.f) + s32 + s42;
        SAVE_1x48(1);

        ep0 = (s10 - s20) * (4.f) + s40 - s30;
        ep1 = (s11 - s21) * (4.f) + s41 - s31;
        ep2 = (s12 - s22) * (4.f) + s42 - s32;
        SAVE_1x48(2);

        ep0 = s10 * (-2.f) - s20 + s30 * 2.f + s40;
        ep1 = s11 * (-2.f) - s21 + s31 * 2.f + s41;
        ep2 = s12 * (-2.f) - s22 + s32 * 2.f + s42;
        SAVE_1x48(3);

        ep0 = s10 * 2.f - s20 - s30 * 2.f + s40;
        ep1 = s11 * 2.f - s21 - s31 * 2.f + s41;
        ep2 = s12 * 2.f - s22 - s32 * 2.f + s42;
        SAVE_1x48(4);

        ep0 = s10 * 4.f - s30 * 5.f + s50;
        ep1 = s11 * 4.f - s31 * 5.f + s51;
        ep2 = s12 * 4.f - s32 * 5.f + s52;
        SAVE_1x48(5);
        srcPtr += ePack;
        dstPtr += ePack;
    }
}


static void _sourceTransformUnit8x8Pack48(float* srcBlock, float* dstStart, size_t dstStep) {
    // register number: (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 48;
    constexpr size_t packCUnit = 16;
    const size_t loadTransposeStride = packCUnit * ePack;
    float* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_48X16_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    float* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        LOAD_1x48(0);
        LOAD_1x48(1);
        LOAD_1x48(2);
        LOAD_1x48(3);
        LOAD_1x48(4);
        LOAD_1x48(5);
        LOAD_1x48(6);
        LOAD_1x48(7);

        // dstStep =  ePack * pack * ic_4
        auto ep0 = s00 * 36.f - s20 * 49.f + s40 * 14.f - s60;
        auto ep1 = s01 * 36.f - s21 * 49.f + s41 * 14.f - s61;
        auto ep2 = s02 * 36.f - s22 * 49.f + s42 * 14.f - s62;
        SAVE_1x48(0);

        ep0 = (s10 + s20) * 36.f - (s30 + s40) * 13.f + (s50 + s60);
        ep1 = (s11 + s21) * 36.f - (s31 + s41) * 13.f + (s51 + s61);
        ep2 = (s12 + s22) * 36.f - (s32 + s42) * 13.f + (s52 + s62);
        SAVE_1x48(1);

        ep0 = (s20 - s10) * 36.f + (s30 - s40) * 13.f + (s60 - s50);
        ep1 = (s21 - s11) * 36.f + (s31 - s41) * 13.f + (s61 - s51);
        ep2 = (s22 - s12) * 36.f + (s32 - s42) * 13.f + (s62 - s52);
        SAVE_1x48(2);

        ep0 = s10 * 18.f + s20 * 9.f - s30 * 20.f - s40 * 10.f + s50 * 2.f + s60;
        ep1 = s11 * 18.f + s21 * 9.f - s31 * 20.f - s41 * 10.f + s51 * 2.f + s61;
        ep2 = s12 * 18.f + s22 * 9.f - s32 * 20.f - s42 * 10.f + s52 * 2.f + s62;
        SAVE_1x48(3);

        ep0 = s20 * 9.f - s10 * 18.f + s30 * 20.f - s40 * 10.f - s50 * 2.f + s60;
        ep1 = s21 * 9.f - s11 * 18.f + s31 * 20.f - s41 * 10.f - s51 * 2.f + s61;
        ep2 = s22 * 9.f - s12 * 18.f + s32 * 20.f - s42 * 10.f - s52 * 2.f + s62;
        SAVE_1x48(4);

        ep0 = s10 * 12.f + s20 * 4.f - s30 * 15.f - s40 * 5.f + s50 * 3.f + s60;
        ep1 = s11 * 12.f + s21 * 4.f - s31 * 15.f - s41 * 5.f + s51 * 3.f + s61;
        ep2 = s12 * 12.f + s22 * 4.f - s32 * 15.f - s42 * 5.f + s52 * 3.f + s62;
        SAVE_1x48(5);

        ep0 = s20 * 4.f - s10 * 12.f + s30 * 15.f - s40 * 5.f - s50 * 3.f + s60;
        ep1 = s21 * 4.f - s11 * 12.f + s31 * 15.f - s41 * 5.f - s51 * 3.f + s61;
        ep2 = s22 * 4.f - s12 * 12.f + s32 * 15.f - s42 * 5.f - s52 * 3.f + s62;
        SAVE_1x48(6);

        ep0 = s30 * 49.f - s10 * 36.f - s50 * 14.f + s70;
        ep1 = s31 * 49.f - s11 * 36.f - s51 * 14.f + s71;
        ep2 = s32 * 49.f - s12 * 36.f - s52 * 14.f + s72;
        SAVE_1x48(7);
        srcPtr += ePack;
        dstPtr += ePack;
    }
}


static void _sourceTransformUnit4x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);

    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit4x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
}
static void _destTransformUnit4x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);

    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
}

#define LOAD8                                     \
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep); \
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep); \
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep); \
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep); \
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep); \
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep); \
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep); \
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

static void _sourceTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    Vec16 m0 = s0 * 36.f - s2 * 49.f + s4 * 14.f - s6;

    Vec16 m1 = (s1 + s2) * 36.f - (s3 + s4) * 13.f + (s5 + s6);
    Vec16 m2 = (s2 - s1) * 36.f + (s3 - s4) * 13.f + (s6 - s5);

    Vec16 m3 = s1 * 18.f + s2 * 9.f - s3 * 20.f - s4 * 10.f + s5 * 2.f + s6;
    Vec16 m4 = s2 * 9.f - s1 * 18.f + s3 * 20.f - s4 * 10.f - s5 * 2.f + s6;

    Vec16 m5 = s1 * 12.f + s2 * 4.f - s3 * 15.f - s4 * 5.f + s5 * 3.f + s6;
    Vec16 m6 = s2 * 4.f - s1 * 12.f + s3 * 15.f - s4 * 5.f - s5 * 3.f + s6;

    Vec16 m7 = s3 * 49.f - s1 * 36.f - s5 * 14.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
    Vec16::save(dstStart + 5 * dstStep, m5);
    Vec16::save(dstStart + 6 * dstStep, m6);
    Vec16::save(dstStart + 7 * dstStep, m7);
}

static void _destTransformUnit8x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
}

static void _destTransformUnit8x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
}

static void _destTransformUnit8x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
}

static void _destTransformUnit8x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD8;
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
}

static void _destTransformUnit8x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
    Vec16::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit8x7(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f + (s5 - s6) * 27.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + (s5 + s6) * 81.f;
    auto m5 = (s1 - s2) + (s3 - s4) * 32.f + (s5 - s6) * 243.f;
    auto m6 = (s1 + s2) + (s3 + s4) * 64.f + (s5 + s6) * 729.f + s7;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
    Vec16::save(dstStart + 5 * dstStep, m5);
    Vec16::save(dstStart + 6 * dstStep, m6);
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
Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep); \
Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep); \
Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep); \
Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep); \
Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep); \
Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

static void _sourceTransformUnit6x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;
    Vec16 m0 = s0 * 4.f - s2 * 5.f + s4;

    Vec16 m1 = (s1 + s2) * (-4.f) + (s3 + s4);
    Vec16 m2 = (s1 - s2) * (4.f) + (s4 - s3);

    Vec16 m3 = s1 * -2.f - s2 + s3 * 2.f + s4;
    Vec16 m4 = s1 * 2.f - s2 - s3 * 2.f + s4;

    Vec16 m5 = s1 * 4.f - s3 * 5.f + s5;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
    Vec16::save(dstStart + 5 * dstStep, m5);
}

static void _destTransformUnit6x5(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
    Vec16::save(dstStart + 4 * dstStep, m4);
}
static void _destTransformUnit6x4(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
    Vec16::save(dstStart + 3 * dstStep, m3);
}
static void _destTransformUnit6x3(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
    Vec16::save(dstStart + 2 * dstStep, m2);
}
static void _destTransformUnit6x2(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;

    Vec16::save(dstStart + 0 * dstStep, m0);
    Vec16::save(dstStart + 1 * dstStep, m1);
}

static CoreFunctions::WinoTransFunc gProcUnit6[] = {
    nullptr, // 0
    nullptr, // 1
    _destTransformUnit6x2,
    _destTransformUnit6x3,
    _destTransformUnit6x4,
    _destTransformUnit6x5,
};


static CoreFunctions::WinoTransFunc _AVX512_chooseSourceTransform(int k, int w) {
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

static CoreFunctions::WinoTransFunc _AVX512_chooseDestTransform(int k, int h) {
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

static CoreFunctions::WinoTransPackFunc _AVX512_chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {

    if (ePack == 48 && lPack == 1 && packCUnit == 16) {
        if (k == 4 && w == 4) {
            return _sourceTransformUnit4x4Pack48;
        }
        if (k == 6 && w == 6) {
            return _sourceTransformUnit6x6Pack48;
        }
        if (k == 8 && w == 8) {
            return _sourceTransformUnit8x8Pack48;
        }
        // other packing size
    }
    MNN_ERROR("Can not find function for ePack:%d, packCUnit:%d\n", ePack, packCUnit);
    MNN_ASSERT(false);
    return nullptr;
}

#undef LOAD_16Ix16
#undef SAVE_3Ix16
#undef TRANSPOSE_48X16_SAVE
#undef LOAD_1x48
#undef SAVE_1x48

};
void _AVX512_WinogradInit(void* functions) {
    auto core = reinterpret_cast<MNN::CoreFunctions*>(functions);
    core->chooseWinoDestTransform = MNN::_AVX512_chooseDestTransform;
    core->chooseWinoSourceTransform = MNN::_AVX512_chooseSourceTransform;
    core->chooseWinoSourceTransformPack = MNN::_AVX512_chooseWinoSourceTransformPack;
}
