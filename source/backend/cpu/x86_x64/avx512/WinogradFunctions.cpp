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

#define FUSE_BIAS_ACTIVATE(count, vecBias, minValue, maxValue)            \
    /* count is constantexpr, if would be optimized out*/                 \
    if (count > 0) m0 = m0 + vecBias;                                     \
    if (count > 1) m1 = m1 + vecBias;                                     \
    if (count > 2) m2 = m2 + vecBias;                                     \
    if (count > 3) m3 = m3 + vecBias;                                     \
    if (count > 4) m4 = m4 + vecBias;                                     \
    if (count > 5) m5 = m5 + vecBias;                                     \
    if (count > 6) m6 = m6 + vecBias;                                     \
                                                                          \
    if (count > 0) m0 = VecType::max(m0, minValue);                       \
    if (count > 1) m1 = VecType::max(m1, minValue);                       \
    if (count > 2) m2 = VecType::max(m2, minValue);                       \
    if (count > 3) m3 = VecType::max(m3, minValue);                       \
    if (count > 4) m4 = VecType::max(m4, minValue);                       \
    if (count > 5) m5 = VecType::max(m5, minValue);                       \
    if (count > 6) m6 = VecType::max(m6, minValue);                       \
                                                                          \
    if (count > 0) m0 = VecType::min(m0, maxValue);                       \
    if (count > 1) m1 = VecType::min(m1, maxValue);                       \
    if (count > 2) m2 = VecType::min(m2, maxValue);                       \
    if (count > 3) m3 = VecType::min(m3, maxValue);                       \
    if (count > 4) m4 = VecType::min(m4, maxValue);                       \
    if (count > 5) m5 = VecType::min(m5, maxValue);                       \
    if (count > 6) m6 = VecType::min(m6, maxValue);                       \
                                                                          \
    if (count > 7) MNN_ASSERT(false);


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


// unroll, fuse, interleave row and colum of source matrix multiply with load
static void _sourceFuseTransformUnit6x6(const float* srcBlock, float* midBuffer, float* dstStart, size_t srcRowStep, size_t unitStep) {
    Vec16 two(2.f);
    Vec16 four(4.f);
    Vec16 five(5.f);
    constexpr size_t srcUnit = 6; // srcUnit
    constexpr size_t packCUnit = 16;
    constexpr size_t bytes = 4;

    for (int i = 0; i < srcUnit; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + i * srcRowStep);
        auto dstFloatPtr = (float*)(midBuffer + i * packCUnit);
        constexpr size_t srcStep = packCUnit;
        constexpr size_t dstStep = packCUnit * srcUnit;
        Vec16 m0, m1, m2, m3, m4, m5;
        Vec16 s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16 s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        m0 = s0 * four;
        m2 = s1 * four;
        m1 = - m2;
        m5 = m2;
        m4 = s1 * two;
        m3 = - m4;
        Vec16 s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        m0 = Vec16::fms(m0, s2, five); // m0 -= s2 * five;
        m1 = Vec16::fms(m1, s2, four); // m1 -= s2 * four;
        m2 = Vec16::fms(m2, s2, four); // m2 -= s2 * four;
        m3 -= s2;
        m4 -= s2;
        Vec16 s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        m1 += s3;
        m2 -= s3;
        m3 = Vec16::fma(m3, s3, two); //m3 += s3 * two;
        m4 = Vec16::fms(m4, s3, two); // m4 -= s3 * two;
        m5 = Vec16::fms(m5, s3, five);// m5 -= s3 * five;
        Vec16 s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        m0 += s4;
        m1 += s4;
        m2 += s4;
        m3 += s4;
        m4 += s4;
        Vec16 s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        m5 += s5;
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
    }

    for (int i = 0; i < srcUnit; ++i) { //Nh iteration
        auto srcFloatPtr = (const float*)(midBuffer + i * srcUnit * packCUnit);
        auto dstFloatPtr = (float*)(dstStart + i * unitStep);
        constexpr size_t srcStep = packCUnit;
        size_t dstStep = unitStep * srcUnit;

        Vec16 m0, m1, m2, m3, m4, m5;
        Vec16 s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16 s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        m0 = s0 * four;
        m2 = s1 * four;
        m1 = - m2;
        m5 = m2;
        m4 = s1 * two;
        m3 = - m4;
        Vec16 s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        m0 = Vec16::fms(m0, s2, five); // m0 -= s2 * five;
        m1 = Vec16::fms(m1, s2, four); // m1 -= s2 * four;
        m2 = Vec16::fms(m2, s2, four); // m2 -= s2 * four;
        m3 -= s2;
        m4 -= s2;
        Vec16 s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        m1 += s3;
        m2 -= s3;
        m3 = Vec16::fma(m3, s3, two); //m3 += s3 * two;
        m4 = Vec16::fms(m4, s3, two); // m4 -= s3 * two;
        m5 = Vec16::fms(m5, s3, five);// m5 -= s3 * five;
        Vec16 s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        m0 += s4;
        m1 += s4;
        m2 += s4;
        m3 += s4;
        m4 += s4;
        Vec16 s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        m5 += s5;
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
    }

}


// unroll interleave
static void _sourceUnrollTransformUnit4x4(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep1, size_t dstStep) {

    constexpr size_t srcUnit = 4; // srcUnit
    constexpr int srcStep = PACK_UNIT;
    // constexpr size_t dstStep = PACK_UNIT * srcUnit;
    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 - s2;
        auto m1 = s1 + s2;
        auto m2 = s2 - s1;
        auto m3 = s3 - s1;

        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    }
    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);

}

// unroll, interleave load.
static void _sourceUnrollTransformUnit6x6(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep1, size_t dstStep) {
    Vec16 two(2.f);
    Vec16 four(4.f);
    Vec16 five(5.f);
    constexpr size_t srcUnit = 6; // srcUnit
    constexpr int srcStep = PACK_UNIT;
    // constexpr size_t dstStep = PACK_UNIT * srcUnit;

    Vec16 buf0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 buf1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 buf2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 buf3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 buf4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 buf5 = Vec16::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto mid0 = Vec16::fma(buf4, buf2, Vec16(-4));
        auto mid1 = Vec16::fma(buf3, buf1, Vec16(-4));
        auto mid2 = Vec16::fma(buf2, buf0, Vec16(-4));
        auto mid3 = Vec16::fma(buf5, buf3, Vec16(-4));
        auto mid4 = buf4 - buf2;
        auto mid5 = (buf3 - buf1) * Vec16(2);
        Vec16 m0 = mid0 - mid2;
        Vec16 m1 = mid0 + mid1;
        Vec16 m2 = mid0 - mid1;
        Vec16 m3 = mid4 + mid5;
        Vec16 m4 = mid4 - mid5;
        Vec16 m5 = mid3 - mid1;

        buf0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto mid0 = Vec16::fma(buf4, buf2, Vec16(-4));
    auto mid1 = Vec16::fma(buf3, buf1, Vec16(-4));
    auto mid2 = Vec16::fma(buf2, buf0, Vec16(-4));
    auto mid3 = Vec16::fma(buf5, buf3, Vec16(-4));
    auto mid4 = buf4 - buf2;
    auto mid5 = (buf3 - buf1) * Vec16(2);
    Vec16 m0 = mid0 - mid2;
    Vec16 m1 = mid0 + mid1;
    Vec16 m2 = mid0 - mid1;
    Vec16 m3 = mid4 + mid5;
    Vec16 m4 = mid4 - mid5;
    Vec16 m5 = mid3 - mid1;

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    Vec16::save(dstFloatPtr + 4 * dstStep, m4);
    Vec16::save(dstFloatPtr + 5 * dstStep, m5);
}


// interleave load, reuse fma
static void _sourceUnrollTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep1, size_t dstStep) {

    constexpr size_t srcUnit = 8; // srcUnit
    constexpr int srcStep = PACK_UNIT;
    // constexpr int dstStep = PACK_UNIT * srcUnit;
    Vec16 buf0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 buf1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 buf2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 buf3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 buf4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 buf5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 buf6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 buf7 = Vec16::load(srcBlock + 7 * srcStep);
// #pragma unroll(srcUnit - 1)
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec16 mid0, mid1, mid2;
        mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(36)), buf4, Vec16(-13));
        mid1     = Vec16::fma(Vec16::fma(buf4, buf0, Vec16(36)), buf2, Vec16(-13));
        Vec16 m0 = mid1 - mid0;

        mid2     = Vec16::fma(Vec16::fma(buf5, buf1, Vec16(36)), buf3, Vec16(-13));
        Vec16 m1 = mid0 + mid2;
        Vec16 m2 = mid0 - mid2;
        mid1     = Vec16::fma(Vec16::fma(buf7, buf3, Vec16(36)), buf5, Vec16(-13));
        Vec16 m7 = mid1 - mid2;

        mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(9)), buf4, Vec16(-10));
        mid1     = Vec16::fma(buf5, buf1, Vec16(18)) + Vec16::fma(buf5, buf3, Vec16(-20));
        mid2     = Vec16::fma(buf5 * 3, buf1, Vec16(12));
        Vec16 m3 = mid0 + mid1;
        Vec16 m4 = mid0 - mid1;

        mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(4)), buf4, Vec16(-5));
        mid1     = Vec16::fma(mid2, buf3, Vec16(-15));
        Vec16 m5 = mid0 + mid1;
        Vec16 m6 = mid0 - mid1;

        buf0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
        buf6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 6 * dstStep, m6);
        buf7 = Vec16::load(srcFloatPtr + 7 * srcStep);
        Vec16::save(dstFloatPtr + 7 * dstStep, m7);
    }

    auto dstFloatPtr = (float*)(dstStart + (srcUnit - 1) * dstRowStep);
    Vec16 mid0, mid1, mid2;
    mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(36)), buf4, Vec16(-13));
    mid1     = Vec16::fma(Vec16::fma(buf4, buf0, Vec16(36)), buf2, Vec16(-13));
    Vec16 m0 = mid1 - mid0;

    mid2     = Vec16::fma(Vec16::fma(buf5, buf1, Vec16(36)), buf3, Vec16(-13));
    Vec16 m1 = mid0 + mid2;
    Vec16 m2 = mid0 - mid2;
    mid1     = Vec16::fma(Vec16::fma(buf7, buf3, Vec16(36)), buf5, Vec16(-13));
    Vec16 m7 = mid1 - mid2;

    mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(9)), buf4, Vec16(-10));
    mid1     = Vec16::fma(buf5, buf1, Vec16(18)) + Vec16::fma(buf5, buf3, Vec16(-20));
    mid2     = Vec16::fma(buf5 * 3, buf1, Vec16(12));
    Vec16 m3 = mid0 + mid1;
    Vec16 m4 = mid0 - mid1;

    mid0     = Vec16::fma(Vec16::fma(buf6, buf2, Vec16(4)), buf4, Vec16(-5));
    mid1     = Vec16::fma(mid2, buf3, Vec16(-15));
    Vec16 m5 = mid0 + mid1;
    Vec16 m6 = mid0 - mid1;

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    Vec16::save(dstFloatPtr + 4 * dstStep, m4);
    Vec16::save(dstFloatPtr + 5 * dstStep, m5);
    Vec16::save(dstFloatPtr + 6 * dstStep, m6);
    Vec16::save(dstFloatPtr + 7 * dstStep, m7);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit4x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + s3;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);

        if (nullptr != bias) {
            VecType m2, m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(2, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;

    if (nullptr != bias) {
        VecType m2, m3, m4, m5, m6;
        FUSE_BIAS_ACTIVATE(2, vecBias, minValue, maxValue);
    }
    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);

}
template<size_t IterLoop>
static void _destUnrollTransformUnit4x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        auto m1 = (s1 - s2);
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + s3;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        if (nullptr != bias) {
            VecType m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
        }
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    if (nullptr != bias) {
        VecType m3, m4, m5, m6;
        FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
}


template<size_t IterLoop>
static void _destUnrollTransformUnit6x5(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);

        if (nullptr != bias) {
            VecType m5, m6;
            FUSE_BIAS_ACTIVATE(5, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    if (nullptr != bias) {
        VecType m5, m6;
        FUSE_BIAS_ACTIVATE(5, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    Vec16::save(dstFloatPtr + 4 * dstStep, m4);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x4(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto v0 = s3 + s4;
        auto v1 = s3 - s4;
        auto v2 = s1 + s2;
        auto v3 = s1 - s2;

        auto m0 = s0 + v2 + v0;
        auto m1 = v3 + v1 + v1;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m2 = v2 + v0 * 4.f;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        auto m3 = v3 + v1 * 8.f + s5;
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);

        if (nullptr != bias) {
            VecType m4, m5, m6;
            FUSE_BIAS_ACTIVATE(4, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
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

    if (nullptr != bias) {
        VecType m4, m5, m6;
        FUSE_BIAS_ACTIVATE(4, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

        if (nullptr != bias) {
            VecType m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);

    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    if (nullptr != bias) {
        VecType m3, m4, m5, m6;
        FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);


}
template<size_t IterLoop>
static void _destUnrollTransformUnit6x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
        Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
        Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
        Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
        Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
        Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);

        VecType vecBias, minValue, maxValue;
        if (nullptr != bias) {
            vecBias = VecType::load(bias);
            minValue = VecType::broadcast(postParameters + 2);
            maxValue = VecType::broadcast(postParameters + 3);
        }
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);

        if (nullptr != bias) {
            VecType m2, m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(2, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;

    if (nullptr != bias) {
        VecType m2, m3, m4, m5, m6;
        FUSE_BIAS_ACTIVATE(2, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
}


template<size_t IterLoop>
static void _destUnrollTransformUnit8x2(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }
// #pragma unroll(IterLoop)
    for (int i = 0; i < IterLoop; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + i * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        Vec16 s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        Vec16 s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        Vec16 s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16 s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16 s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16 s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16 s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16 s7 = Vec16::load(srcFloatPtr + 7 * srcStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

        if (nullptr != bias) {
            VecType m2, m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(2, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    }
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x3(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

// #pragma unroll(IterLoop - 1)
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        m2 += (s5 + s6) * 9.f + s7;
        m1 += (s5 - s6) * 3.f;
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);

        if (nullptr != bias) {
            VecType m3, m4, m5, m6;
            FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s7 = Vec16::load(srcFloatPtr + 7 * srcStep);

    }
    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    if (nullptr != bias) {
        VecType m3, m4, m5, m6;
        FUSE_BIAS_ACTIVATE(3, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x4(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

// #pragma unroll(IterLoop - 1)
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f + s7;
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);

        if (nullptr != bias) {
            VecType m4, m5, m6;
            FUSE_BIAS_ACTIVATE(4, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = Vec16::load(srcFloatPtr + 7 * srcStep);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop  - 1) * dstRowStep);
    Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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
    if (nullptr != bias) {
        VecType m4, m5, m6;
        FUSE_BIAS_ACTIVATE(4, vecBias, minValue, maxValue);
    }
    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x5(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }

// #pragma unroll(IterLoop - 1)
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f + s7;
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);

        if (nullptr != bias) {
            VecType m5, m6;
            FUSE_BIAS_ACTIVATE(5, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = Vec16::load(srcFloatPtr + 7 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);
    Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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

    if (nullptr != bias) {
        VecType m5, m6;
        FUSE_BIAS_ACTIVATE(5, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    Vec16::save(dstFloatPtr + 4 * dstStep, m4);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x6(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
    Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
    Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
    Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
    Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
    Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
    Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
    Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

    VecType vecBias, minValue, maxValue;
    if (nullptr != bias) {
        vecBias = VecType::load(bias);
        minValue = VecType::broadcast(postParameters + 2);
        maxValue = VecType::broadcast(postParameters + 3);
    }
// #pragma unroll(IterLoop - 1)
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);

        if (nullptr != bias) {
            VecType m6;
            FUSE_BIAS_ACTIVATE(6, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        s7 = Vec16::load(srcFloatPtr + 7 * srcStep);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (float*)(dstStart + (IterLoop - 1) * dstRowStep);

    Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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

    if (nullptr != bias) {
        VecType m6;
        FUSE_BIAS_ACTIVATE(6, vecBias, minValue, maxValue);
    }

    Vec16::save(dstFloatPtr + 0 * dstStep, m0);
    Vec16::save(dstFloatPtr + 1 * dstStep, m1);
    Vec16::save(dstFloatPtr + 2 * dstStep, m2);
    Vec16::save(dstFloatPtr + 3 * dstStep, m3);
    Vec16::save(dstFloatPtr + 4 * dstStep, m4);
    Vec16::save(dstFloatPtr + 5 * dstStep, m5);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x7(const float* srcBlock, float* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        Vec16 s0 = Vec16::load(srcBlock + 0 * srcStep);
        Vec16 s1 = Vec16::load(srcBlock + 1 * srcStep);
        Vec16 s2 = Vec16::load(srcBlock + 2 * srcStep);
        Vec16 s3 = Vec16::load(srcBlock + 3 * srcStep);
        Vec16 s4 = Vec16::load(srcBlock + 4 * srcStep);
        Vec16 s5 = Vec16::load(srcBlock + 5 * srcStep);
        Vec16 s6 = Vec16::load(srcBlock + 6 * srcStep);
        Vec16 s7 = Vec16::load(srcBlock + 7 * srcStep);

        VecType vecBias, minValue, maxValue;
        if (nullptr != bias) {
            vecBias = VecType::load(bias);
            minValue = VecType::broadcast(postParameters + 2);
            maxValue = VecType::broadcast(postParameters + 3);
        }
// #pragma unroll(IterLoop - 1)
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const float*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (float*)(dstStart + i * dstRowStep);

        Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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
        s0 = Vec16::load(srcFloatPtr + 0 * srcStep);
        auto m6 = mid0 + mid2 * 64.f + mid4 * 729.f + s7;
        s1 = Vec16::load(srcFloatPtr + 1 * srcStep);

        if (nullptr != bias) {
            FUSE_BIAS_ACTIVATE(7, vecBias, minValue, maxValue);
        }

        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = Vec16::load(srcFloatPtr + 2 * srcStep);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = Vec16::load(srcFloatPtr + 3 * srcStep);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = Vec16::load(srcFloatPtr + 4 * srcStep);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = Vec16::load(srcFloatPtr + 5 * srcStep);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        s6 = Vec16::load(srcFloatPtr + 6 * srcStep);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
        s7 = Vec16::load(srcFloatPtr + 7 * srcStep);
        Vec16::save(dstFloatPtr + 6 * dstStep, m6);
    }

    auto dstFloatPtr = (float*)(dstStart +(IterLoop  - 1) * dstRowStep);

        Vec16 mid0, mid1, mid2, mid3, mid4, mid5;
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

        if (nullptr != bias) {
            FUSE_BIAS_ACTIVATE(7, vecBias, minValue, maxValue);
        }
        Vec16::save(dstFloatPtr + 0 * dstStep, m0);
        Vec16::save(dstFloatPtr + 1 * dstStep, m1);
        Vec16::save(dstFloatPtr + 2 * dstStep, m2);
        Vec16::save(dstFloatPtr + 3 * dstStep, m3);
        Vec16::save(dstFloatPtr + 4 * dstStep, m4);
        Vec16::save(dstFloatPtr + 5 * dstStep, m5);
        Vec16::save(dstFloatPtr + 6 * dstStep, m6);

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

static CoreFunctions::WinoUnrollTransFunc AVX512_chooseSourceUnrollTransform(int k, int w) {
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


static void AVX512_chooseWinoDestUnrollTransform(CoreFunctions::WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h) {

    static CoreFunctions::WinoUnrollDestTransFunc gDestTransUnit4[][5] = {
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

    static CoreFunctions::WinoUnrollDestTransFunc gDestTransUnit6[][7] = {
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

    static CoreFunctions::WinoUnrollDestTransFunc gDestTransUnit8[][9] = {
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

    ::memset((void*)destFunctions, 0, maxUnit * sizeof(CoreFunctions::WinoUnrollDestTransFunc));
    if (8 == k && h > 1 && h < 8) {
        memcpy((void*)destFunctions, gDestTransUnit8[h], (8 + 1) * sizeof(CoreFunctions::WinoUnrollDestTransFunc));
        return;
    }
    if (6 == k && h > 1 && h < 6) {
        ::memcpy((void*)destFunctions, gDestTransUnit6[h], (6 + 1) * sizeof(CoreFunctions::WinoUnrollDestTransFunc));
        return;
    }
    if (4 == k && h > 1 && h < 4) {
        memcpy((void*)destFunctions, gDestTransUnit4[h], (4 + 1) * sizeof(CoreFunctions::WinoUnrollDestTransFunc));
        return;
    }
    MNN_ASSERT(false);
    MNN_ERROR("Can not find function for AVX512_chooseWinoDestUnrollTransform:k %d, h:%d\n", k, h);
    return;
}
#undef FUSE_BIAS_ACTIVATE
#undef LOAD_16Ix16
#undef SAVE_3Ix16
#undef TRANSPOSE_48X16_SAVE
#undef LOAD_1x48
#undef SAVE_1x48

};
void _AVX512_WinogradInit(void* functions) {
    auto core = reinterpret_cast<MNN::CoreFunctions*>(functions);

    core->chooseWinoSourceTransformPack = MNN::_AVX512_chooseWinoSourceTransformPack;
    core->chooseWinoSourceUnrollTransform = MNN::AVX512_chooseSourceUnrollTransform;
    core->chooseWinoDestUnrollTransform = MNN::AVX512_chooseWinoDestUnrollTransform;
}
