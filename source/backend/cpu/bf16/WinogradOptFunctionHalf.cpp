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
#include <map>
#include "core/Macro.h"
#include "VecHalf.hpp"
using BFVec4 = MNN::Math::VecHalf<4>;
using VecType = BFVec4;
using ElementType = int16_t;

// to be optimized into VecType::transpose12
#define TRANSPOSE_12X4_SAVE()                             \
    VecType s0  = VecType::load(srcPtr + 0 * packCUnit);  \
    VecType s3  = VecType::load(srcPtr + 1 * packCUnit);  \
    VecType s6  = VecType::load(srcPtr + 2 * packCUnit);  \
    VecType s9  = VecType::load(srcPtr + 3 * packCUnit);  \
    VecType s1  = VecType::load(srcPtr + 4 * packCUnit);  \
    VecType s4  = VecType::load(srcPtr + 5 * packCUnit);  \
    VecType s7  = VecType::load(srcPtr + 6 * packCUnit);  \
    VecType s10 = VecType::load(srcPtr + 7 * packCUnit);  \
    VecType s2  = VecType::load(srcPtr + 8 * packCUnit);  \
    VecType s5  = VecType::load(srcPtr + 9 * packCUnit);  \
    VecType s8  = VecType::load(srcPtr + 10 * packCUnit); \
    VecType s11 = VecType::load(srcPtr + 11 * packCUnit); \
    VecType::transpose4(s0, s3, s6, s9);                  \
    VecType::transpose4(s1, s4, s7, s10);                 \
    VecType::transpose4(s2, s5, s8, s11);                 \
    VecType::save(srcPtr + 0 * packCUnit, s0);            \
    VecType::save(srcPtr + 1 * packCUnit, s1);            \
    VecType::save(srcPtr + 2 * packCUnit, s2);            \
    VecType::save(srcPtr + 3 * packCUnit, s3);            \
    VecType::save(srcPtr + 4 * packCUnit, s4);            \
    VecType::save(srcPtr + 5 * packCUnit, s5);            \
    VecType::save(srcPtr + 6 * packCUnit, s6);            \
    VecType::save(srcPtr + 7 * packCUnit, s7);            \
    VecType::save(srcPtr + 8 * packCUnit, s8);            \
    VecType::save(srcPtr + 9 * packCUnit, s9);            \
    VecType::save(srcPtr + 10 * packCUnit, s10);          \
    VecType::save(srcPtr + 11 * packCUnit, s11);

namespace MNN {


static void _sourceTransformUnit4x4Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {
    // register number: (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 4; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        // TRANSPOSE_12X4_SAVE();
        VecType::transpose12(srcPtr, packCUnit);
        srcPtr += loadTransposeStride;
    }
    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        // source transform D * B. register number : srcUnit * (EPack/4 + 1)
        VecType s00 = VecType::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        VecType s01 = VecType::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        VecType s02 = VecType::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        VecType s10 = VecType::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        VecType s11 = VecType::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        VecType s12 = VecType::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        VecType s20 = VecType::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        VecType s21 = VecType::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        VecType s22 = VecType::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        VecType s30 = VecType::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        VecType s31 = VecType::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        VecType s32 = VecType::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        // dstStep =  ePack * pack * ic_4
        auto ep0 = s00 - s20;
        auto ep1 = s01 - s21;
        auto ep2 = s02 - s22;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 + s20;
        ep1 = s11 + s21;
        ep2 = s12 + s22;
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 - s10;
        ep1 = s21 - s11;
        ep2 = s22 - s12;
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s30 - s10;
        ep1 = s31 - s11;
        ep2 = s32 - s12;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        // VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, s00);
        // VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, s01);
        // VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, s02);

        // VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, s10);
        // VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, s11);
        // VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, s12);

        // VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, s20);
        // VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, s21);
        // VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, s22);

        // VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, s30);
        // VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, s31);
        // VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, s32);

        // MNN_PRINT("\nwinograd in BT*D*B, iNh:0-3, i4c:%d\n", i4c);
        // formatMatrix(dstPtr + 0 * dstStep , {ePack});
        // formatMatrix(dstPtr + 1 * dstStep , {ePack});
        // formatMatrix(dstPtr + 2 * dstStep , {ePack});
        // formatMatrix(dstPtr + 3 * dstStep , {ePack});

        srcPtr += ePack;
        dstPtr += ePack;
    }
}

static void _sourceTransformUnit8x8Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit = 27
    // todo: impliment
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        VecType::transpose12(srcPtr, packCUnit);
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        VecType s00 = VecType::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        VecType s01 = VecType::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        VecType s02 = VecType::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        VecType s10 = VecType::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        VecType s11 = VecType::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        VecType s12 = VecType::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        VecType s20 = VecType::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        VecType s21 = VecType::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        VecType s22 = VecType::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        VecType s30 = VecType::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        VecType s31 = VecType::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        VecType s32 = VecType::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        VecType s40 = VecType::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        VecType s41 = VecType::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        VecType s42 = VecType::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        VecType s50 = VecType::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        VecType s51 = VecType::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        VecType s52 = VecType::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        VecType s60 = VecType::load(srcPtr + 6 * loadTransposeStride + 0 * packCUnit);
        VecType s61 = VecType::load(srcPtr + 6 * loadTransposeStride + 1 * packCUnit);
        VecType s62 = VecType::load(srcPtr + 6 * loadTransposeStride + 2 * packCUnit);

        VecType s70 = VecType::load(srcPtr + 7 * loadTransposeStride + 0 * packCUnit);
        VecType s71 = VecType::load(srcPtr + 7 * loadTransposeStride + 1 * packCUnit);
        VecType s72 = VecType::load(srcPtr + 7 * loadTransposeStride + 2 * packCUnit);


        // to-try: reorder complicated commpute of 8x8
        auto ep0 = s00 * 36.f - s20 * 49.f + s40 * 14.f - s60;
        auto ep1 = s01 * 36.f - s21 * 49.f + s41 * 14.f - s61;
        auto ep2 = s02 * 36.f - s22 * 49.f + s42 * 14.f - s62;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 + s20) * 36.f - (s30 + s40) * 13.f + (s50 + s60);
        ep1 = (s11 + s21) * 36.f - (s31 + s41) * 13.f + (s51 + s61);
        ep2 = (s12 + s22) * 36.f - (s32 + s42) * 13.f + (s52 + s62);
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s20 - s10) * 36.f + (s30 - s40) * 13.f + (s60 - s50);
        ep1 = (s21 - s11) * 36.f + (s31 - s41) * 13.f + (s61 - s51);
        ep2 = (s22 - s12) * 36.f + (s32 - s42) * 13.f + (s62 - s52);
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 18.f + s20 * 9.f - s30 * 20.f - s40 * 10.f + s50 * 2.f + s60;
        ep1 = s11 * 18.f + s21 * 9.f - s31 * 20.f - s41 * 10.f + s51 * 2.f + s61;
        ep2 = s12 * 18.f + s22 * 9.f - s32 * 20.f - s42 * 10.f + s52 * 2.f + s62;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 9.f - s10 * 18.f + s30 * 20.f - s40 * 10.f - s50 * 2.f + s60;
        ep1 = s21 * 9.f - s11 * 18.f + s31 * 20.f - s41 * 10.f - s51 * 2.f + s61;
        ep2 = s22 * 9.f - s12 * 18.f + s32 * 20.f - s42 * 10.f - s52 * 2.f + s62;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 12.f + s20 * 4.f - s30 * 15.f - s40 * 5.f + s50 * 3.f + s60;
        ep1 = s11 * 12.f + s21 * 4.f - s31 * 15.f - s41 * 5.f + s51 * 3.f + s61;
        ep2 = s12 * 12.f + s22 * 4.f - s32 * 15.f - s42 * 5.f + s52 * 3.f + s62;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        ep0 = s20 * 4.f - s10 * 12.f + s30 * 15.f - s40 * 5.f - s50 * 3.f + s60;
        ep1 = s21 * 4.f - s11 * 12.f + s31 * 15.f - s41 * 5.f - s51 * 3.f + s61;
        ep2 = s22 * 4.f - s12 * 12.f + s32 * 15.f - s42 * 5.f - s52 * 3.f + s62;
        VecType::save(dstPtr + 6 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 6 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 6 * dstStep + 2 * packCUnit, ep2);

        ep0 = s30 * 49.f - s10 * 36.f - s50 * 14.f + s70;
        ep1 = s31 * 49.f - s11 * 36.f - s51 * 14.f + s71;
        ep2 = s32 * 49.f - s12 * 36.f - s52 * 14.f + s72;
        VecType::save(dstPtr + 7 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 7 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 7 * dstStep + 2 * packCUnit, ep2);
        srcPtr += ePack;
        dstPtr += ePack;
    }
}

static void _sourceTransformUnit6x6Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 6; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 4;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        VecType::transpose12(srcPtr, packCUnit);
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit; ++i4c)
    {
        VecType s00 = VecType::load(srcPtr + 0 * loadTransposeStride + 0 * packCUnit);
        VecType s01 = VecType::load(srcPtr + 0 * loadTransposeStride + 1 * packCUnit);
        VecType s02 = VecType::load(srcPtr + 0 * loadTransposeStride + 2 * packCUnit);

        VecType s10 = VecType::load(srcPtr + 1 * loadTransposeStride + 0 * packCUnit);
        VecType s11 = VecType::load(srcPtr + 1 * loadTransposeStride + 1 * packCUnit);
        VecType s12 = VecType::load(srcPtr + 1 * loadTransposeStride + 2 * packCUnit);

        VecType s20 = VecType::load(srcPtr + 2 * loadTransposeStride + 0 * packCUnit);
        VecType s21 = VecType::load(srcPtr + 2 * loadTransposeStride + 1 * packCUnit);
        VecType s22 = VecType::load(srcPtr + 2 * loadTransposeStride + 2 * packCUnit);

        VecType s30 = VecType::load(srcPtr + 3 * loadTransposeStride + 0 * packCUnit);
        VecType s31 = VecType::load(srcPtr + 3 * loadTransposeStride + 1 * packCUnit);
        VecType s32 = VecType::load(srcPtr + 3 * loadTransposeStride + 2 * packCUnit);

        VecType s40 = VecType::load(srcPtr + 4 * loadTransposeStride + 0 * packCUnit);
        VecType s41 = VecType::load(srcPtr + 4 * loadTransposeStride + 1 * packCUnit);
        VecType s42 = VecType::load(srcPtr + 4 * loadTransposeStride + 2 * packCUnit);

        VecType s50 = VecType::load(srcPtr + 5 * loadTransposeStride + 0 * packCUnit);
        VecType s51 = VecType::load(srcPtr + 5 * loadTransposeStride + 1 * packCUnit);
        VecType s52 = VecType::load(srcPtr + 5 * loadTransposeStride + 2 * packCUnit);

        // to-try: reorder
        auto ep0 = s00 * 4.f - s20 * 5.f + s40;
        auto ep1 = s01 * 4.f - s21 * 5.f + s41;
        auto ep2 = s02 * 4.f - s22 * 5.f + s42;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 + s20) * (-4.f) + s30 + s40;
        ep1 = (s11 + s21) * (-4.f) + s31 + s41;
        ep2 = (s12 + s22) * (-4.f) + s32 + s42;
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = (s10 - s20) * (4.f) + s40 - s30;
        ep1 = (s11 - s21) * (4.f) + s41 - s31;
        ep2 = (s12 - s22) * (4.f) + s42 - s32;
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * (-2.f) - s20 + s30 * 2.f + s40;
        ep1 = s11 * (-2.f) - s21 + s31 * 2.f + s41;
        ep2 = s12 * (-2.f) - s22 + s32 * 2.f + s42;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 2.f - s20 - s30 * 2.f + s40;
        ep1 = s11 * 2.f - s21 - s31 * 2.f + s41;
        ep2 = s12 * 2.f - s22 - s32 * 2.f + s42;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = s10 * 4.f - s30 * 5.f + s50;
        ep1 = s11 * 4.f - s31 * 5.f + s51;
        ep2 = s12 * 4.f - s32 * 5.f + s52;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        srcPtr += ePack;
        dstPtr += ePack;
    }
}

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
WinogradFunctionHalf::TransformPackFunc WinogradFunctionHalf::chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {

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
    // if (ePack == 3 && lPack == 8 && packCUnit == 4)  no need to transformPack for x86 bf16 pack format of 3 x 8 x 4, will not be called in ConvolutionWinograd.cpp by allow_x86_bf16_winograd
    return nullptr;
}


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
        switch (h) {
            case 5:
                return _destTransformUnit6x5;
            case 4:
                return _destTransformUnit6x4;
            case 3:
                return _destTransformUnit6x3;
            case 2:
                return _destTransformUnit6x2;
            default:
                return nullptr;
        }
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

#undef TRANSPOSE_12X4_SAVE


