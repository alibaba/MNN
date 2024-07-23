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

/* CAUTION:
            bf16 8x8 winograd would lead to larger error for some kinds of models.
            uncomment the following code only if you are sure accuracy is enough for your model.
*/
// #define USE_8x8_WINOGRAD_KERNEL


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


static void _sourceUnrollTransformUnit4x4(const ElementType* srcBlock, ElementType* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    constexpr size_t srcUnit = 4; // srcUnit

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        auto m0 = s0 - s2;
        auto m1 = s1 + s2;
        auto m2 = s2 - s1;
        auto m3 = s3 - s1;

        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
    }
    auto dstFloatPtr = (ElementType*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto m0 = s0 - s2;
    auto m1 = s1 + s2;
    auto m2 = s2 - s1;
    auto m3 = s3 - s1;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
}

static void _sourceUnrollTransformUnit6x6(const ElementType* srcBlock, ElementType* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType two(2.f);
    VecType four(4.f);
    VecType five(5.f);
    constexpr size_t srcUnit = 6; // srcUnit

    VecType buf0 = VecType::load(srcBlock + 0 * srcStep);
    VecType buf1 = VecType::load(srcBlock + 1 * srcStep);
    VecType buf2 = VecType::load(srcBlock + 2 * srcStep);
    VecType buf3 = VecType::load(srcBlock + 3 * srcStep);
    VecType buf4 = VecType::load(srcBlock + 4 * srcStep);
    VecType buf5 = VecType::load(srcBlock + 5 * srcStep);
// #pragma unroll(srcUnit)
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);
        auto mid0 = VecType::fma(buf4, buf2, VecType(-4));
        auto mid1 = VecType::fma(buf3, buf1, VecType(-4));
        auto mid2 = VecType::fma(buf2, buf0, VecType(-4));
        auto mid3 = VecType::fma(buf5, buf3, VecType(-4));
        auto mid4 = buf4 - buf2;
        auto mid5 = (buf3 - buf1) * VecType(2);
        VecType m0 = mid0 - mid2;
        VecType m1 = mid0 + mid1;
        VecType m2 = mid0 - mid1;
        VecType m3 = mid4 + mid5;
        VecType m4 = mid4 - mid5;
        VecType m5 = mid3 - mid1;

        buf0 = VecType::load(srcFloatPtr + 0 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (srcUnit - 1) * dstRowStep);
    auto mid0 = VecType::fma(buf4, buf2, VecType(-4));
    auto mid1 = VecType::fma(buf3, buf1, VecType(-4));
    auto mid2 = VecType::fma(buf2, buf0, VecType(-4));
    auto mid3 = VecType::fma(buf5, buf3, VecType(-4));
    auto mid4 = buf4 - buf2;
    auto mid5 = (buf3 - buf1) * VecType(2);
    VecType m0 = mid0 - mid2;
    VecType m1 = mid0 + mid1;
    VecType m2 = mid0 - mid1;
    VecType m3 = mid4 + mid5;
    VecType m4 = mid4 - mid5;
    VecType m5 = mid3 - mid1;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
    VecType::save(dstFloatPtr + 4 * dstStep, m4);
    VecType::save(dstFloatPtr + 5 * dstStep, m5);

 }


static void _sourceUnrollTransformUnit8x8(const ElementType* srcBlock, ElementType* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    constexpr size_t srcUnit = 8; // srcUnit
    VecType buf0 = VecType::load(srcBlock + 0 * srcStep);
    VecType buf1 = VecType::load(srcBlock + 1 * srcStep);
    VecType buf2 = VecType::load(srcBlock + 2 * srcStep);
    VecType buf3 = VecType::load(srcBlock + 3 * srcStep);
    VecType buf4 = VecType::load(srcBlock + 4 * srcStep);
    VecType buf5 = VecType::load(srcBlock + 5 * srcStep);
    VecType buf6 = VecType::load(srcBlock + 6 * srcStep);
    VecType buf7 = VecType::load(srcBlock + 7 * srcStep);
// #pragma unroll(srcUnit - 1)
    for (int i = 0; i < srcUnit - 1; ++i) { //Nw iteration
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        VecType mid0, mid1, mid2;
        mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(36)), buf4, VecType(-13));
        mid1     = VecType::fma(VecType::fma(buf4, buf0, VecType(36)), buf2, VecType(-13));
        VecType m0 = mid1 - mid0;

        mid2     = VecType::fma(VecType::fma(buf5, buf1, VecType(36)), buf3, VecType(-13));
        VecType m1 = mid0 + mid2;
        VecType m2 = mid0 - mid2;
        mid1     = VecType::fma(VecType::fma(buf7, buf3, VecType(36)), buf5, VecType(-13));
        VecType m7 = mid1 - mid2;

        mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(9)), buf4, VecType(-10));
        mid1     = VecType::fma(buf5, buf1, VecType(18)) + VecType::fma(buf5, buf3, VecType(-20));
        mid2     = VecType::fma(buf5 * 3, buf1, VecType(12));
        VecType m3 = mid0 + mid1;
        VecType m4 = mid0 - mid1;

        mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(4)), buf4, VecType(-5));
        mid1     = VecType::fma(mid2, buf3, VecType(-15));
        VecType m5 = mid0 + mid1;
        VecType m6 = mid0 - mid1;

        buf0 = VecType::load(srcFloatPtr + 0 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        buf1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        buf2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        buf3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        buf4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
        buf5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 5 * dstStep, m5);
        buf6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 6 * dstStep, m6);
        buf7 = VecType::load(srcFloatPtr + 7 * srcStep);
        VecType::save(dstFloatPtr + 7 * dstStep, m7);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (srcUnit - 1) * dstRowStep);
    VecType mid0, mid1, mid2;
    mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(36)), buf4, VecType(-13));
    mid1     = VecType::fma(VecType::fma(buf4, buf0, VecType(36)), buf2, VecType(-13));
    VecType m0 = mid1 - mid0;

    mid2     = VecType::fma(VecType::fma(buf5, buf1, VecType(36)), buf3, VecType(-13));
    VecType m1 = mid0 + mid2;
    VecType m2 = mid0 - mid2;
    mid1     = VecType::fma(VecType::fma(buf7, buf3, VecType(36)), buf5, VecType(-13));
    VecType m7 = mid1 - mid2;

    mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(9)), buf4, VecType(-10));
    mid1     = VecType::fma(buf5, buf1, VecType(18)) + VecType::fma(buf5, buf3, VecType(-20));
    mid2     = VecType::fma(buf5 * 3, buf1, VecType(12));
    VecType m3 = mid0 + mid1;
    VecType m4 = mid0 - mid1;

    mid0     = VecType::fma(VecType::fma(buf6, buf2, VecType(4)), buf4, VecType(-5));
    mid1     = VecType::fma(mid2, buf3, VecType(-15));
    VecType m5 = mid0 + mid1;
    VecType m6 = mid0 - mid1;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
    VecType::save(dstFloatPtr + 4 * dstStep, m4);
    VecType::save(dstFloatPtr + 5 * dstStep, m5);
    VecType::save(dstFloatPtr + 6 * dstStep, m6);
    VecType::save(dstFloatPtr + 7 * dstStep, m7);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit4x2(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + s3;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
    }
    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2) + s3;
    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);

}
template<size_t IterLoop>
static void _destUnrollTransformUnit4x3(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {
    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2;
        auto m1 = (s1 - s2);
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + s3;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
    }
    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2;
    auto m1 = (s1 - s2);
    auto m2 = (s1 + s2) + s3;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
}


template<size_t IterLoop>
static void _destUnrollTransformUnit6x5(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
    }
    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
    auto m3 = (s1 - s2) + (s3 - s4) * 8.f;
    auto m4 = (s1 + s2) + (s3 + s4) * 16.f + s5;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
    VecType::save(dstFloatPtr + 4 * dstStep, m4);


}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x4(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);
        auto v0 = s3 + s4;
        auto v1 = s3 - s4;
        auto v2 = s1 + s2;
        auto v3 = s1 - s2;

        auto m0 = s0 + v2 + v0;
        auto m1 = v3 + v1 + v1;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m2 = v2 + v0 * 4.f;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        auto m3 = v3 + v1 * 8.f + s5;
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto v0 = s3 + s4;
    auto v1 = s3 - s4;
    auto v2 = s1 + s2;
    auto v3 = s1 - s2;

    auto m0 = s0 + v2 + v0;
    auto m1 = v3 + v1 + v1;
    auto m2 = v2 + v0 * 4.f;
    auto m3 = v3 + v1 * 8.f + s5;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit6x3(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        auto m0 = s0 + s1 + s2 + s3 + s4;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);

    }

    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + s5;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);

}
template<size_t IterLoop>
static void _destUnrollTransformUnit6x2(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        VecType s0 = VecType::load(srcBlock + 0 * srcStep);
        VecType s1 = VecType::load(srcBlock + 1 * srcStep);
        VecType s2 = VecType::load(srcBlock + 2 * srcStep);
        VecType s3 = VecType::load(srcBlock + 3 * srcStep);
        VecType s4 = VecType::load(srcBlock + 4 * srcStep);
        VecType s5 = VecType::load(srcBlock + 5 * srcStep);

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
    }
    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + s5;
    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);

}


template<size_t IterLoop>
static void _destUnrollTransformUnit8x2(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    for (int i = 0; i < IterLoop; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + i * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);
        VecType s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        VecType s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        VecType s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType s7 = VecType::load(srcFloatPtr + 7 * srcStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f + s7;

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
    }
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x3(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {
    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    VecType s6 = VecType::load(srcBlock + 6 * srcStep);
    VecType s7 = VecType::load(srcBlock + 7 * srcStep);

    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);
        auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m2 = (s1 + s2) + (s3 + s4) * 4.f;
        auto m1 = (s1 - s2) + (s3 - s4) * 2.f;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        m2 += (s5 + s6) * 9.f + s7;
        m1 += (s5 - s6) * 3.f;
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s7 = VecType::load(srcFloatPtr + 7 * srcStep);

    }
    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) + (s3 - s4) * 2.f + (s5 - s6) * 3.f;
    auto m2 = (s1 + s2) + (s3 + s4) * 4.f + (s5 + s6) * 9.f + s7;

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x4(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {
    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    VecType s6 = VecType::load(srcBlock + 6 * srcStep);
    VecType s7 = VecType::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        VecType mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f + s7;
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = VecType::load(srcFloatPtr + 7 * srcStep);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop  - 1) * dstRowStep);
    VecType mid0, mid1, mid2, mid3, mid4, mid5;
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

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);


}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x5(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    VecType s6 = VecType::load(srcBlock + 6 * srcStep);
    VecType s7 = VecType::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        VecType mid0, mid1, mid2, mid3, mid4, mid5;
        mid0 = s1 + s2;
        mid1 = s1 - s2;
        mid2 = s3 + s4;
        mid3 = s3 - s4;
        mid4 = s5 + s6;
        mid5 = s5 - s6;
        auto m0 = s0 + mid0 + mid2 + mid4;
        auto m1 = mid1 + mid3 * 2.f + mid5 * 3.f;
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m2 = mid0 + mid2 * 4.f + mid4 * 9.f;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        auto m3 = mid1 + mid3 * 8.f + mid5 * 27.f;
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f + s7;
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        s7 = VecType::load(srcFloatPtr + 7 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);
    VecType mid0, mid1, mid2, mid3, mid4, mid5;
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

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
    VecType::save(dstFloatPtr + 4 * dstStep, m4);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x6(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

    VecType s0 = VecType::load(srcBlock + 0 * srcStep);
    VecType s1 = VecType::load(srcBlock + 1 * srcStep);
    VecType s2 = VecType::load(srcBlock + 2 * srcStep);
    VecType s3 = VecType::load(srcBlock + 3 * srcStep);
    VecType s4 = VecType::load(srcBlock + 4 * srcStep);
    VecType s5 = VecType::load(srcBlock + 5 * srcStep);
    VecType s6 = VecType::load(srcBlock + 6 * srcStep);
    VecType s7 = VecType::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        VecType mid0, mid1, mid2, mid3, mid4, mid5;
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
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m4 = mid0 + mid2 * 16.f + mid4 * 81.f;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);
        auto m5 = mid1 + mid3 * 32.f + mid5 * 243.f + s7;
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
        s7 = VecType::load(srcFloatPtr + 7 * srcStep);
        VecType::save(dstFloatPtr + 5 * dstStep, m5);
    }

    auto dstFloatPtr = (ElementType*)(dstStart + (IterLoop - 1) * dstRowStep);

    VecType mid0, mid1, mid2, mid3, mid4, mid5;
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

    VecType::save(dstFloatPtr + 0 * dstStep, m0);
    VecType::save(dstFloatPtr + 1 * dstStep, m1);
    VecType::save(dstFloatPtr + 2 * dstStep, m2);
    VecType::save(dstFloatPtr + 3 * dstStep, m3);
    VecType::save(dstFloatPtr + 4 * dstStep, m4);
    VecType::save(dstFloatPtr + 5 * dstStep, m5);

}

template<size_t IterLoop>
static void _destUnrollTransformUnit8x7(const ElementType* srcBlock, ElementType* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep) {

        VecType s0 = VecType::load(srcBlock + 0 * srcStep);
        VecType s1 = VecType::load(srcBlock + 1 * srcStep);
        VecType s2 = VecType::load(srcBlock + 2 * srcStep);
        VecType s3 = VecType::load(srcBlock + 3 * srcStep);
        VecType s4 = VecType::load(srcBlock + 4 * srcStep);
        VecType s5 = VecType::load(srcBlock + 5 * srcStep);
        VecType s6 = VecType::load(srcBlock + 6 * srcStep);
        VecType s7 = VecType::load(srcBlock + 7 * srcStep);
    for (int i = 0; i < IterLoop - 1; ++i) {
        auto srcFloatPtr = (const ElementType*)(srcBlock + (i + 1) * srcRowStep);
        auto dstFloatPtr = (ElementType*)(dstStart + i * dstRowStep);

        VecType mid0, mid1, mid2, mid3, mid4, mid5;
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
        s0 = VecType::load(srcFloatPtr + 0 * srcStep);
        auto m6 = mid0 + mid2 * 64.f + mid4 * 729.f + s7;
        s1 = VecType::load(srcFloatPtr + 1 * srcStep);

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        s2 = VecType::load(srcFloatPtr + 2 * srcStep);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        s3 = VecType::load(srcFloatPtr + 3 * srcStep);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        s4 = VecType::load(srcFloatPtr + 4 * srcStep);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        s5 = VecType::load(srcFloatPtr + 5 * srcStep);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
        s6 = VecType::load(srcFloatPtr + 6 * srcStep);
        VecType::save(dstFloatPtr + 5 * dstStep, m5);
        s7 = VecType::load(srcFloatPtr + 7 * srcStep);
        VecType::save(dstFloatPtr + 6 * dstStep, m6);
    }

    auto dstFloatPtr = (ElementType*)(dstStart +(IterLoop  - 1) * dstRowStep);

        VecType mid0, mid1, mid2, mid3, mid4, mid5;
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

        VecType::save(dstFloatPtr + 0 * dstStep, m0);
        VecType::save(dstFloatPtr + 1 * dstStep, m1);
        VecType::save(dstFloatPtr + 2 * dstStep, m2);
        VecType::save(dstFloatPtr + 3 * dstStep, m3);
        VecType::save(dstFloatPtr + 4 * dstStep, m4);
        VecType::save(dstFloatPtr + 5 * dstStep, m5);
        VecType::save(dstFloatPtr + 6 * dstStep, m6);

}


WinogradFunctionHalf::TransformPackFunc WinogradFunctionHalf::chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {

    if (ePack == 12 && lPack == 1 && packCUnit == 4) {
        if (k == 4 && w == 4) {
            return _sourceTransformUnit4x4Pack12;
        }
        if (k == 6 && w == 6) {
            return _sourceTransformUnit6x6Pack12;
        }
#ifdef USE_8x8_WINOGRAD_KERNEL
        if (k == 8 && w == 8) {
            return _sourceTransformUnit8x8Pack12;
        }
#endif
        // other packing size
    }

    // if (ePack == 3 && lPack == 8 && packCUnit == 4)  no need to transformPack for x86 bf16 pack format of 3 x 8 x 4, will not be called in ConvolutionWinograd.cpp by allow_x86_bf16_winograd
    return nullptr;
}

WinogradFunctionHalf::WinoUnrollTransFunc WinogradFunctionHalf::chooseSourceUnrollTransform(int k, int w) {
#ifdef USE_8x8_WINOGRAD_KERNEL
    if (8 == k && 8 == w) {
        return _sourceUnrollTransformUnit8x8;
    }
#endif
    if (6 == k && 6 == w) {
        return _sourceUnrollTransformUnit6x6;
    }
    if (4 == k && 4 == w) {
        return _sourceUnrollTransformUnit4x4;
    }
    return nullptr;
}


void WinogradFunctionHalf::chooseWinoDestUnrollTransform(WinogradFunctionHalf::WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h) {

    static WinogradFunctionHalf::WinoUnrollDestTransFunc gDestTransUnit4[][5] = {
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

    static WinogradFunctionHalf::WinoUnrollDestTransFunc gDestTransUnit6[][7] = {
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

    static WinogradFunctionHalf::WinoUnrollDestTransFunc gDestTransUnit8[][9] = {
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

    ::memset((void*)destFunctions, 0, maxUnit * sizeof(WinogradFunctionHalf::WinoUnrollDestTransFunc));

#ifdef USE_8x8_WINOGRAD_KERNEL
    if (8 == k && h > 1 && h < 8) {
        memcpy((void*)destFunctions, gDestTransUnit8[h], (8 + 1) * sizeof(WinogradFunctionHalf::WinoUnrollDestTransFunc));
        return;
    }
#endif
    if (6 == k && h > 1 && h < 6) {
        ::memcpy((void*)destFunctions, gDestTransUnit6[h], (6 + 1) * sizeof(WinogradFunctionHalf::WinoUnrollDestTransFunc));
        return;
    }
    if (4 == k && h > 1 && h < 4) {
        memcpy((void*)destFunctions, gDestTransUnit4[h], (4 + 1) * sizeof(WinogradFunctionHalf::WinoUnrollDestTransFunc));
        return;
    }
    return;
}


} // namespace MNN

#undef TRANSPOSE_12X4_SAVE
#undef USE_8x8_WINOGRAD_KERNEL

