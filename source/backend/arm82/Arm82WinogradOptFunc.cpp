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
#include <map>
#include "core/Macro.h"
#include "math/Vec.hpp"
using Vec = MNN::Math::Vec<FLOAT16, 8>;
using VecType = Vec;
using ElementType = FLOAT16;

/* CAUTION:
            fp16 8x8 winograd would lead to larger error for some kinds of models.
            uncomment the following code only if you are sure accuracy is enough for your model.
*/
// #define USE_8x8_WINOGRAD_KERNEL

#define TRANSPOSE_12X8_SAVE()                                               \
    VecType v0  = VecType::load(srcPtr + 0 * packCUnit);                    \
    VecType v1  = VecType::load(srcPtr + 1 * packCUnit);                    \
    VecType v2  = VecType::load(srcPtr + 2 * packCUnit);                    \
    VecType v3  = VecType::load(srcPtr + 3 * packCUnit);                    \
    VecType v4  = VecType::load(srcPtr + 4 * packCUnit);                    \
    VecType v5  = VecType::load(srcPtr + 5 * packCUnit);                    \
    VecType v6  = VecType::load(srcPtr + 6 * packCUnit);                    \
    VecType v7  = VecType::load(srcPtr + 7 * packCUnit);                    \
    VecType v8  = VecType::load(srcPtr + 8 * packCUnit);                    \
    VecType v9  = VecType::load(srcPtr + 9 * packCUnit);                    \
    VecType v10 = VecType::load(srcPtr + 10 * packCUnit);                   \
    VecType v11 = VecType::load(srcPtr + 11 * packCUnit);                   \
    VecType::transpose12(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11); \
    VecType::save(srcPtr + 0 * packCUnit, v0);                              \
    VecType::save(srcPtr + 1 * packCUnit, v1);                              \
    VecType::save(srcPtr + 2 * packCUnit, v2);                              \
    VecType::save(srcPtr + 3 * packCUnit, v3);                              \
    VecType::save(srcPtr + 4 * packCUnit, v4);                              \
    VecType::save(srcPtr + 5 * packCUnit, v5);                              \
    VecType::save(srcPtr + 6 * packCUnit, v6);                              \
    VecType::save(srcPtr + 7 * packCUnit, v7);                              \
    VecType::save(srcPtr + 8 * packCUnit, v8);                              \
    VecType::save(srcPtr + 9 * packCUnit, v9);                              \
    VecType::save(srcPtr + 10 * packCUnit, v10);                            \
    VecType::save(srcPtr + 11 * packCUnit, v11);

namespace MNN {

static void _sourceTransformUnit4x4Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {
    // register number: (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 4; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 8;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_12X8_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit >> 1; ++i4c) // calculate 2 line in 8 packCUnit at once
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

        srcPtr += ePack << 1;
        dstPtr += ePack << 1;
    }
}

static void _sourceTransformUnit8x8Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {

    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit = 27
    // todo: impliment
    constexpr int Nh = 8; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 8;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_12X8_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit >> 1; ++i4c)
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
        Vec ep0, ep1, ep2;
        Vec a0, a1, a2;
        Vec b0, b1, b2;

        a0 = Vec::fma(Vec::fma(s60, s20, Vec(36)), s40, Vec(-13));
        a1 = Vec::fma(Vec::fma(s61, s21, Vec(36)), s41, Vec(-13));
        a2 = Vec::fma(Vec::fma(s62, s22, Vec(36)), s42, Vec(-13));

        b0 = Vec::fma(Vec::fma(s40, s00, Vec(36)), s20, Vec(-13));
        b1 = Vec::fma(Vec::fma(s41, s01, Vec(36)), s21, Vec(-13));
        b2 = Vec::fma(Vec::fma(s42, s02, Vec(36)), s22, Vec(-13));

        ep0 = b0 - a0;
        ep1 = b1 - a1;
        ep2 = b2 - a2;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        b0 = Vec::fma(Vec::fma(s50, s10, Vec(36)), s30, Vec(-13));
        b1 = Vec::fma(Vec::fma(s51, s11, Vec(36)), s31, Vec(-13));
        b2 = Vec::fma(Vec::fma(s52, s12, Vec(36)), s32, Vec(-13));

        ep0 = a0 + b0;
        ep1 = a1 + b1;
        ep2 = a2 + b2;
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = a0 - b0;
        ep1 = a1 - b1;
        ep2 = a2 - b2;
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        a0 = Vec::fma(Vec::fma(s70, s30, Vec(36)), s50, Vec(-13));
        a1 = Vec::fma(Vec::fma(s71, s31, Vec(36)), s51, Vec(-13));
        a2 = Vec::fma(Vec::fma(s72, s32, Vec(36)), s52, Vec(-13));

        ep0 = a0 - b0;
        ep1 = a1 - b1;
        ep2 = a2 - b2;

        VecType::save(dstPtr + 7 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 7 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 7 * dstStep + 2 * packCUnit, ep2);

        a0 = Vec::fma(Vec::fma(s60, s20, Vec(9)), s40, Vec(-10));
        a1 = Vec::fma(Vec::fma(s61, s21, Vec(9)), s41, Vec(-10));
        a2 = Vec::fma(Vec::fma(s62, s22, Vec(9)), s42, Vec(-10));

        b0 = Vec::fma(s50, s10, Vec(18)) + Vec::fma(s50, s30, Vec(-20));
        b1 = Vec::fma(s51, s11, Vec(18)) + Vec::fma(s51, s31, Vec(-20));
        b2 = Vec::fma(s52, s12, Vec(18)) + Vec::fma(s52, s32, Vec(-20));

        ep0 = a0 + b0;
        ep1 = a1 + b1;
        ep2 = a2 + b2;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = a0 - b0;
        ep1 = a1 - b1;
        ep2 = a2 - b2;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);


        a0 = Vec::fma(Vec::fma(s60, s20, Vec(4)), s40, Vec(-5));
        a1 = Vec::fma(Vec::fma(s61, s21, Vec(4)), s41, Vec(-5));
        a2 = Vec::fma(Vec::fma(s62, s22, Vec(4)), s42, Vec(-5));

        b0 = Vec::fma(Vec::fma(s50 * 3, s10, Vec(12)), s30, Vec(-15));
        b1 = Vec::fma(Vec::fma(s51 * 3, s11, Vec(12)), s31, Vec(-15));
        b2 = Vec::fma(Vec::fma(s52 * 3, s12, Vec(12)), s32, Vec(-15));

        ep0 = a0 + b0;
        ep1 = a1 + b1;
        ep2 = a2 + b2;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        ep0 = a0 - b0;
        ep1 = a1 - b1;
        ep2 = a2 - b2;
        VecType::save(dstPtr + 6 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 6 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 6 * dstStep + 2 * packCUnit, ep2);

        srcPtr += ePack << 1;
        dstPtr += ePack << 1;
    }
}

static void _sourceTransformUnit6x6Pack12(ElementType* srcBlock, ElementType* dstStart, size_t dstStep) {
    // source transform D * B. register number : (srcUnit + 1) * EPack/packCUnit
    constexpr int Nh = 6; // srcUnit
    constexpr int ePack = 12;
    constexpr size_t packCUnit = 8;
    const size_t loadTransposeStride = packCUnit * ePack;
    ElementType* srcPtr = srcBlock;
    for (int iNh = 0; iNh < Nh; ++iNh)
    {
        // register number : ePack
        TRANSPOSE_12X8_SAVE();
        srcPtr += loadTransposeStride;
    }

    srcPtr = srcBlock;
    ElementType* dstPtr = dstStart;
    for (int i4c = 0; i4c < packCUnit >> 1; ++i4c)
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
        auto b00 = Vec::fma(s40, s20, Vec(-4));
        auto b01 = Vec::fma(s41, s21, Vec(-4));
        auto b02 = Vec::fma(s42, s22, Vec(-4));

        auto b10 = Vec::fma(s30, s10, Vec(-4));
        auto b11 = Vec::fma(s31, s11, Vec(-4));
        auto b12 = Vec::fma(s32, s12, Vec(-4));

        auto b20 = Vec::fma(s20, s00, Vec(-4));
        auto b21 = Vec::fma(s21, s01, Vec(-4));
        auto b22 = Vec::fma(s22, s02, Vec(-4));

        auto b30 = Vec::fma(s50, s30, Vec(-4));
        auto b31 = Vec::fma(s51, s31, Vec(-4));
        auto b32 = Vec::fma(s52, s32, Vec(-4));

        auto b40 = s40 - s20;
        auto b41 = s41 - s21;
        auto b42 = s42 - s22;

        auto b50 = (s30 - s10) * Vec(2);
        auto b51 = (s31 - s11) * Vec(2);
        auto b52 = (s32 - s12) * Vec(2);

        auto ep0 = b00 - b20;
        auto ep1 = b01 - b21;
        auto ep2 = b02 - b22;
        VecType::save(dstPtr + 0 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 0 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 0 * dstStep + 2 * packCUnit, ep2);

        ep0 = b00 + b10;
        ep1 = b01 + b11;
        ep2 = b02 + b12;
        VecType::save(dstPtr + 1 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 1 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 1 * dstStep + 2 * packCUnit, ep2);

        ep0 = b00 - b10;
        ep1 = b01 - b11;
        ep2 = b02 - b12;
        VecType::save(dstPtr + 2 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 2 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 2 * dstStep + 2 * packCUnit, ep2);

        ep0 = b40 + b50;
        ep1 = b41 + b51;
        ep2 = b42 + b52;
        VecType::save(dstPtr + 3 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 3 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 3 * dstStep + 2 * packCUnit, ep2);

        ep0 = b40 - b50;
        ep1 = b41 - b51;
        ep2 = b42 - b52;
        VecType::save(dstPtr + 4 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 4 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 4 * dstStep + 2 * packCUnit, ep2);

        ep0 = b30 - b10;
        ep1 = b31 - b11;
        ep2 = b32 - b12;
        VecType::save(dstPtr + 5 * dstStep + 0 * packCUnit, ep0);
        VecType::save(dstPtr + 5 * dstStep + 1 * packCUnit, ep1);
        VecType::save(dstPtr + 5 * dstStep + 2 * packCUnit, ep2);

        srcPtr += ePack << 1;
        dstPtr += ePack << 1;
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


Arm82WinogradFunction::TransformPackFunc Arm82WinogradFunction::chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {
    if (ePack == 12 && lPack == 1 && packCUnit == 8) {
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
    //MNN_ERROR("Arm82WinogradFunction Can not find function for ePack:%d, packCUnit:%d\n", ePack, packCUnit);
    return nullptr;
}


Arm82WinogradFunction::WinoUnrollTransFunc Arm82WinogradFunction::chooseSourceUnrollTransform(int k, int w) {

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
    MNN_ASSERT(false);
    return nullptr;
}


void Arm82WinogradFunction::chooseWinoDestUnrollTransform(Arm82WinogradFunction::WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h) {

    static Arm82WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit4[][5] = {
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

    static Arm82WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit6[][7] = {
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

    static Arm82WinogradFunction::WinoUnrollDestTransFunc gDestTransUnit8[][9] = {
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

    ::memset((void*)destFunctions, 0, maxUnit * sizeof(Arm82WinogradFunction::WinoUnrollDestTransFunc));

#ifdef USE_8x8_WINOGRAD_KERNEL
    if (8 == k && h > 1 && h < 8) {
        memcpy((void*)destFunctions, gDestTransUnit8[h], (8 + 1) * sizeof(Arm82WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
#endif
    if (6 == k && h > 1 && h < 6) {
        ::memcpy((void*)destFunctions, gDestTransUnit6[h], (6 + 1) * sizeof(Arm82WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
    if (4 == k && h > 1 && h < 4) {
        memcpy((void*)destFunctions, gDestTransUnit4[h], (4 + 1) * sizeof(Arm82WinogradFunction::WinoUnrollDestTransFunc));
        return;
    }
    //MNN_ERROR("Can not find function for fp16 chooseWinoDestUnrollTransform: k:%d, h:%d\n", k, h);
    return;
}


int Arm82MNNGetConvTileNumber() {
    int eP, lP, hP;
    Arm82MNNGetMatMulPackMode(&eP, &lP, &hP);
    return eP; // 8
}

} // namespace MNN

#undef TRANSPOSE_12X8_SAVE
#undef USE_8x8_WINOGRAD_KERNEL

#endif


