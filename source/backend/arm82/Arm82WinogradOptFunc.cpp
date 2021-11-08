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

extern "C" {
    void MNNConvWinoSourceTransformUnit6x6FP16(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep);
}

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

static void MNNConvWinoSourceTransformUnit6x6FP16(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep) {
    LOAD6;

    auto b0 = Vec::fma(s4, s2, Vec(-4));
    auto b1 = Vec::fma(s3, s1, Vec(-4));
    auto b2 = Vec::fma(s2, s0, Vec(-4));
    auto b3 = Vec::fma(s5, s3, Vec(-4));
    auto b4 = s4 - s2;
    auto b5 = (s3 - s1) * Vec(2);

    Vec m0 = b0 - b2;
    Vec m1 = b0 + b1;
    Vec m2 = b0 - b1;
    Vec m3 = b4 + b5;
    Vec m4 = b4 - b5;
    Vec m5 = b3 - b1;

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

    auto v0 = s1 + s2;
    auto v1 = s1 - s2;
    auto v2 = s3 + s4;
    auto v3 = s3 - s4;
    
    auto m0 = s0 + v0 + v2;
    auto m1 = Vec::fma(v1, v3, Vec(2));
    auto m2 = Vec::fma(v0, v2, Vec(4));
    auto m3 = Vec::fma(v1, v3, Vec(8));
    auto m4 = Vec::fma(v0, v2, Vec(16)) + s5;

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
    auto m2 = Vec::fma(v2, v0, Vec(4));
    auto m3 = Vec::fma(v3, v1, Vec(8)) + s5;

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

    auto v0 = s1 + s2;
    auto v1 = s3 + s4;

    auto m0 = s0 + v0 + v1;
    auto m1 = Vec::fma(s1 - s2, s3 - s4, Vec(2));
    auto m2 = Vec::fma(v0, v1, Vec(4)) + s5;

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


Arm82WinogradFunction::TransformPackFunc Arm82WinogradFunction::chooseWinoSourceTransformPack(int k, int w, int ePack, int lPack, int packCUnit) {
    if (ePack == 12 && lPack == 1 && packCUnit == 8) {
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
    MNN_ERROR("Arm82WinogradFunction Can not find function for ePack:%d, packCUnit:%d\n", ePack, packCUnit);
    MNN_ASSERT(false);
    return nullptr;
}


Arm82WinogradFunction::TransformFunc Arm82WinogradFunction::chooseSourceTransform(int k, int w) {
    if (6 == k && 6 == w) {
        return MNNConvWinoSourceTransformUnit6x6FP16;
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

#undef TRANSPOSE_12X8_SAVE

#endif


