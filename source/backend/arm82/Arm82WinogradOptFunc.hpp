//
//  Arm82WinogradOptFunc.hpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82WinogradOptFunc_hpp
#define Arm82WinogradOptFunc_hpp

#include "Arm82Backend.hpp"

namespace MNN {
class Arm82WinogradFunction {
public:
    typedef void (*TransformFunc)(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep);
    typedef void (*Int8TransFunc)(const int8_t* srcBlock, int8_t* dstStart, size_t srcStep, size_t dstStep);
    typedef void (*TransformPackFunc)(FLOAT16* srcBlock, FLOAT16* dstStart, size_t dstStep);
    typedef void (*WinoUnrollTransFunc)(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);
    typedef void (*WinoUnrollDestTransFunc)(const FLOAT16* srcBlock, FLOAT16* dstStart,  const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);

    /*Use the generator with interp 0.5*/
    static TransformPackFunc chooseWinoSourceTransformPack(int k, int h, int ePack, int lPack, int packCUnit);
    static WinoUnrollTransFunc chooseSourceUnrollTransform(int k, int w) ;
    static void chooseWinoDestUnrollTransform(WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h);

    static Int8TransFunc chooseInt8SourceTransform(int k, int w);
    static TransformFunc chooseInt8DestTransform(int k, int h);
};

int Arm82MNNGetConvTileNumber();

} // namespace MNN

#endif /* Arm82WinogradOptFunc_hpp */
#endif
