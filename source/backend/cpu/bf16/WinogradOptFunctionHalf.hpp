//
//  WinogradOptFunctionHalf.hpp
//  MNN
//
//  Created by MNN on 2021/03/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradOptFunctionHalf_hpp
#define WinogradOptFunctionHalf_hpp

#include <stdint.h>
#include <stdio.h>

namespace MNN {
class WinogradFunctionHalf {
public:
    typedef void (*TransformFunc)(const int16_t* srcBlock, int16_t* dstStart, size_t srcStep, size_t dstStep);
    typedef void (*TransformPackFunc)(int16_t* srcBlock, int16_t* dstStart, size_t dstStep);
    typedef void (*WinoUnrollTransFunc)(const int16_t* srcBlock, int16_t* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);
    typedef void (*WinoUnrollDestTransFunc)(const int16_t* srcBlock, int16_t* dstStart, const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);

    /*Use the generator with interp 0.5*/
    static TransformPackFunc chooseWinoSourceTransformPack(int k, int h, int ePack, int lPack, int packCUnit);
    static WinoUnrollTransFunc chooseSourceUnrollTransform(int k, int w) ;
    static void chooseWinoDestUnrollTransform(WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h);
};
} // namespace MNN

#endif /* WinogradOptFunctionHalf_hpp */
