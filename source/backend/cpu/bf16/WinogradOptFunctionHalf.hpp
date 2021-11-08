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
    /*Use the generator with interp 0.5*/
    static TransformFunc chooseSourceTransform(int k, int w);
    static TransformFunc chooseDestTransform(int k, int h);
    static TransformPackFunc chooseWinoSourceTransformPack(int k, int h, int ePack, int lPack, int packCUnit);
};
} // namespace MNN

#endif /* WinogradOptFunctionHalf_hpp */
