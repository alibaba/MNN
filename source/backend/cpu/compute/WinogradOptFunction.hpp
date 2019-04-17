//
//  WinogradOptFunction.hpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradOptFunction_hpp
#define WinogradOptFunction_hpp

#include <stdint.h>
#include <stdio.h>

namespace MNN {
class WinogradFunction {
public:
    static void productLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);
    static void productRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);

    static int getPreferNumber();

    typedef void (*TransformFunc)(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep);

    /*Use the generator with interp 0.5*/
    static TransformFunc chooseSourceTransform(int k, int w);
    static TransformFunc chooseDestTransform(int k, int h);
};
} // namespace MNN

#endif /* WinogradOptFunction_hpp */
