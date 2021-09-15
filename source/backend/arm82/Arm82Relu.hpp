//
//  Arm82Relu.hpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Relu_hpp
#define Arm82Relu_hpp
#include <stddef.h>

namespace MNN {

class Arm82Relu {
public:
    static void reluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);
};

} // namespace MNN

#endif /* Arm82Relu_hpp */
#endif
