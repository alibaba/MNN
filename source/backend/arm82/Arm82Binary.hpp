//
//  Arm82Binary.hpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2021, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Binary_hpp
#define Arm82Binary_hpp

#include "core/Execution.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
namespace MNN {
class Arm82BinaryFloat {
public:
    static MNNBinaryExecute select(int32_t type);
};
} // namespace MNN

#endif /* Arm82Binary_hpp */
#endif
