//
//  Arm82Unary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Unary_hpp
#define Arm82Unary_hpp

#include "core/Execution.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
class Arm82Unary {
public:
    static MNNUnaryExecute select(int type, int precision);
};
} // namespace MNN
#endif /* Arm82Unary_hpp */
#endif
