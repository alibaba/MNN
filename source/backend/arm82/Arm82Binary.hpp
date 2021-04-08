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
#include "MNN_generated.h"

namespace MNN {
class Arm82BinaryFloat : public Execution {
public:
    Arm82BinaryFloat(Backend *b, int32_t type);
    virtual ~Arm82BinaryFloat() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int32_t mType;
    int mNeedBroadcastIndex; // -1 do not need broadcast, 0 for input0, 1 for input1
    int mTotalSize = 0;
};
} // namespace MNN

#endif /* Arm82Binary_hpp */
#endif
