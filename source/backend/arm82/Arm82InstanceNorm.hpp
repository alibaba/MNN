//
//  Arm82InstanceNorm.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82InstanceNorm_hpp
#define Arm82InstanceNorm_hpp

#include "Arm82Backend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class Arm82InstanceNorm : public Execution {
public:
    Arm82InstanceNorm(Backend *backend, const MNN::Op *op);
    virtual ~Arm82InstanceNorm() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    AutoStorage<FLOAT16> mScale;
    AutoStorage<FLOAT16> mBias;
    FLOAT16 mEpsilon;
};
} // namespace MNN

#endif /* Arm82InstanceNorm_hpp */
#endif
