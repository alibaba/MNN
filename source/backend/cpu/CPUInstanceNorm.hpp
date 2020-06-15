//
//  CPUInstanceNorm.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInstanceNorm_hpp
#define CPUInstanceNorm_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUInstanceNorm : public Execution {
public:
    CPUInstanceNorm(Backend *backend, const MNN::Op *op);
    virtual ~CPUInstanceNorm() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    AutoStorage<float> mScale;
    AutoStorage<float> mBias;
    float mEpsilon;
};
} // namespace MNN

#endif /* CPUInstanceNorm_hpp */
