//
//  CPUEltwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUEltwise_hpp
#define CPUEltwise_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUEltwise : public Execution {
public:
    CPUEltwise(Backend *b, EltwiseType type, std::vector<float> coef);
    virtual ~CPUEltwise() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    EltwiseType mType;
    std::vector<float> mCoeff;
};

} // namespace MNN

#endif /* CPUEltwise_hpp */
