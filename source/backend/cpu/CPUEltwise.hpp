//
//  CPUEltwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUEltwise_hpp
#define CPUEltwise_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUEltwise : public Execution {
public:
    CPUEltwise(Backend *b, MNN::EltwiseType type) : Execution(b), mType(type) {
        // nothing to do
    }
    virtual ~CPUEltwise() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    EltwiseType mType;
};

} // namespace MNN

#endif /* CPUEltwise_hpp */
