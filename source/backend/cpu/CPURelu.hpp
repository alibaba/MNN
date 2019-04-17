//
//  CPURelu.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURelu_hpp
#define CPURelu_hpp

#include "AutoStorage.h"
#include "Execution.hpp"

namespace MNN {
class CPURelu : public Execution {
public:
    CPURelu(Backend *b, float slope) : Execution(b), mSlope(slope) {
        // nothing to do
    }
    virtual ~CPURelu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mSlope;
};

class CPUPRelu : public Execution {
public:
    CPUPRelu(Backend *b, const Op *op);
    virtual ~CPUPRelu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    AutoStorage<float> mSlope;
};

class CPURelu6 : public Execution {
public:
    CPURelu6(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CPURelu6() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPURelu_hpp */
