//
//  Arm82Relu.hpp
//  MNN
//
//  Created by MNN on 2020/2/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82Relu_hpp
#define Arm82Relu_hpp

#include "core/Execution.hpp"

namespace MNN {

class Arm82Relu : public Execution { 
public:
    Arm82Relu(Backend *backend, const Op *op);
    virtual ~Arm82Relu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mSlope = 0.0;
    int mThreadNumbers;
};

class Arm82PRelu : public Execution {
public:
    Arm82PRelu(Backend *backend, const Op *op);
    virtual ~Arm82PRelu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mSlope;
    int mThreadNumbers;
};

} // namespace MNN

#endif /* Arm82Relu_hpp */
#endif
