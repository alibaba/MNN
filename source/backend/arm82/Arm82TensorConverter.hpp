//
//  Arm82TensorConverter.hpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82TensorConverter_hpp
#define Arm82TensorConverter_hpp

#include "core/Execution.hpp"

namespace MNN {

class Arm82TensorConverter : public Execution {
public:
    Arm82TensorConverter(Backend* backend) : Execution(backend) {
    }
    virtual ~Arm82TensorConverter() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
};

} // namespace MNN

#endif
#endif