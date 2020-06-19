//
//  Arm82TensorConverter.cpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "backend/arm82/Arm82TensorConverter.hpp"
#include "backend/arm82/Arm82Backend.hpp"

namespace MNN {

ErrorCode Arm82TensorConverter::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    auto arm82Backend = static_cast<Arm82Backend*>(backend());

    arm82Backend->onCopyBuffer(input, output);

    return NO_ERROR;
}

class Arm82TensorConverterCreator : public Arm82Backend::Arm82Creator {
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new Arm82TensorConverter(backend);
    }
};

// REGISTER_ARM82_OP_CREATOR(OpType_ConvertTensor, Arm82TensorConverterCreator);

} // namespace MNN

#endif
