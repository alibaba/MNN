//
//  CPUMatMul.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUMatMul.hpp"
#include "CPUBackend.hpp"
#include "Matrix.hpp"

namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB) {
    // nothing to do
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    Math::Matrix::multi(output, input0, input1);
    return NO_ERROR;
}

class CPUMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new CPUMatMul(backend, param->transposeA(), param->transposeB());
    }
};

REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);

} // namespace MNN
