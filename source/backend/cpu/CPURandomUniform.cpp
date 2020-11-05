//
//  CPURandomUniform.cpp
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <ctime>
#include "backend/cpu/CPURandomUniform.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
ErrorCode CPURandomUniform::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode CPURandomUniform::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    int size = output->elementSize();
    auto parameter = mOp->main_as_RandomUniform();
    int seed = parameter->seed();
    int seed1 = parameter->seed2();
    if (seed || seed1) {
        std::srand(seed || seed1);
    } else {
        std::srand(std::time(nullptr));
    }
    auto outputPtr = output->host<float>();
    for (int i = 0; i < size; i++) {
        outputPtr[i] = std::rand() / static_cast<float>(RAND_MAX);
    }
    return NO_ERROR;
}

class CPURandomUniformCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPURandomUniform(backend, op);
    }
};
REGISTER_CPU_OP_CREATOR(CPURandomUniformCreator, OpType_RandomUniform);
} // namespace MNN
