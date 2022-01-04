//
//  CPURandomUniform.cpp
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <random>
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
    auto outputPtr = output->host<float>();
    std::uniform_real_distribution<float> distribution(parameter->low(),parameter->high());
    int seed = parameter->seed();
    int seed1 = parameter->seed2();
    if (seed || seed1) {
        std::mt19937 generator(seed || seed1);
        for (int i = 0; i < size; i++) {
            outputPtr[i] = distribution(generator);
        }
    } else {
        std::default_random_engine generator;
        for (int i = 0; i < size; i++) {
            outputPtr[i] = distribution(generator);
        }
    }
    return NO_ERROR;
}

ErrorCode CPURandomNormal::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode CPURandomNormal::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    int size = output->elementSize();
    auto parameter = mOp->main_as_RandomUniform();
    auto outputPtr = output->host<float>();
    // RandomUniform and RandomNormal use same param table. low -> mean, high -> scale
    std::normal_distribution<float> distribution(parameter->low(),parameter->high());
    int seed = parameter->seed();
    int seed1 = parameter->seed2();
    if (seed || seed1) {
        std::mt19937 generator(seed || seed1);
        for (int i = 0; i < size; i++) {
            outputPtr[i] = distribution(generator);
        }
    } else {
        std::default_random_engine generator;
        for (int i = 0; i < size; i++) {
            outputPtr[i] = distribution(generator);
        }
    }
    return NO_ERROR;
}

class CPURandomCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->type() == OpType_RandomUniform) {
            return new CPURandomUniform(backend, op);
        } else {
            return new CPURandomNormal(backend, op);
        }
    }
};
REGISTER_CPU_OP_CREATOR(CPURandomCreator, OpType_RandomUniform);
REGISTER_CPU_OP_CREATOR(CPURandomCreator, OpType_RandomNormal);
} // namespace MNN
