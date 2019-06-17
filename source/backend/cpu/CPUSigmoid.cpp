//
//  CPUSigmoid.cpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSigmoid.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {
ErrorCode CPUSigmoid::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputData  = inputs[0]->host<float>();
    auto outputData = outputs[0]->host<float>();

    const int dataSize = outputs[0]->elementSize();
    MNNExp(outputData, inputData, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        outputData[i] = 1.0f / (1.0f + outputData[i]);
    }
    return NO_ERROR;
}

class CPUSigmoidCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUSigmoid(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSigmoidCreator, OpType_Sigmoid);
} // namespace MNN
