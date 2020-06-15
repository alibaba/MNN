//
//  CPUTanh.cpp
//  MNN
//
//  Created by MNN on 2018/08/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUTanh.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

ErrorCode CPUTanh::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputData  = inputs[0]->host<float>();
    auto outputData = outputs[0]->host<float>();

    const int dataSize = outputs[0]->elementSize();
    MNNTanh(outputData, inputData, dataSize);
    return NO_ERROR;
}

class CPUTanhCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUTanh(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUTanhCreator, OpType_TanH);
} // namespace MNN
