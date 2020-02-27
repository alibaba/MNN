//
//  CPULinSpace.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPULinSpace.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
ErrorCode CPULinSpace::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 3);
    MNN_ASSERT(outputs.size() == 1);
    const float start = inputs[0]->host<float>()[0];
    const float stop = inputs[1]->host<float>()[0];
    const int num = inputs[2]->host<int32_t>()[0];
    MNN_ASSERT(num > 0);

    float* outputData = outputs[0]->host<float>();

    if (num == 1) {
        outputData[0] = start;
        return NO_ERROR;
    }

    if (num == 2) {
        outputData[0] = start;
        outputData[1] = stop;
        return NO_ERROR;
    }
    
    // make sure that start with the first and end with the last.
    outputData[0] = start;
    outputData[num - 1] = stop;
    const float step = (stop - start) / (num - 1);
    for (int i = 1; i < num - 1; ++i) {
        outputData[i] = start + i * step;
    }

    return NO_ERROR;
}

class CPULinSpaceCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPULinSpace(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPULinSpaceCreator, OpType_LinSpace);
} // namespace MNN
