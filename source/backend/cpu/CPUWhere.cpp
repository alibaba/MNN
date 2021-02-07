//
//  CPUWhere.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUWhere.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

ErrorCode CPUWhere::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib           = inputs[0]->buffer();
    int32_t* inputData = inputs[0]->host<int32_t>();
    auto outputData    = outputs[0]->host<int32_t>();
    auto inputTotal = inputs[0]->elementSize();

    std::vector<int32_t> trueVec;
    for (int i = 0; i < inputTotal; i++) {
        if (inputData[i] > 0) {
            trueVec.push_back(i);
        }
    }

    //MNN_ASSERT(outputs[0]->batch() == trueVec.size());
    for (int i = 0; i < trueVec.size(); i++) {
        int index = trueVec[i];
        for (int j = 0; j < ib.dimensions; j++) {
            int result    = ib.dim[j].stride == 0 ? index : index / ib.dim[j].stride;
            index         = index - result * ib.dim[j].stride;
            outputData[i * ib.dimensions + j] = result;
        }
    }
    return NO_ERROR;
}

class CPUWhereCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUWhere(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUWhereCreator, OpType_Where);
} // namespace MNN
