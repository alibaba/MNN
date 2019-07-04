//
//  CPUWhere.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUWhere.hpp"
#include "CPUBackend.hpp"

namespace MNN {

ErrorCode CPUWhere::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib           = inputs[0]->buffer();
    auto& ob           = outputs[0]->buffer();
    int32_t* inputData = inputs[0]->host<int32_t>();
    auto outputData    = outputs[0]->host<int32_t>();

    std::vector<int32_t> trueVec;
    for (int i = 0; i < ob.dim[0].extent; i++) {
        if (inputData[i] > 0) {
            trueVec.push_back(i);
        }
    }

    // ob.dim[0].extent = (int)trueVec.size();
    int k = 0;
    for (int i = 0; i < trueVec.size(); i++) {
        int index = trueVec[i];
        for (int j = 0; j < ib.dimensions; j++) {
            int result    = index / ib.dim[j].stride;
            index         = index - result * ib.dim[j].stride;
            outputData[k] = result;
            k++;
        }
    }
    int defaultValue = 0;
    if (!trueVec.empty()) {
        defaultValue = trueVec[0];
    }
    for (int i = (int)trueVec.size(); i < ob.dim[0].extent; ++i) {
        outputData[i] = defaultValue;
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
