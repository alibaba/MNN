//
//  CPUBroadcastTo.cpp
//  MNN
//
//  Created by MNN on 2019/12/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBroadcastTo.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

static void bcastImpl(int curDim, int* flag, const std::vector<int>& dimElements, const int bytes, const Tensor* input,
                      Tensor* output) {
    if (curDim < 0) {
        return;
    }
    int bcastNum = output->length(curDim) / input->length(curDim);
    if (bcastNum == 1) {
        bcastImpl(curDim - 1, flag, dimElements, bytes, input, output);
        return;
    }

    const auto srcStart = input->host<char>();
    auto dstStart       = output->host<char>();

    // flag == 0, represent the first broadcast
    for (int i = 0; i < dimElements[curDim]; ++i) {
        int k = 0;
        if (*flag) {
            k = 1;
        }
        auto dstCurStart = dstStart + i * output->length(curDim) * output->stride(curDim) * bytes;

        for (; k < bcastNum; ++k) {
            auto copyedPtr = dstCurStart + k * output->stride(curDim) * bytes;
            if (*flag == 0) {
                memcpy(copyedPtr, srcStart + i * input->stride(curDim) * bytes, input->stride(curDim) * bytes);
            } else {
                memcpy(copyedPtr, dstCurStart, output->stride(curDim) * bytes);
            }
        }
    }
    *flag = 1;

    bcastImpl(curDim - 1, flag, dimElements, bytes, input, output);
}

ErrorCode CPUBroadcastTo::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input          = inputs[0];
    auto output         = outputs[0];
    const int dimension = input->dimensions();
    if (input->elementSize() == output->elementSize()) {
        ::memcpy(output->host<void>(), input->host<void>(), input->size());
        return NO_ERROR;
    }

    auto bytes = input->getType().bytes();

    std::vector<int> dimElements(dimension, 1);
    for (int i = 1; i < dimension; ++i) {
        dimElements[i] = dimElements[i - 1] * input->length(i - 1);
    }

    int flag = 0;
    bcastImpl(dimension - 1, &flag, dimElements, bytes, input, output);
    return NO_ERROR;
}

class CPUBroadcastToCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBroadcastTo(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUBroadcastToCreator, OpType_BroadcastTo);

} // namespace MNN
