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

static void bcastImpl(int curDim, int* flag, const std::vector<int>& dimElements, std::vector<int>& dimBroadCastStride, const int bytes, const Tensor* input,
                      Tensor* output) {
    if (curDim < 0) {
        return;
    }
    int bcastNum = output->length(curDim) / input->length(curDim);
    if (bcastNum == 1) {
        bcastImpl(curDim - 1, flag, dimElements, dimBroadCastStride, bytes, input, output);
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
        auto dstCurStart = dstStart + i * dimBroadCastStride[curDim] * bytes;

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

    bcastImpl(curDim - 1, flag, dimElements, dimBroadCastStride, bytes, input, output);
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

    std::vector<int> dimBroadCastStride(dimension, 1);
    for(int i = dimension - 1; i >= 0; --i){
        int bcastNum = output->length(i) / input->length(i);
        if(bcastNum == 1){
            dimBroadCastStride[i] = output->length(i) * output->stride(i);
        }else{
            for(int j = i - 1; j >= 0; --j){
                int bcastNum = output->length(j) / input->length(j);
                if(bcastNum == 1){
                    dimBroadCastStride[i] = output->stride(j);
                    break;
                }
            }
        }
    }

    int flag = 0;
    bcastImpl(dimension - 1, &flag, dimElements, dimBroadCastStride, bytes, input, output);
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
