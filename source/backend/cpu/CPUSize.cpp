//
//  CPUSize.cpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSize.hpp"
#include "CPUBackend.hpp"

namespace MNN {

template <typename T>
CPUSize<T>::CPUSize(Backend *backend, const Op *op) : Execution(backend) {
    // nothing to do
}

template <typename T>
ErrorCode CPUSize<T>::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int count = 1;
    for (int i = 0; i < inputs[0]->buffer().dimensions; i++) {
        count *= inputs[0]->buffer().dim[i].extent;
    }
    outputs[0]->host<T>()[0] = count;
    return NO_ERROR;
}

class CPUSizeCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUSize<int32_t>(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSizeCreator, OpType_Size);
} // namespace MNN
