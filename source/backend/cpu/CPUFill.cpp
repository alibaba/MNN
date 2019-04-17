//
//  CPUFill.cpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUFill.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUFill<T>::CPUFill(Backend *backend) : Execution(backend) {
    // nothing to do
}

template <typename T>
ErrorCode CPUFill<T>::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(0 == inputs[1]->buffer().dimensions);
    T value = inputs[1]->host<T>()[0];
    for (int i = 0; i < outputs[0]->elementSize(); i++) {
        outputs[0]->host<T>()[i] = value;
    }
    return NO_ERROR;
}

class CPUFillCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUFill<int32_t>(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUFillCreator, OpType_Fill);
} // namespace MNN
