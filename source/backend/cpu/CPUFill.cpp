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

CPUFill::CPUFill(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode CPUFill::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(0 == inputs[1]->buffer().dimensions);
    auto bytes = outputs[0]->getType().bytes();
    auto size = outputs[0]->elementSize();
    switch (bytes) {
        case 1: {
            auto value = inputs[1]->host<uint8_t>()[0];
            auto outputPtr = outputs[0]->host<uint8_t>();
            ::memset(outputPtr, value, size);
            break;
        }
        case 2: {
            auto value = inputs[1]->host<uint16_t>()[0];
            auto outputPtr = outputs[0]->host<uint16_t>();
            for (int i=0; i<size; ++i) {
                outputPtr[i] = value;
            }
            break;
        }
        case 4: {
            auto value = inputs[1]->host<uint32_t>()[0];
            auto outputPtr = outputs[0]->host<uint32_t>();
            for (int i=0; i<size; ++i) {
                outputPtr[i] = value;
            }
            break;
        }
        default:
            return INPUT_DATA_ERROR;
    }
    
    return NO_ERROR;
}

class CPUFillCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUFill(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUFillCreator, OpType_Fill);
} // namespace MNN
