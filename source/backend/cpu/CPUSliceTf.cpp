//
//  CPUSliceTf.cpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUSliceTf.hpp"
#include <cmath>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUSliceTf::CPUSliceTf(Backend *b) : MNN::Execution(b) {
    // nothing to do
}

ErrorCode CPUSliceTf::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    // these two inputs should be const
    auto begin  = inputs[1];
    auto output = outputs[0];

    const int outputDims = output->buffer().dimensions;
    if (0 == outputDims) {
        return NO_ERROR;
    }

    const int numElements = output->elementSize();
    int r, inputOffset;
    for (int offset = 0; offset < numElements; offset++) {
        r           = offset;
        inputOffset = 0;
        for (int j = 0, outputCord; j < outputDims; j++) {
            outputCord = r / output->buffer().dim[j].stride;

            // add the begin_tensor to get the input_cord
            // reuse output_cord as input_cord
            outputCord += begin->host<int32_t>()[j];

            // get input offset
            inputOffset += outputCord * input->buffer().dim[j].stride;

            r = offset % output->buffer().dim[j].stride;
        }
        output->host<int32_t>()[offset] = input->host<int32_t>()[inputOffset];
    }

    return NO_ERROR;
}

class CPUSliceTfCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs[0]->getType().bits != 32) {
            return nullptr;
        }
        return new CPUSliceTf(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSliceTfCreator, OpType_SliceTf);

} // namespace MNN
