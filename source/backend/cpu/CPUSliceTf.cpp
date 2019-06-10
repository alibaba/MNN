//
//  CPUSliceTf.cpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSliceTf.hpp"
#include <cmath>
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUSliceTf<T>::CPUSliceTf(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    // nothing to do
}

template <typename T>
ErrorCode CPUSliceTf<T>::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
        ((T *)output->buffer().host)[offset] = input->host<T>()[inputOffset];
    }

    return NO_ERROR;
}

class CPUSliceTfCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        switch (op->main_as_SliceTf()->T()) {
            case DataType_DT_INT32:
                return new CPUSliceTf<int32_t>(backend, op);
            case DataType_DT_FLOAT:
                return new CPUSliceTf<float_t>(backend, op);
            default:
                MNN_ASSERT(false); // unsupported type
                return nullptr;
        }
    }
};

REGISTER_CPU_OP_CREATOR(CPUSliceTfCreator, OpType_SliceTf);

} // namespace MNN
