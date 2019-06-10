//
//  CPUPack.cpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUPack.hpp"
#include "CPUBackend.hpp"

namespace MNN {

CPUPack::CPUPack(Backend *backend, const Op *op, DataType type, int axis)
    : Execution(backend), mDataType(type), mAxis(axis) {
    // nothing to do
}

template <typename T>
ErrorCode CPUPack::MNNPackLayerForward(const std::vector<MNN::Tensor *> &inputs,
                                       const std::vector<MNN::Tensor *> &outputs) {
    auto output                = outputs[0];
    const int outputDimensions = output->buffer().dimensions;
    auto mN                    = inputs.size();

    if (mAxis == 0) {
        auto *dstPtr = outputs[0]->buffer().host;
        for (int i = 0; i < mN; i++) {
            auto inputX    = inputs[i];
            auto sourcePtr = inputX->buffer().host;
            memcpy(dstPtr, sourcePtr, inputX->size());
            dstPtr += inputX->size();
        }
    } else {
        int outputDataCount = 1;
        for (int i = 0; i < outputDimensions; i++) {
            outputDataCount *= output->buffer().dim[i].extent;
        }

        int r;
        for (int offset = 0, cordOnAxis = 0; offset < outputDataCount; offset++) {
            r               = offset;
            int inputOffset = 0;
            for (int i = 0, j = 0, cord; i < outputDimensions; i++) {
                cord          = r / output->buffer().dim[i].stride;
                r             = r % output->buffer().dim[i].stride;

                if (i != mAxis) {
                    inputOffset += (cord * inputs[0]->buffer().dim[j++].stride);
                } else {
                    cordOnAxis = cord;
                }
            }

            ((T *)output->buffer().host)[offset] = ((T *)inputs[cordOnAxis]->buffer().host)[inputOffset];
        }
    }

    return NO_ERROR;
}

ErrorCode CPUPack::onExecute(const std::vector<MNN::Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    if (inputs.size() == 1) {
        ::memcpy(output->buffer().host, input->buffer().host, input->size());
        return NO_ERROR;
    }

    if (mDataType == DataType_DT_INT32) {
        return MNNPackLayerForward<int32_t>(inputs, outputs);
    } else if (mDataType == DataType_DT_FLOAT) {
        return MNNPackLayerForward<float>(inputs, outputs);
    }

    return NO_ERROR;
}

class CPUPackCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto pack = op->main_as_PackParam();
        return new CPUPack(backend, op, pack->dataType(), pack->axis());
    }
};
REGISTER_CPU_OP_CREATOR(CPUPackCreator, OpType_Pack);
} // namespace MNN
