//
//  CPUUnpack.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUUnpack.hpp"
#include "CPUBackend.hpp"

namespace MNN {

CPUUnpack::CPUUnpack(Backend *backend, const Op *op, int axis) : Execution(backend), mAxis(axis) {
    // nothing to do
}

ErrorCode CPUUnpack::onExecute(const std::vector<MNN::Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) {
    auto &input = inputs[0]->buffer();

    int axis = mAxis;
    if (mAxis < 0) {
        axis += input.dimensions;
    }

    int beforeDim = 1;
    for (int i = 0; i < axis; i++) {
        beforeDim *= input.dim[i].extent;
    }

    const int inputBytes  = inputs[0]->getType().bytes();
    const int outputBytes = outputs[0]->getType().bytes();

    int axisStride  = inputs[0]->stride(axis) * inputBytes;
    int inputStride = inputBytes;
    if (axis > 0) {
        inputStride *= inputs[0]->stride(axis - 1);
    }
    int outputStride = outputBytes;
    if (axis > 0) {
        outputStride *= outputs[0]->stride(axis - 1);
    }

    int curPos          = 0;
    const auto srcStart = inputs[0]->host<char>();

    for (int i = 0; i < outputs.size(); i++) {
        auto srcCur = srcStart + curPos * axisStride;
        auto dstCur = outputs[i]->host<char>();

        for (int j = 0; j < beforeDim; j++) {
            auto src = srcCur + j * inputStride;
            auto dst = dstCur + j * outputStride;
            memcpy(dst, src, axisStride);
        }
        curPos += 1;
    }

    return NO_ERROR;
}

class CPUUnpackCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto unpack = op->main_as_Axis();
        return new CPUUnpack(backend, op, unpack->axis());
    }
};
REGISTER_CPU_OP_CREATOR(CPUUnpackCreator, OpType_Unpack);
} // namespace MNN
