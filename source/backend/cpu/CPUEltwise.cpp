//
//  CPUEltwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUEltwise.hpp"
#include <math.h>
#include <string.h>
#include <algorithm>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

ErrorCode CPUEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    const int size   = inputTensor->elementSize();
    auto sizeQuad    = size / 4;

    auto outputTensor = outputs[0];
    auto outputHost   = outputTensor->host<float>();
    auto proc         = MNNMatrixProd;
    switch (mType) {
        case EltwiseType_PROD:
            proc = MNNMatrixProd;
            break;
        case EltwiseType_SUM:
            proc = MNNMatrixAdd;
            break;
        case EltwiseType_MAXIMUM:
            proc = MNNMatrixMax;
            break;
        default:
            MNN_ERROR("Don't support %d type for eltwise", mType);
            return INPUT_DATA_ERROR;
    }

    auto inputT1 = inputs[1];
    proc(outputHost, inputs[0]->host<float>(), inputT1->host<float>(), sizeQuad, 0, 0, 0, 1);
    for (int i = 2; i < inputs.size(); ++i) {
        proc(outputHost, outputHost, inputs[i]->host<float>(), sizeQuad, 0, 0, 0, 1);
    }
    return NO_ERROR;
}

class CPUEltwiesCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto elt = op->main_as_Eltwise();
        return new CPUEltwise(backend, elt->type());
    }
};
REGISTER_CPU_OP_CREATOR(CPUEltwiesCreator, OpType_Eltwise);

} // namespace MNN
