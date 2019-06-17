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

CPUEltwise::CPUEltwise(Backend *b, const MNN::Op *op) : Execution(b) {
    auto eltwiseParam = op->main_as_Eltwise();
    mType             = eltwiseParam->type();

    // keep compatible with old model
    if (eltwiseParam->coeff()) {
        const int size = eltwiseParam->coeff()->size();
        mCoeff.resize(size);
        memcpy(mCoeff.data(), eltwiseParam->coeff()->data(), size * sizeof(float));
    }
}

ErrorCode CPUEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    const int size   = inputTensor->elementSize();
    auto sizeQuad    = UP_DIV(size, 4);
    auto outputSize = outputs[0]->elementSize();
    MNN_ASSERT(outputSize == size);

    auto outputTensor    = outputs[0];
    auto outputHost      = outputTensor->host<float>();
    const auto input0Ptr = inputs[0]->host<float>();

    auto coeffSize = mCoeff.size();
    bool isIdentity     = coeffSize >= 2;
    if (isIdentity) {
        // when Eltwise has coeff
        if (mCoeff[0] == 1.0f && mCoeff[1] == 0.0f) {
            memcpy(outputHost, input0Ptr, inputs[0]->size());
            return NO_ERROR;
        } else {
            return NOT_SUPPORT;
        }
    }

    auto proc = MNNMatrixProd;
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
        case EltwiseType_SUB:
            proc = MNNMatrixSub;
            break;
        default:
            MNN_ERROR("Don't support %d type for eltwise", mType);
            return INPUT_DATA_ERROR;
    }

    auto inputT1 = inputs[1];
    proc(outputHost, input0Ptr, inputT1->host<float>(), sizeQuad, 0, 0, 0, 1);
    for (int i = 2; i < inputs.size(); ++i) {
        proc(outputHost, outputHost, inputs[i]->host<float>(), sizeQuad, 0, 0, 0, 1);
    }
    return NO_ERROR;
}

class CPUEltwiesCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUEltwise(backend, op);
    }
};
REGISTER_CPU_OP_CREATOR(CPUEltwiesCreator, OpType_Eltwise);

} // namespace MNN
