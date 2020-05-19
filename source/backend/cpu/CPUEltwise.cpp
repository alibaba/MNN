//
//  CPUEltwise.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUEltwise.hpp"
#include <math.h>
#include <string.h>
#include "core/Concurrency.h"
#include <algorithm>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPUEltwise::CPUEltwise(Backend *b, EltwiseType type, std::vector<float> coef) : Execution(b) {
    mType = type;
    mCoeff = coef;
}

ErrorCode CPUEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    const int size   = inputTensor->elementSize();
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

    auto proc = MNNMatrixProdCommon;
    switch (mType) {
        case EltwiseType_PROD:
            proc = MNNMatrixProdCommon;
            break;
        case EltwiseType_SUM:
            proc = MNNMatrixAddCommon;
            break;
        case EltwiseType_MAXIMUM:
            proc = MNNMatrixMaxCommon;
            break;
        case EltwiseType_SUB:
            proc = MNNMatrixSubCommon;
            break;
        default:
            MNN_ERROR("Don't support %d type for eltwise", mType);
            return INPUT_DATA_ERROR;
    }
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(size);
    int sizeDivide = schedule.first;
    int scheduleNumber = schedule.second;

    MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
        int start = sizeDivide * (int)tId;
        int realSize = sizeDivide;
        if (tId == scheduleNumber -1 ) {
            realSize = size - start;
        }
        if (realSize > 0) {
            auto inputT1 = inputs[1];
            proc(outputHost + start, input0Ptr + start, inputT1->host<float>() + start, realSize, 0, 0, 0, 1);
            for (int i = 2; i < inputs.size(); ++i) {
                proc(outputHost + start, outputHost + start, inputs[i]->host<float>() + start, realSize, 0, 0, 0, 1);
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUEltwiseCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto eltwiseParam = op->main_as_Eltwise();
        auto type         = eltwiseParam->type();
        std::vector<float> coeff;
        // keep compatible with old model
        if (eltwiseParam->coeff()) {
            const int size = eltwiseParam->coeff()->size();
            coeff.resize(size);
            memcpy(coeff.data(), eltwiseParam->coeff()->data(), size * sizeof(float));
        }
        return new CPUEltwise(backend, type, coeff);
    }
};
REGISTER_CPU_OP_CREATOR(CPUEltwiseCreator, OpType_Eltwise);

} // namespace MNN
