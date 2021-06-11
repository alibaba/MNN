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

namespace MNN {

CPUEltwise::CPUEltwise(Backend *b, EltwiseType type, std::vector<float> coef) : Execution(b) {
    mType = type;
    mCoeff = coef;
}

ErrorCode CPUEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    const int size   = static_cast<CPUBackend*>(backend())->getTensorSize(inputTensor);
    auto core = static_cast<CPUBackend*>(backend())->functions();

    auto outputTensor    = outputs[0];
    auto outputHost      = outputTensor->host<uint8_t>();
    const auto input0Ptr = inputs[0]->host<uint8_t>();
    const auto input1Ptr = inputs[1]->host<uint8_t>();

    auto coeffSize = mCoeff.size();
    bool isIdentity     = coeffSize >= 2;
    if (isIdentity) {
        // when Eltwise has coeff
        if (mCoeff[0] == 1.0f && mCoeff[1] == 0.0f) {
            memcpy(outputHost, input0Ptr, size * core->bytes);
            return NO_ERROR;
        } else {
            return NOT_SUPPORT;
        }
    }
    int opType = -1;

    switch (mType) {
        case EltwiseType_PROD:
            opType = BinaryOpOperation_MUL;
            break;
        case EltwiseType_SUM:
            opType = BinaryOpOperation_ADD;
            break;
        case EltwiseType_MAXIMUM:
            opType = BinaryOpOperation_MAXIMUM;
            break;
        case EltwiseType_SUB:
            opType = BinaryOpOperation_SUB;
            break;
        default:
            MNN_ERROR("Don't support %d type for eltwise", mType);
            return INPUT_DATA_ERROR;
    }
    auto proc = core->MNNSelectBinaryFunctionForFloat(opType);
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
            auto inp0 = input0Ptr + start * core->bytes;
            auto inp1 = input1Ptr + start * core->bytes;
            auto out = outputHost + start * core->bytes;

            proc(out, inp0, inp1, realSize, -1);
            for (int i = 2; i < inputs.size(); ++i) {
                proc(out, out, inputs[i]->host<uint8_t>() + start * core->bytes, realSize, -1);
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
