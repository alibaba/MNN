//
//  CPUQuantizedLogistic.cpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "backend/cpu/CPUQuantizedLogistic.hpp"
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/OptimizedComputer.hpp"

namespace MNN {

CPUQuantizedLogistic::CPUQuantizedLogistic(Backend *backend, const Op *op) : Execution(backend) {
    mLogisticParam = op->main_as_QuantizedLogistic();
}

ErrorCode CPUQuantizedLogistic::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size() && 1 == outputs.size());
    MNN_ASSERT(0 == mLogisticParam->outputQuantizedParam()->zeroPoint() &&
               1. / 256 == mLogisticParam->outputQuantizedParam()->scale());

    static constexpr int kInputIntegerBits = 4;
    const double inputRealMultiplier =
        mLogisticParam->inputQuantizedParam()->scale() * static_cast<double>(1 << (31 - kInputIntegerBits));
    QuantizeMultiplierGreaterThanOne(inputRealMultiplier, &mInputMultiplier, &mInputLeftShift);
    mInputZeroPoint = mLogisticParam->inputQuantizedParam()->zeroPoint();
    mInputRangeRadius = CalculateInputRadius(kInputIntegerBits, mInputLeftShift);
    return NO_ERROR;
}

ErrorCode CPUQuantizedLogistic::onExecute(const std::vector<MNN::Tensor *> &inputs,
                                          const std::vector<MNN::Tensor *> &outputs) {
    auto input = inputs[0], output = outputs[0];
    std::vector<int> inputDims, outputDims;
    for (int i = 0; i < input->buffer().dimensions; i++) {
        inputDims.push_back(input->buffer().dim[i].extent);
    }
    for (int i = 0; i < output->buffer().dimensions; i++) {
        outputDims.push_back(output->buffer().dim[i].extent);
    }

    Optimized::Logistic(input->host<uint8_t>(), inputDims, mInputZeroPoint,
                        mInputRangeRadius, mInputMultiplier, mInputLeftShift, output->host<uint8_t>(), outputDims);

    return NO_ERROR;
}

class CPUQuantizedLogisticCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUQuantizedLogistic(backend, op);
    }
};
} // namespace MNN
#endif
namespace MNN {
REGISTER_CPU_OP_CREATOR_OLD(CPUQuantizedLogisticCreator, OpType_QuantizedLogistic);
};
