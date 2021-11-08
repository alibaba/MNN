//
//  CPUBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUBatchMatMul.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "math/Matrix.hpp"

namespace MNN {

CPUBatchMatMul::CPUBatchMatMul(Backend* backend, bool adjX, bool adjY) : Execution(backend) {
    mMatMul.reset(new CPUMatMul(backend, adjX, adjY, true));
}

ErrorCode CPUBatchMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0          = inputs[0];
    auto input1          = inputs[1];
    auto output          = outputs[0];
    // Fill output by zero if one of inputs is empty.
    if (input0->elementSize() == 0 || input1->elementSize() == 0) {
        return NO_ERROR;
    }
    auto dimensions = input0->dimensions();
    mMatrixA.reset(Tensor::createDevice<float>({input0->length(input0->dimensions()-2), input0->length(input0->dimensions()-1)}));
    mMatrixB.reset(Tensor::createDevice<float>({input1->length(input1->dimensions()-2), input1->length(input0->dimensions()-1)}));
    mMatrixC.reset(Tensor::createDevice<float>({output->length(output->dimensions()-2), output->length(output->dimensions()-1)}));
    mTempInputs = {mMatrixA.get(), mMatrixB.get()};
    mTempOutputs = {mMatrixC.get()};
    auto res = backend()->onAcquireBuffer(mMatrixA.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mMatrixB.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mMatrixC.get(), Backend::DYNAMIC);

    if (!res) {
        return OUT_OF_MEMORY;
    }
    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= input0->length(i);
    }
    mBatch = batch;
    auto code = mMatMul->onResize(mTempInputs, mTempOutputs);
    backend()->onReleaseBuffer(mMatrixA.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mMatrixB.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mMatrixC.get(), Backend::DYNAMIC);
    return code;
}

ErrorCode CPUBatchMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0          = inputs[0];
    auto input1          = inputs[1];
    auto output          = outputs[0];
    // Fill output by zero if one of inputs is empty.
    if (input0->elementSize() == 0 || input1->elementSize() == 0) {
        ::memset(output->host<float>(), 0, output->size());
        return NO_ERROR;
    }
    const int dimensions = input0->dimensions();
    MNN_ASSERT(dimensions >= 3);
    const int input0Stride = input0->length(dimensions - 1) * input0->length(dimensions - 2);
    const int input1Stride = input1->length(dimensions - 1) * input1->length(dimensions - 2);
    const int outputStride = output->length(dimensions - 1) * output->length(dimensions - 2);
    const auto input0Ptr   = input0->host<float>();
    const auto input1Ptr   = input1->host<float>();
    float* const outputPtr = output->host<float>();

    for (int i = 0; i < mBatch; ++i) {
        ::memcpy(mMatrixA->host<float>(), input0Ptr + i * input0Stride, input0Stride * sizeof(float));
        ::memcpy(mMatrixB->host<float>(), input1Ptr + i * input1Stride, input1Stride * sizeof(float));
        mMatMul->onExecute(mTempInputs, mTempOutputs);
        ::memcpy(outputPtr + i * outputStride, mMatrixC->host<float>(), outputStride * sizeof(float));
    }
    return NO_ERROR;
}

class CPUBatchMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBatchMatMul(backend, op->main_as_BatchMatMulParam()->adjX(), op->main_as_BatchMatMulParam()->adjY());
    }
};

REGISTER_CPU_OP_CREATOR(CPUBatchMatMulCreator, OpType_BatchMatMul);

} // namespace MNN
