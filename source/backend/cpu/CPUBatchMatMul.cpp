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
#include "core/TensorUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "compute/CommonOptFunction.h"
namespace MNN {

CPUBatchMatMul::CPUBatchMatMul(Backend* backend, bool adjX, bool adjY) : Execution(backend) {
    auto threadNumber = static_cast<CPUBackend*>(backend)->threadNumber();
    for (int i = 0; i < threadNumber; ++i) {
        Unit unit;
        unit.mMatrixA.reset(new Tensor);
        unit.mMatrixB.reset(new Tensor);
        unit.mMatrixC.reset(new Tensor);
        unit.mMatMul.reset(new CPUMatMul(backend, adjX, adjY, false));
        unit.mMatrixB->buffer().dimensions = 2;
        unit.mMatrixA->buffer().dimensions = 2;
        unit.mMatrixC->buffer().dimensions = 2;
        unit.mTempInputs = {unit.mMatrixA.get(), unit.mMatrixB.get()};
        unit.mTempOutputs = {unit.mMatrixC.get()};
        mUnits.emplace_back(std::move(unit));
    }
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
    int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= input0->length(i);
    }
    mBatch = batch;
    if (threadNumber > batch) {
        threadNumber = batch;
    }
    auto memoryPool = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    memoryPool->barrierBegin();
    std::shared_ptr<void> __a(nullptr, [memoryPool](void *) { memoryPool->barrierEnd(); });
    for (int i = 0; i < threadNumber; ++i) {
        memoryPool->beginGroup();
        std::shared_ptr<void> __b(nullptr, [memoryPool](void *) { memoryPool->endGroup(); });
        auto& unit = mUnits[i];
        unit.mMatrixA->setLength(0, input0->length(input0->dimensions()-2));
        unit.mMatrixA->setLength(1, input0->length(input0->dimensions()-1));

        unit.mMatrixB->setLength(0, input1->length(input1->dimensions()-2));
        unit.mMatrixB->setLength(1, input1->length(input1->dimensions()-1));

        unit.mMatrixC->setLength(0, output->length(output->dimensions()-2));
        unit.mMatrixC->setLength(1, output->length(output->dimensions()-1));
        
        TensorUtils::setLinearLayout(unit.mMatrixA.get());
        TensorUtils::setLinearLayout(unit.mMatrixB.get());
        TensorUtils::setLinearLayout(unit.mMatrixC.get());

        auto code = unit.mMatMul->onResize(unit.mTempInputs, unit.mTempOutputs);
    }
    return NO_ERROR;
}

ErrorCode CPUBatchMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0          = inputs[0];
    auto input1          = inputs[1];
    auto output          = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    // Fill output by zero if one of inputs is empty.
    if (input0->elementSize() == 0 || input1->elementSize() == 0) {
        ::memset(output->host<float>(), 0, output->elementSize() * core->bytes);
        return NO_ERROR;
    }
    const int dimensions = input0->dimensions();
    MNN_ASSERT(dimensions >= 3);
    const int input0Stride = input0->length(dimensions - 1) * input0->length(dimensions - 2);
    const int input1Stride = input1->length(dimensions - 1) * input1->length(dimensions - 2);
    const int outputStride = output->length(dimensions - 1) * output->length(dimensions - 2);
    auto input0Ptr   = input0->host<uint8_t>();
    auto input1Ptr   = input1->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    if (threadNumber > mBatch) {
        threadNumber = mBatch;
    }
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        auto& unit = mUnits[tId];
        for (int i = (int)tId; i < mBatch; i+=threadNumber) {
            unit.mMatrixA->buffer().host = (uint8_t*)(input0Ptr + i * input0Stride * core->bytes);
            unit.mMatrixB->buffer().host = (uint8_t*)(input1Ptr + i * input1Stride * core->bytes);
            unit.mMatrixC->buffer().host = (uint8_t*)(outputPtr + i * outputStride * core->bytes);
            unit.mMatMul->onExecute(unit.mTempInputs, unit.mTempOutputs);
        }
    }
    MNN_CONCURRENCY_END();
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
