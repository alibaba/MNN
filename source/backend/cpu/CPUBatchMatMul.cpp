//
//  CPUBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBatchMatMul.hpp"
#include "CPUBackend.hpp"
#include "Matrix.hpp"

namespace MNN {

CPUBatchMatMul::CPUBatchMatMul(const Op* op, Backend* backend) : Execution(backend) {
    // nothing to do
}

ErrorCode CPUBatchMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0          = inputs[0];
    auto input1          = inputs[1];
    auto output          = outputs[0];
    const int dimensions = input0->dimensions();

    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= input0->length(i);
    }
    mBatch = batch;

    std::vector<int> dimSizes(2);

    dimSizes[0] = input0->length(dimensions - 2);
    dimSizes[1] = input0->length(dimensions - 1);
    mMatrixA.reset(Tensor::createDevice<float>(dimSizes));

    dimSizes[0] = input1->length(dimensions - 2);
    dimSizes[1] = input1->length(dimensions - 1);
    mMatrixB.reset(Tensor::createDevice<float>(dimSizes));

    dimSizes[0] = output->length(dimensions - 2);
    dimSizes[1] = output->length(dimensions - 1);
    mMatrixC.reset(Tensor::createDevice<float>(dimSizes));

    return NO_ERROR;
}

ErrorCode CPUBatchMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0          = inputs[0];
    auto input1          = inputs[1];
    auto output          = outputs[0];
    const int dimensions = input0->dimensions();
    MNN_ASSERT(dimensions >= 3);
    const int input0Stride = input0->stride(dimensions - 3);
    const int input1Stride = input1->stride(dimensions - 3);
    const int outputStride = output->stride(dimensions - 3);
    const auto input0Ptr   = input0->host<float>();
    const auto input1Ptr   = input1->host<float>();
    float* const outputPtr = output->host<float>();

    for (int i = 0; i < mBatch; ++i) {
        mMatrixA->buffer().host = reinterpret_cast<uint8_t*>(input0Ptr + i * input0Stride);
        mMatrixB->buffer().host = reinterpret_cast<uint8_t*>(input1Ptr + i * input1Stride);
        mMatrixC->buffer().host = reinterpret_cast<uint8_t*>(outputPtr + i * outputStride);
        Math::Matrix::multi(mMatrixC.get(), mMatrixA.get(), mMatrixB.get());
    }
    return NO_ERROR;
}

class CPUBatchMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBatchMatMul(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUBatchMatMulCreator, OpType_BatchMatMul);

} // namespace MNN
