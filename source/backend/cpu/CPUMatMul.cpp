//
//  CPUMatMul.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUMatMul.hpp"
#include "CPUBackend.hpp"
#include "Matrix.hpp"

namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB) {
    // nothing to do
}

ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto w1         = inputs[1]->length(1);
    auto h1         = inputs[1]->length(0);
    mFunction.clear();
    std::shared_ptr<Tensor> transposeA;
    if (mTransposeA) {
        transposeA.reset(Tensor::createDevice<float>({w0, h0}));
        auto success = backend()->onAcquireBuffer(transposeA.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        mFunction.emplace_back([A, transposeA]() { Math::Matrix::transpose(transposeA.get(), A); });
        A = transposeA.get();
    }
    std::shared_ptr<Tensor> transposeB;
    if (mTransposeB) {
        transposeB.reset(Tensor::createDevice<float>({w1, h1}));
        auto success = backend()->onAcquireBuffer(transposeB.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        mFunction.emplace_back([B, transposeB]() { Math::Matrix::transpose(transposeB.get(), B); });
        B = transposeB.get();
    }
    mFunction.emplace_back([A, B, C]() { Math::Matrix::multi(C, A, B); });
    if (nullptr != transposeA) {
        backend()->onReleaseBuffer(transposeA.get(), Backend::DYNAMIC);
    }
    if (nullptr != transposeB) {
        backend()->onReleaseBuffer(transposeB.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    for (auto& f : mFunction) {
        f();
    }
    return NO_ERROR;
}

class CPUMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new CPUMatMul(backend, param->transposeA(), param->transposeB());
    }
};

REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);

} // namespace MNN
