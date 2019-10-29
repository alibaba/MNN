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
#include "compute/CommonOptFunction.h"
#include "compute/StrassenMatmulComputor.hpp"
#include "Macro.h"
namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB) {
    // nothing to do
}

ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto APtr = A->host<float>();
    auto BPtr = B->host<float>();
    Tensor* C       = outputs[0];
    auto CPtr = C->host<float>();
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    mFunction.clear();
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    std::shared_ptr<Tensor> AT(Tensor::createDevice<float>({UP_DIV(l, 4), e, 4}));
    std::shared_ptr<Tensor> BT(Tensor::createDevice<float>({UP_DIV(h, 4), UP_DIV(l, 4), 16}));
    std::shared_ptr<Tensor> CT(Tensor::createDevice<float>({UP_DIV(h, 4), e, 4}));
    std::shared_ptr<Tensor> BTemp;
    if (l % 4 != 0) {
        BTemp.reset(Tensor::createDevice<float>({UP_DIV(h, 4), l, 4}));
        auto res = backend()->onAcquireBuffer(BTemp.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    auto res = backend()->onAcquireBuffer(BT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto BTPtr = BT->host<float>();
    float* BTempPtr = BTPtr;
    if(l % 4 != 0) {
        BTempPtr = BTemp->host<float>();
    }
    if (mTransposeB) {
        mFunction.emplace_back([BPtr, BTempPtr, l, h] {
            MNNPackC4(BTempPtr, BPtr, l, h);
        });
    } else {
        mFunction.emplace_back([BPtr, BTempPtr, l, h] {
            MNNTensorConvertNHWCToNC4HW4(BTempPtr, BPtr, l, h);
        });
    }
    if (l % 4 != 0) {
        mFunction.emplace_back([BTPtr, BTempPtr, l, h] {
            auto hC4 = UP_DIV(h, 4);
            auto lC4 = UP_DIV(l, 4);
            for (int y=0; y<hC4; ++y) {
                auto dst = BTPtr + 16*lC4 * y;
                auto src = BTempPtr + 4 * l * y;
                ::memcpy(dst, src, 4*l*sizeof(float));
                ::memset(dst+4*l, 0, (lC4*4-l) * sizeof(float));
            }
        });
        backend()->onReleaseBuffer(BTemp.get(), Backend::DYNAMIC);
    }
    res = backend()->onAcquireBuffer(AT.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(CT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto ATPtr = AT->host<float>();
    if (mTransposeA) {
        mFunction.emplace_back([ATPtr, APtr, e, l]() {
            MNNPackC4(ATPtr, APtr, e, l);
        });
    } else {
        mFunction.emplace_back([ATPtr, APtr, e, l]() {
            MNNTensorConvertNHWCToNC4HW4(ATPtr, APtr, e, l);
        });
    }
    std::shared_ptr<StrassenMatrixComputor> computor(new StrassenMatrixComputor(backend()));

    auto code = computor->onEncode({AT.get(), BT.get()}, {CT.get()});
    if (NO_ERROR != code) {
        return code;
    }
    auto CTPtr = CT->host<float>();
    mFunction.emplace_back([computor, CPtr, CTPtr, e, h]() {
        computor->onExecute();
        MNNTensorConvertNC4HW4ToNHWC(CPtr, CTPtr, e, h);
    });
    backend()->onReleaseBuffer(AT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(BT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(CT.get(), Backend::DYNAMIC);
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
