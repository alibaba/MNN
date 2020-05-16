//
//  CPUMatMul.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUMatMul.hpp"
#include "CPUBackend.hpp"
#include "math/Matrix.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB, bool multiThread)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB), mSupportMultiThread(multiThread) {
}
ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    mE = e;
    mH = h;
    mL = l;
    mAPtr = inputs[0]->host<float>();
    mBPtr = inputs[1]->host<float>();
    mCPtr = outputs[0]->host<float>();
    int eU, hU, lU;
    MNNGetMatMulPackMode(&eU, &lU, &hU);
    auto eP = UP_DIV(e, eU);
    auto hP = UP_DIV(h, hU);
    auto lP = UP_DIV(l, lU);

    std::shared_ptr<Tensor> APack(Tensor::createDevice<float>({eP, lP, eU * lU}));
    std::shared_ptr<Tensor> BPack(Tensor::createDevice<float>({hP, lP, hU * lU}));
    std::shared_ptr<Tensor> CPack(Tensor::createDevice<float>({eP, hP, eU * hU}));
    mAPack = APack;
    mBPack = BPack;
    mCPack = CPack;
    backend()->onAcquireBuffer(APack.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(BPack.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(CPack.get(), Backend::DYNAMIC);

    backend()->onReleaseBuffer(APack.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(BPack.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(CPack.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

extern "C" {
void _AVX_MNNGemm16x4(float* C, const float* A, const float* B, size_t e, size_t l, size_t h);
}
static void _packMatMul(float* C, const float* A, const float* B, int e, int l, int h) {
    _AVX_MNNGemm16x4(C, A, B, e, l, h);
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto APtr = mAPtr;
    auto BPtr = mBPtr;
    auto CPtr = mCPtr;
    auto e = mE;
    auto h = mH;
    auto l = mL;
    int eU, hU, lU;
    MNNGetMatMulPackMode(&eU, &lU, &hU);
    auto eP = UP_DIV(e, eU);
    auto hP = UP_DIV(h, hU);
    auto lp = UP_DIV(l, lU);
    
    auto APPtr = mAPack->host<float>();
    auto BPPtr = mBPack->host<float>();
    auto CPPtr = mCPack->host<float>();
    MNNPackForMatMul_A(APPtr, APtr, e, l, mTransposeA);
    MNNPackForMatMul_B(BPPtr, BPtr, h, l, mTransposeB);
    _packMatMul(CPPtr, APPtr, BPPtr, eP, lp, hP);
    MNNUnpackForMatMul_C(CPtr, CPPtr, e, h);
    return NO_ERROR;
}


class CPUMultiMatMul : public Execution {
public:
    CPUMultiMatMul(Backend *backend, bool transposeA, bool tranposeB) : Execution(backend) {
        mMatMul.reset(new CPUMatMul(backend, transposeA, tranposeB, true));
    }
    virtual ~CPUMultiMatMul() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input0          = inputs[0];
        auto input1          = inputs[1];
        auto output          = outputs[0];
        auto i0Dim = input0->dimensions();
        auto i1Dim = input1->dimensions();
        auto o0Dim = output->dimensions();
        const int input0Stride = input0->length(i0Dim - 1) * input0->length(i0Dim - 2);
        const int input1Stride = input1->length(i1Dim - 1) * input1->length(i1Dim - 2);
        const int outputStride = output->length(o0Dim - 1) * output->length(o0Dim - 2);
        // Compute BroastCast Dims
        auto dimOffset = o0Dim - 2;
        const int maxDimensions = dimOffset;
        std::vector<int> outputStrides(maxDimensions);
        std::vector<int> input0Strides(maxDimensions, 0);
        std::vector<int> input1Strides(maxDimensions, 0);
        auto i0Offset = output->dimensions() - input0->dimensions();
        auto i1Offset = output->dimensions() - input1->dimensions();
        int totalSize = 1;
        int i0Size = 1;
        int i1Size = 1;
        for (int i = maxDimensions - 1; i >=0 ; --i) {
            outputStrides[i] = totalSize;
            totalSize *= output->length(i);
            if (i >= i0Offset && input0->length(i - i0Offset) > 1) {
                input0Strides[i] = i0Size;
                i0Size *= input0->length(i - i0Offset);
            }
            if (i >= i1Offset && input1->length(i - i1Offset) > 1) {
                input1Strides[i] = i1Size;
                i1Size *= input1->length(i - i1Offset);
            }
        }
        const auto input0Ptr   = input0->host<float>();
        const auto input1Ptr   = input1->host<float>();
        float* const outputPtr = output->host<float>();
        for (int index = 0; index < totalSize; ++index) {
            // Unrool the cords
            auto c = index;
            i0Offset = 0;
            i1Offset = 0;
            for (int i=0; i<maxDimensions; ++i) {
                auto cord = c / outputStrides[i];
                i0Offset += input0Strides[i] * cord;
                i1Offset += input1Strides[i] * cord;
                c = c % outputStrides[i];
            }
            ::memcpy(mMatrixA->host<float>(), input0Ptr + i0Offset * input0Stride, input0Stride * sizeof(float));
            ::memcpy(mMatrixB->host<float>(), input1Ptr + i1Offset * input1Stride, input1Stride * sizeof(float));
            mMatMul->onExecute(mTempInputs, mTempOutputs);
            ::memcpy(outputPtr + index * outputStride, mMatrixC->host<float>(), outputStride * sizeof(float));
        }
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input0          = inputs[0];
        auto input1          = inputs[1];
        auto output          = outputs[0];
        mMatrixA.reset(Tensor::createDevice<float>({input0->length(input0->dimensions()-2), input0->length(input0->dimensions()-1)}));
        mMatrixB.reset(Tensor::createDevice<float>({input1->length(input1->dimensions()-2), input1->length(input1->dimensions()-1)}));
        mMatrixC.reset(Tensor::createDevice<float>({output->length(output->dimensions()-2), output->length(output->dimensions()-1)}));
        mTempInputs = {mMatrixA.get(), mMatrixB.get()};
        mTempOutputs = {mMatrixC.get()};
        auto res = backend()->onAcquireBuffer(mMatrixA.get(), Backend::DYNAMIC);
        res = res && backend()->onAcquireBuffer(mMatrixB.get(), Backend::DYNAMIC);
        res = res && backend()->onAcquireBuffer(mMatrixC.get(), Backend::DYNAMIC);

        if (!res) {
            return OUT_OF_MEMORY;
        }
        auto code = mMatMul->onResize(mTempInputs, mTempOutputs);
        backend()->onReleaseBuffer(mMatrixA.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mMatrixB.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mMatrixC.get(), Backend::DYNAMIC);
        return code;
    }
private:
    std::shared_ptr<Execution> mMatMul;
    std::vector<Tensor*> mTempInputs;
    std::vector<Tensor*> mTempOutputs;
    std::shared_ptr<Tensor> mMatrixA;
    std::shared_ptr<Tensor> mMatrixB;
    std::shared_ptr<Tensor> mMatrixC;
};

class CPUMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        if (outputs[0]->dimensions() > 2) {
            return new CPUMultiMatMul(backend, param->transposeA(), param->transposeB());
        }
        return new CPUMatMul(backend, param->transposeA(), param->transposeB(), true);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);

} // namespace MNN
