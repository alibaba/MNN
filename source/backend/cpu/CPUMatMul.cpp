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
    mComputer.reset(new StrassenMatrixComputor(backend, mSupportMultiThread, 5));
}
static void _TransposeUnpackC4MultiThread(float* BPtr, const float* BTempPtr, int tId, int hC4, int l, int h, int numberThread) {
    for (int y = tId; y < hC4 - 1; y+=numberThread) {
        auto src = y * 4 + BPtr;
        auto dst = y * 4 * l + BTempPtr;
        for (int x = 0; x< l ; ++x) {
            auto srcX = src + x * h;
            auto dstX = dst + 4 * x;
            for (int i=0; i<4; ++i) {
                srcX[i] = dstX[i];
            }
        }
    }
    if (tId != numberThread - 1) {
        return;
    }
    int lastY = 4 * (hC4 - 1);
    int remain = h - lastY;
    auto lastDst = BTempPtr + lastY * l;
    auto lastSrc = lastY + BPtr;
    for (int x=0; x<l; ++x) {
        auto srcX = lastSrc + x * h;
        auto dstX = lastDst + x * 4;
        for (int y = 0; y < remain; ++y) {
            srcX[y] = dstX[y];
        }
    }
}
static void _TransposePackC4MultiThread(const float* BPtr, float* BTempPtr, int tId, int hC4, int l, int h, int numberThread) {
    for (int y = tId; y < hC4 - 1; y+=numberThread) {
        auto src = y * 4 + BPtr;
        auto dst = y * 4 * l + BTempPtr;
        for (int x = 0; x< l ; ++x) {
            auto srcX = src + x * h;
            auto dstX = dst + 4 * x;
            for (int i=0; i<4; ++i) {
                dstX[i] = srcX[i];
            }
        }
    }
    if (tId != numberThread - 1) {
        return;
    }
    int lastY = 4 * (hC4 - 1);
    int remain = h - lastY;
    auto lastDst = BTempPtr + lastY * l;
    auto lastSrc = lastY + BPtr;
    for (int x=0; x<l; ++x) {
        auto srcX = lastSrc + x * h;
        auto dstX = lastDst + x * 4;
        ::memset(dstX, 0, 4 * sizeof(float));
        for (int y = 0; y < remain; ++y) {
            dstX[y] = srcX[y];
        }
    }
}
ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto APtr = A->host<float>();
    auto BPtr = B->host<float>();
    Tensor* C       = outputs[0];
    auto CPtr = C->host<float>();
    // Fill output by zero if one of inputs is empty.
    if (A->elementSize() == 0 || B->elementSize() == 0) {
        return NO_ERROR;
    }
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    mComputer->onReset();
    mPreFunctions.clear();
    mPostFunctions.clear();
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    std::shared_ptr<Tensor> AT(Tensor::createDevice<float>({UP_DIV(l, 4), e, 4}));
    std::shared_ptr<Tensor> BT(Tensor::createDevice<float>({UP_DIV(h, hP), l, hP}));
    std::shared_ptr<Tensor> CT(Tensor::createDevice<float>({UP_DIV(h, 4), e, 4}));
    auto res = backend()->onAcquireBuffer(BT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto BTPtr = BT->host<float>();
    float* BTempPtr = BTPtr;
    auto hC4 = UP_DIV(h, 4);
    auto lC4 = UP_DIV(l, 4);
    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    mPreFunctions.emplace_back(std::make_pair([BPtr, BTempPtr, l, h, this] (int tId) {
        MNNPackForMatMul_B(BTempPtr, BPtr, h, l, mTransposeB);
    } , 1));
    res = backend()->onAcquireBuffer(AT.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(CT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto ATPtr = AT->host<float>();
    if (mTransposeA) {
        // l, e -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair([ATPtr, APtr, e, l](int tId) {
            MNNPackC4(ATPtr, APtr, e, l);
        }, 1));
    } else {
        // e, l -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair([ATPtr, APtr, e, l, lC4, numberThread](int tId) {
            _TransposePackC4MultiThread(APtr, ATPtr, tId, lC4, e, l, numberThread);
        }, numberThread));
    }

    auto code = mComputer->onEncode({AT.get(), BT.get()}, {CT.get()});
    if (NO_ERROR != code) {
        return code;
    }
    auto CTPtr = CT->host<float>();

    // hC4, e, 4 -> e, h
    mPostFunctions.emplace_back(std::make_pair([CPtr, CTPtr, e, h, hC4, numberThread](int tId) {
        _TransposeUnpackC4MultiThread(CPtr, CTPtr, tId, hC4, e, h, numberThread);
    }, numberThread));
    backend()->onReleaseBuffer(AT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(BT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(CT.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Fill output by zero if one of inputs is empty.
    if (inputs.size() == 2 && outputs.size() == 1 &&
        (inputs[0]->elementSize() == 0 || inputs[1]->elementSize() == 0)) {
        ::memset(outputs[0]->host<char>(), 0, outputs[0]->size());
        return NO_ERROR;
    }
    for (auto& f : mPreFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId);
        }
        MNN_CONCURRENCY_END();
    }
    mComputer->onExecute();
    for (auto& f : mPostFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId);
        }
        MNN_CONCURRENCY_END();
    }
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
