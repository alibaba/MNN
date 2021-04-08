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
#include "core/AutoStorage.h"
#include "math/Vec.hpp"
#include <limits>

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB, bool multiThread)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB), mSupportMultiThread(multiThread) {
    mComputer.reset(new StrassenMatrixComputor(backend, mSupportMultiThread, 5));
}

void CPUMatMul::_scheduleForVecE(float* C, const float* biasPtr, int e, int l, int h) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    MNN_ASSERT(e == 1);
    MatMulParam param;
    param.e = 1;
    param.l = l;
    param.h = h;
    param.BTranspose = mTransposeB;
    param.numberThread = numberThread;
    mPostFunctions.emplace_back(std::make_pair([param, biasPtr](
                                                                             int tId, const float* A, const float* B, float* C) {
        MNNComputeMatMulForE_1(A, B, C, biasPtr, &param, tId);
    }, numberThread));
}

void CPUMatMul::_scheduleForVec(float* C, const float* biasPtr, int e, int l, int h) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    // TODD: Support e = 1
    MNN_ASSERT(h == 1);
    float biasValue = 0.0f;
    if (nullptr != biasPtr) {
        biasValue = *biasPtr;
    }
    if (mTransposeA) {
        mPostFunctions.emplace_back(std::make_pair([e, l, numberThread, biasValue](
            int tId, const float* A, const float* B, float* C) {
            auto eC4 = e / 4;
            auto eR = eC4 * 4;
            for (int y=tId; y<eC4; y+=numberThread) {
                Vec4 sumValue = Vec4(biasValue);
                auto srcY = A + y * 4;
                for (int x=0; x<l; ++x) {
                    sumValue = sumValue + Vec4::load(srcY + x * e) * Vec4(B[x]);
                }
                Vec4::save(C + 4 * y, sumValue);
            }
            if (0 == tId) {
                for (int y=eR; y<e; ++y) {
                    float sumValue = biasValue;
                    auto srcY = A + y;
                    for (int x=0; x<l; ++x) {
                        sumValue = sumValue + srcY[x * e] * B[x];
                    }
                    C[y] = sumValue;
                }
            }
        }, numberThread));
    } else {
        mPostFunctions.emplace_back(std::make_pair([e, l, numberThread, biasValue](
            int tId, const float* A, const float* B, float* C) {
            auto lC4 = l / 4;
            auto lR = lC4 * 4;
            for (int y=tId; y<e; y+=numberThread) {
                Vec4 sumValue = Vec4(biasValue);
                auto srcY = A + y * l;
                for (int x=0; x<lC4; ++x) {
                    sumValue = sumValue + Vec4::load(srcY + 4 * x) * Vec4::load(B + 4 * x);
                }
                float sumSingle = sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
                for (int x=lR; x<l; ++x) {
                    sumSingle += srcY[x] * B[x];
                }
                C[y] = sumSingle;
            }
        }, numberThread));
    }
}

ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* C       = outputs[0];

    // Fill output by zero if one of inputs is empty.
    if (A->elementSize() == 0 || B->elementSize() == 0) {
        return NO_ERROR;
    }
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mComputer->onReset();
    mPreFunctions.clear();
    mPostFunctions.clear();
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    if (core->bytes == 4) {
        if (h == 1) {
            const float* biasPtr = nullptr;
            if (inputs.size() > 2) {
                auto bias = inputs[2];
                biasPtr = bias->host<float>();
            }
            _scheduleForVec(C->host<float>(), biasPtr, e, l, h);
            return NO_ERROR;
        }
        if (e == 1) {
            const float* biasPtr = nullptr;
            if (inputs.size() > 2) {
                auto bias = inputs[2];
                biasPtr = bias->host<float>();
            }
            _scheduleForVecE(C->host<float>(), biasPtr, e, l, h);
            return NO_ERROR;
        }
    }
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    AutoRelease<Tensor> AT(Tensor::createDevice<float>({UP_DIV(l, core->pack), e, core->pack}));
    AutoRelease<Tensor> BT(Tensor::createDevice<float>({UP_DIV(h, hP), UP_DIV(l, lP) * lP, hP}));
    AutoRelease<Tensor> CT(Tensor::createDevice<float>({UP_DIV(h, core->pack), e, core->pack}));
    auto res = backend()->onAcquireBuffer(BT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto BTPtr = BT->host<float>();
    float* BTempPtr = BTPtr;
    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    mPreFunctions.emplace_back(std::make_pair([BTempPtr, l, h, this, core] (int tId, const float* APtr, const float* BPtr) {
        core->MNNPackForMatMul_B(BTempPtr, BPtr, h, l, mTransposeB);
    } , 1));
    res = backend()->onAcquireBuffer(AT.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(CT.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto ATPtr = AT->host<float>();
    if (mTransposeA) {
        // l, e -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair([ATPtr, e, l, core](int tId, const float* APtr, const float* BPtr) {
            core->MNNPackCUnit(ATPtr, APtr, e, l);
        }, 1));
    } else {
        // e, l -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair(
            [ATPtr, e, l, core](int tId, const float* APtr, const float* BPtr) {
            core->MNNPackCUnitTranspose(ATPtr, APtr, e, l);
        }, 1));
    }
    AutoRelease<Tensor> biasWrap;
    std::vector<Tensor*> strassenInputs = {AT.get(), BT.get()};
    std::vector<float> postParameters;
    if (inputs.size() > 2) {
        auto bias = inputs[2];
        auto biasLength = bias->elementSize();
        if (biasLength % core->pack != 0) {
            // Padding to align of 4
            biasWrap.reset(Tensor::createDevice<float>({UP_DIV(biasLength, core->pack) * core->pack}));
            res = backend()->onAcquireBuffer(biasWrap.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            auto borigin = bias->host<float>();
            auto bdest = biasWrap->host<float>();
            mPreFunctions.emplace_back(std::make_pair(
                [borigin, biasLength, bdest, core](int tId, const float* APtr, const float* BPtr) {
                ::memset(bdest, 0, UP_DIV(biasLength, core->pack) * core->bytes * core->pack);
                ::memcpy(bdest, borigin, biasLength * core->bytes);
            }, 1));
            strassenInputs.emplace_back(biasWrap.get());
        } else {
            strassenInputs.emplace_back(bias);
        }
        postParameters = {
            1.0f,
            1.0f,
            -std::numeric_limits<float>().max(),
            std::numeric_limits<float>().max(),
        };
    }
    auto code = mComputer->onEncode(strassenInputs, {CT.get()}, postParameters);
    if (NO_ERROR != code) {
        return code;
    }
    if (nullptr != biasWrap.get()) {
        backend()->onReleaseBuffer(biasWrap.get(), Backend::DYNAMIC);
    }

    auto CTPtr = CT->host<float>();
    // hC4, e, 4 -> e, h
    mPostFunctions.emplace_back(std::make_pair([CTPtr, e, h, core](
            int tId, const float* APtr, const float* BPtr, float* CPtr) {
        core->MNNUnpackCUnitTranspose(CPtr, CTPtr, e, h);
    }, 1));
    backend()->onReleaseBuffer(AT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(BT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(CT.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Fill output by zero if one of inputs is empty.
    if (inputs.size() == 2 && outputs.size() == 1 &&
        (inputs[0]->elementSize() == 0 || inputs[1]->elementSize() == 0)) {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        ::memset(outputs[0]->host<char>(), 0, outputs[0]->elementSize() * core->bytes);
        return NO_ERROR;
    }

    auto APtr = inputs[0]->host<float>();
    auto BPtr = inputs[1]->host<float>();
    auto CPtr = outputs[0]->host<float>();

    for (auto& f : mPreFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId, APtr, BPtr);
        }
        MNN_CONCURRENCY_END();
    }
    mComputer->onExecute();
    for (auto& f : mPostFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId, APtr, BPtr, CPtr);
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
        auto core = static_cast<CPUBackend*>(backend())->functions();
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
        auto input0Ptr   = input0->host<uint8_t>();
        auto input1Ptr   = input1->host<uint8_t>();
        auto outputPtr = output->host<uint8_t>();
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
            ::memcpy(mMatrixA->host<uint8_t>(), input0Ptr + i0Offset * input0Stride * core->bytes, input0Stride * core->bytes);
            ::memcpy(mMatrixB->host<uint8_t>(), input1Ptr + i1Offset * input1Stride * core->bytes, input1Stride * core->bytes);
            mMatMul->onExecute(mTempInputs, mTempOutputs);
            ::memcpy(outputPtr + index * outputStride * core->bytes, mMatrixC->host<uint8_t>(), outputStride * core->bytes);
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
