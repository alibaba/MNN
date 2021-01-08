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
#include "math/Vec.hpp"
#include <limits>

using Vec4 = MNN::Math::Vec<float, 4>;
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
            Vec4::save(srcX, Vec4::load(dstX));
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
            Vec4::save(dstX, Vec4::load(srcX));
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
    mComputer->onReset();
    mPreFunctions.clear();
    mPostFunctions.clear();
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
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
    mPreFunctions.emplace_back(std::make_pair([BTempPtr, l, h, this] (int tId, const float* APtr, const float* BPtr) {
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
        mPreFunctions.emplace_back(std::make_pair([ATPtr, e, l](int tId, const float* APtr, const float* BPtr) {
            MNNPackC4(ATPtr, APtr, e, l);
        }, 1));
    } else {
        // e, l -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair(
            [ATPtr, e, l, lC4, numberThread](int tId, const float* APtr, const float* BPtr) {
            _TransposePackC4MultiThread(APtr, ATPtr, tId, lC4, e, l, numberThread);
        }, numberThread));
    }
    std::shared_ptr<Tensor> biasWrap;
    std::vector<Tensor*> strassenInputs = {AT.get(), BT.get()};
    std::vector<float> postParameters;
    if (inputs.size() > 2) {
        auto bias = inputs[2];
        auto biasLength = bias->elementSize();
        if (biasLength % 4 != 0) {
            // Padding to align of 4
            biasWrap.reset(Tensor::createDevice<float>({UP_DIV(biasLength, 4) * 4}));
            res = backend()->onAcquireBuffer(biasWrap.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            auto borigin = bias->host<float>();
            auto bdest = biasWrap->host<float>();
            mPreFunctions.emplace_back(std::make_pair(
                [borigin, biasLength, bdest](int tId, const float* APtr, const float* BPtr) {
                ::memset(bdest, 0, UP_DIV(biasLength, 4) * 4 * sizeof(float));
                ::memcpy(bdest, borigin, biasLength * sizeof(float));
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
    auto CTPtr = CT->host<float>();

    // hC4, e, 4 -> e, h
    mPostFunctions.emplace_back(std::make_pair([CTPtr, e, h, hC4, numberThread](
            int tId, const float* APtr, const float* BPtr, float* CPtr) {
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

class CPUMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new CPUMatMul(backend, param->transposeA(), param->transposeB(), true);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);

} // namespace MNN
