//
//  CPUMatMul.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include "CPUMatMul.hpp"
#include "CPUBackend.hpp"
#include "math/Matrix.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "math/Vec.hpp"


using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

CPUMatMul::CPUMatMul(Backend* backend, bool transposeA, bool transposeB, bool transposeC, bool multiThread)
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB), mTransposeC(transposeC), mSupportMultiThread(multiThread) {
    mComputer.reset(new StrassenMatrixComputor(backend, mSupportMultiThread, 5));
}

void CPUMatMul::_scheduleForVecE(int e, int l, int h) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    MNN_ASSERT(e == 1);
    MatMulParam param;
    param.e = 1;
    param.l = l;
    param.h = h;
    param.BTranspose = mTransposeB;
    param.numberThread = numberThread;
    auto func = static_cast<CPUBackend*>(backend())->functions()->MNNComputeMatMulForE_1;
    mPostFunctions.emplace_back(std::make_pair([param, func](
                                                                             int tId, const float* A, const float* B, const float* biasPtr, float* C) {
        func(A, B, C, biasPtr, &param, tId);
    }, numberThread));
}

void CPUMatMul::_scheduleForVec(int e, int l, int h) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    MatMulParam param;
    param.e = e;
    param.l = l;
    param.h = 1;
    param.ATranspose = mTransposeA;
    param.numberThread = numberThread;
    auto func = static_cast<CPUBackend*>(backend())->functions()->MNNComputeMatMulForH_1;
    // TODD: Support e = 1
    MNN_ASSERT(h == 1);
    mPostFunctions.emplace_back(std::make_pair([param, func](
        int tId, const float* A, const float* B, const float* biasPtr, float* C) {
        func(A, B, C, biasPtr, &param, tId);
    }, numberThread));
}

ErrorCode CPUMatMul::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mPreFunctions.clear();
    mPostFunctions.clear();
    auto e = A->length(0);
    auto h = B->length(1);
    auto l = A->length(1);
    if (mTransposeA) {
        l = A->length(0);
        e = A->length(1);
    }
    if (mTransposeB) {
        h = B->length(0);
    }
    // If encoded but resized as h=1/e=1, the computer should clear firstly
    mComputer->onReset();
    if (h == 1) {
        _scheduleForVec(e, l, h);
        return NO_ERROR;
    }
    if (e == 1) {
        const float* biasPtr = nullptr;
        _scheduleForVecE(e, l, h);
        return NO_ERROR;
    }
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto ATPtrAlloc = bufferAlloc->alloc(UP_DIV(l, core->pack) * e * core->pack * core->bytes);
    auto BTPtrAlloc = bufferAlloc->alloc(UP_DIV(h, hP) * UP_DIV(l, lP) * lP * hP * core->bytes);
    auto CTPtrAlloc = bufferAlloc->alloc(UP_DIV(h, core->pack) * e * core->pack * core->bytes);
    if (nullptr == ATPtrAlloc.first || nullptr == BTPtrAlloc.first || nullptr == CTPtrAlloc.first) {
        return OUT_OF_MEMORY;
    }
    auto BTPtr = (uint8_t*)BTPtrAlloc.first + BTPtrAlloc.second;
    auto ATPtr = (uint8_t*)ATPtrAlloc.first + ATPtrAlloc.second;
    auto CTPtr = (uint8_t*)CTPtrAlloc.first + CTPtrAlloc.second;

    float* BTempPtr = (float*)BTPtr;
    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    mPreFunctions.emplace_back(std::make_pair([BTempPtr, l, h, this, core] (int tId, const float* APtr, const float* BPtr, const float* Bias) {
        core->MNNPackForMatMul_B(BTempPtr, BPtr, h, l, mTransposeB);
    } , 1));
    if (mTransposeA) {
        // l, e -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair([ATPtr, e, l, core](int tId, const float* APtr, const float* BPtr, const float* Bias) {
            int offset[] = {
                e, e
            };
            core->MNNPackCUnit((float*)ATPtr, APtr, e, l, offset);
        }, 1));
    } else {
        // e, l -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair(
            [ATPtr, e, l, core](int tId, const float* APtr, const float* BPtr, const float* Bias) {
            int offset[] = {
                e, e
            };
            core->MNNPackCUnitTranspose((float*)ATPtr, APtr, e, l, offset);
        }, 1));
    }
    bool useBias = false;
    uint8_t* biasPtr = nullptr;
    std::vector<float> postParameters;
    std::pair<void*, int> bdestAlloc = std::make_pair(nullptr, 0);
    if (inputs.size() > 2) {
        auto bias = inputs[2];
        useBias = true;
        auto biasLength = bias->elementSize();
        if (biasLength % core->pack != 0) {
            mStrassenUseBiasDirectly = false;
            // Padding to align of 4
            bdestAlloc = bufferAlloc->alloc(UP_DIV(biasLength, core->pack) * core->pack * core->bytes);
            if (bdestAlloc.first == nullptr) {
                return OUT_OF_MEMORY;
            }
            auto bdest = (float*)((uint8_t*)bdestAlloc.first + bdestAlloc.second);
            mPreFunctions.emplace_back(std::make_pair(
                [biasLength, bdest, core](int tId, const float* APtr, const float* BPtr, const float* borigin) {
                ::memset(bdest, 0, UP_DIV(biasLength, core->pack) * core->bytes * core->pack);
                ::memcpy(bdest, borigin, biasLength * core->bytes);
            }, 1));
            biasPtr = (uint8_t*)bdest;
        } else {
            mStrassenUseBiasDirectly = true;
            biasPtr = bias->host<uint8_t>();
        }
        postParameters = {
            1.0f,
            1.0f,
            -std::numeric_limits<float>().max(),
            std::numeric_limits<float>().max(),
        };
    }
    auto code = mComputer->onEncode(e, l, h, e * core->pack, UP_DIV(l, lP) * lP * hP, e * core->pack, ATPtr, BTPtr, CTPtr, useBias, biasPtr, postParameters);
    if (NO_ERROR != code) {
        return code;
    }
    if (bdestAlloc.first != nullptr) {
        bufferAlloc->free(bdestAlloc);
    }
    // hC4, e, 4 -> e, h
    if (mTransposeC) {
        mPostFunctions.emplace_back(std::make_pair([CTPtr, e, h, core](
                int tId, const float* APtr, const float* BPtr, const float* biasPtr, float* CPtr) {
            int offset[] = {
                e, e
            };
            core->MNNUnpackCUnitTranspose(CPtr, (float*)CTPtr, e, h, offset);
        }, 1));
    } else {
        mPostFunctions.emplace_back(std::make_pair([CTPtr, e, h, core](
                int tId, const float* APtr, const float* BPtr, const float* biasPtr, float* CPtr) {
            int offset[] = {
                e, e
            };
            core->MNNUnpackCUnit(CPtr, (float*)CTPtr, e, h, offset);
        }, 1));
    }
    bufferAlloc->free(ATPtrAlloc);
    bufferAlloc->free(BTPtrAlloc);
    bufferAlloc->free(CTPtrAlloc);
    return NO_ERROR;
}

ErrorCode CPUMatMul::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    auto APtr = inputs[0]->host<float>();
    auto BPtr = inputs[1]->host<float>();
    auto CPtr = outputs[0]->host<float>();

    const float* biasPtr = nullptr;
    if (inputs.size() > 2) {
        biasPtr = inputs[2]->host<float>();
    }
    execute(APtr, BPtr, CPtr, biasPtr);
    return NO_ERROR;
}

void CPUMatMul::execute(const float* APtr, const float* BPtr, float* CPtr, const float* biasPtr) {
    for (auto& f : mPreFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId, APtr, BPtr, biasPtr);
        }
        MNN_CONCURRENCY_END();
    }
    if (mStrassenUseBiasDirectly) {
        mComputer->onExecute(nullptr, nullptr, (uint8_t*)biasPtr, nullptr);
    } else {
        mComputer->onExecute();
    }
    for (auto& f : mPostFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId, APtr, BPtr, biasPtr, CPtr);
        }
        MNN_CONCURRENCY_END();
    }
}

class CPUMatMulCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new CPUMatMul(backend, param->transposeA(), param->transposeB(), true, true);
    }
};

REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);

} // namespace MNN
