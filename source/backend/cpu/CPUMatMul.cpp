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
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
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
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mPreFunctions.clear();
    mPostFunctions.clear();
    int e, l, h;
    OpCommonUtils::computeMatMulSize(mTransposeA, mTransposeB, A, B, e, l, h);

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
    if (ATPtrAlloc.invalid() || BTPtrAlloc.invalid() || CTPtrAlloc.invalid()) {
        return OUT_OF_MEMORY;
    }

    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    mPreFunctions.emplace_back(std::make_pair([BTPtrAlloc, l, h, this, core] (int tId, const float* APtr, const float* BPtr, const float* Bias) {
        core->MNNPackForMatMul_B((float*)BTPtrAlloc.ptr(), BPtr, h, l, mTransposeB);
    } , 1));
    if (mTransposeA) {
        // l, e -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair([ATPtrAlloc, e, l, core](int tId, const float* APtr, const float* BPtr, const float* Bias) {
            int offset[] = {
                e, e
            };
            core->MNNPackCUnit((float*)ATPtrAlloc.ptr(), APtr, e, l, offset);
        }, 1));
    } else {
        // e, l -> lC4, e, 4
        mPreFunctions.emplace_back(std::make_pair(
            [ATPtrAlloc, e, l, core](int tId, const float* APtr, const float* BPtr, const float* Bias) {
            int offset[] = {
                e, e
            };
            core->MNNPackCUnitTranspose((float*)ATPtrAlloc.ptr(), APtr, e, l, offset);
        }, 1));
    }
    bool useBias = false;
    std::vector<float> postParameters;
    MemChunk bdestAlloc;
    bool bdestNeedFree = false;
    if (inputs.size() > 2) {
        auto bias = inputs[2];
        useBias = true;
        auto biasLength = bias->elementSize();
        if (biasLength % core->pack != 0) {
            mStrassenUseBiasDirectly = false;
            // Padding to align of 4
            bdestAlloc = bufferAlloc->alloc(UP_DIV(biasLength, core->pack) * core->pack * core->bytes);
            bdestNeedFree = true;
            if (bdestAlloc.invalid()) {
                return OUT_OF_MEMORY;
            }
            mPreFunctions.emplace_back(std::make_pair(
                [biasLength, bdestAlloc, core](int tId, const float* APtr, const float* BPtr, const float* borigin) {
                ::memset(bdestAlloc.ptr(), 0, UP_DIV(biasLength, core->pack) * core->bytes * core->pack);
                ::memcpy(bdestAlloc.ptr(), borigin, biasLength * core->bytes);
            }, 1));
        } else {
            mStrassenUseBiasDirectly = true;
            if (TensorUtils::getDescribe(bias)->mem.get()) {
                bdestAlloc = TensorUtils::getDescribe(bias)->mem->chunk();
            }
        }
        postParameters = {
            1.0f,
            1.0f,
            -std::numeric_limits<float>().max(),
            std::numeric_limits<float>().max(),
        };
    }
    auto code = mComputer->onEncode(e, l, h, e * core->pack, UP_DIV(l, lP) * lP * hP, e * core->pack, ATPtrAlloc, BTPtrAlloc, CTPtrAlloc, useBias, bdestAlloc, postParameters);
    if (NO_ERROR != code) {
        return code;
    }
    if (bdestNeedFree) {
        bufferAlloc->free(bdestAlloc);
    }
    // hC4, e, 4 -> e, h
    if (mTransposeC) {
        mPostFunctions.emplace_back(std::make_pair([CTPtrAlloc, e, h, core](
                int tId, const float* APtr, const float* BPtr, const float* biasPtr, float* CPtr) {
            int offset[] = {
                e, e
            };
            core->MNNUnpackCUnitTranspose(CPtr, (float*)CTPtrAlloc.ptr(), e, h, offset);
        }, 1));
    } else {
        mPostFunctions.emplace_back(std::make_pair([CTPtrAlloc, e, h, core](
                int tId, const float* APtr, const float* BPtr, const float* biasPtr, float* CPtr) {
            int offset[] = {
                e, e
            };
            core->MNNUnpackCUnit(CPtr, (float*)CTPtrAlloc.ptr(), e, h, offset);
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
