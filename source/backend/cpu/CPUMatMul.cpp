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
#include "backend/cpu/compute/CommonOptFunction.h"
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
    auto hC4 = UP_DIV(h, 4);
    auto lC4 = UP_DIV(l, 4);
    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    if (mTransposeB) {
        // h, l -> hC4, l, 4
        mPreFunctions.emplace_back(std::make_pair([BPtr, BTempPtr, l, h] (int tId) {
            MNNPackC4(BTempPtr, BPtr, l, h);
        }, 1));
    } else {
        // l, h -> hC4, l, 4
        mPreFunctions.emplace_back(std::make_pair([BPtr, BTempPtr, l, h, hC4, numberThread] (int tId) {
            _TransposePackC4MultiThread(BPtr, BTempPtr, tId, hC4, l, h, numberThread);
        }, numberThread));
    }
    if (l % 4 != 0) {
        // hC4, l, 4 -> hC4, lC4, 4, 4
        mPreFunctions.emplace_back(std::make_pair([BTPtr, BTempPtr, l, hC4, lC4, numberThread](int tId) {
            for (int y=tId; y<hC4; y+=numberThread) {
                auto dst = BTPtr + 16*lC4 * y;
                auto src = BTempPtr + 4 * l * y;
                ::memcpy(dst, src, 4*l*sizeof(float));
                ::memset(dst+4*l, 0, 4 * (lC4*4-l) * sizeof(float));
            }
        }, numberThread));
        backend()->onReleaseBuffer(BTemp.get(), Backend::DYNAMIC);
    }
    if (MNNReorder4x4ByPlatform(nullptr, 0)) {
        mPreFunctions.emplace_back(std::make_pair([BTPtr, hC4, lC4, numberThread](int tId) {
            for (int y=tId; y<hC4; y+=numberThread) {
                auto dst = BTPtr + 16*lC4 * y;
                MNNReorder4x4ByPlatform(dst, lC4);
            }
        }, numberThread));
    }
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
