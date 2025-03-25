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
        // Do nothing
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
    mPreFunctions.emplace_back(std::make_pair([param, func](
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
    mPreFunctions.emplace_back(std::make_pair([param, func](
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
    int e, l, h;
    bool valid = OpCommonUtils::computeMatMulSize(mTransposeA, mTransposeB, A, B, e, l, h);
    if (!valid) {
        return COMPUTE_SIZE_ERROR;
    }
    mE = 0;
    mL = 0;
    mH = 0;

    // If encoded but resized as h=1/e=1, the computer should clear firstly
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
    int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto ATPtrAlloc = bufferAlloc->alloc(eP * UP_DIV(l, lP) * lP * core->bytes * numberThread);
    int matmulBytes = core->bytes;
    if (core->matmulBytes != 0) {
        matmulBytes = core->matmulBytes;
    }
    auto BTPtrAlloc = bufferAlloc->alloc(UP_DIV(h, hP) * UP_DIV(l, lP) * lP * hP * matmulBytes);
    auto CTPtrAlloc = bufferAlloc->alloc(UP_DIV(h, core->pack) * eP * core->pack * core->bytes * numberThread);
    if (ATPtrAlloc.invalid() || BTPtrAlloc.invalid() || CTPtrAlloc.invalid()) {
        return OUT_OF_MEMORY;
    }

    mPreFunctions.emplace_back(std::make_pair([BTPtrAlloc, l, h, this, core] (int tId, const float* APtr, const float* BPtr, const float* Bias, float* C) {
        core->MNNPackForMatMul_B((float*)BTPtrAlloc.ptr(), BPtr, h, l, mTransposeB);
    } , 1));
    bool useBias = false;
    MemChunk bdestAlloc;
    bool bdestNeedFree = false;
    if (inputs.size() > 2) {
        auto bias = inputs[2];
        useBias = true;
        auto biasLength = bias->elementSize();
        if (biasLength % core->pack != 0) {
            mUseBiasDirectly = false;
            // Padding to align of 4
            bdestAlloc = bufferAlloc->alloc(UP_DIV(biasLength, core->pack) * core->pack * core->bytes);
            bdestNeedFree = true;
            if (bdestAlloc.invalid()) {
                return OUT_OF_MEMORY;
            }
            mTempBias = bdestAlloc;
            mPreFunctions.emplace_back(std::make_pair(
                [biasLength, bdestAlloc, core](int tId, const float* APtr, const float* BPtr, const float* borigin, float* C) {
                ::memset(bdestAlloc.ptr(), 0, UP_DIV(biasLength, core->pack) * core->bytes * core->pack);
                ::memcpy(bdestAlloc.ptr(), borigin, biasLength * core->bytes);
            }, 1));
        } else {
            mUseBiasDirectly = true;
            if (TensorUtils::getDescribeOrigin(bias)->mem.get()) {
                bdestAlloc = TensorUtils::getDescribeOrigin(bias)->mem->chunk();
            }
        }
        mPostParameters = {
            1.0f,
            1.0f,
            -std::numeric_limits<float>().max(),
            std::numeric_limits<float>().max(),
        };
    }
    if (bdestNeedFree) {
        bufferAlloc->free(bdestAlloc);
    }
    bufferAlloc->free(ATPtrAlloc);
    bufferAlloc->free(BTPtrAlloc);
    bufferAlloc->free(CTPtrAlloc);
    mTempA = ATPtrAlloc;
    mTempB = BTPtrAlloc;
    mTempC = CTPtrAlloc;
    mE = e;
    mL = l;
    mH = h;
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
            f.first(tId, APtr, BPtr, biasPtr, CPtr);
        }
        MNN_CONCURRENCY_END();
    }
    if (mE > 0) {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        int eP, lP, hP;
        core->MNNGetMatMulPackMode(&eP, &lP, &hP);
        const float* postPtr = mPostParameters.data();
        if (!mUseBiasDirectly) {
            biasPtr = (const float*)mTempBias.ptr();
        }
        if (nullptr == biasPtr) {
            postPtr = nullptr;
        }
        auto lAlign = UP_DIV(mL, lP) * lP;
        int tileCount = UP_DIV(mE, eP);
        int numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            auto TA = mTempA.ptr() + tId * eP * lAlign * core->bytes;
            auto TB = mTempB.ptr();
            auto hC4 = UP_DIV(mH, core->pack);
            auto TC = mTempC.ptr() + tId * eP * hC4 * core->pack * core->bytes;
            size_t parameters[6];
            parameters[0] = eP * core->bytes;
            parameters[1] = mL;
            parameters[2] = mH;
            parameters[3] = eP * core->pack * core->bytes;
            parameters[4] = 0;
            parameters[5] = 0;
            for (int tx=tId; tx<tileCount; tx+=numberThread) {
                int xStart = tx * eP;
                int xEnd = ALIMIN(xStart + eP, mE);
                int xC = xEnd - xStart;
                if (mTransposeA) {
                    // l, e -> l/lp, xC|eP, lp
                    if (lP > 1) {
                        // TODO: Speed up it
                        if (mL % lP != 0) {
                            ::memset(TA, 0, eP * lAlign * core->bytes);
                        }
                        if (core->bytes == 4) {
                            auto D = (int32_t*)TA;
                            auto S = (int32_t*)APtr;
                            for (int y=0; y<mL; ++y) {
                                int yc = y / lP;
                                int yr = y % lP;
                                for (int xx=0; xx<xC; ++xx) {
                                    D[yc * lP * eP + xx * lP + yr] = S[y * mE + xStart + xx];
                                }
                            }
                        } else {
                            MNN_ASSERT(core->bytes == 2);
                            auto D = (int16_t*)TA;
                            auto S = (int16_t*)APtr;
                            for (int y=0; y<mL; ++y) {
                                int yc = y / lP;
                                int yr = y % lP;
                                for (int xx=0; xx<xC; ++xx) {
                                    D[yc * lP * eP + xx * lP + yr] = S[y * mE + xStart + xx];
                                }
                            }
                        }
                    } else {
                        for (int y=0; y<mL; ++y) {
                            ::memcpy(TA + y*eP*core->bytes, (uint8_t*)APtr + (y * mE + xStart) * core->bytes, core->bytes * xC);
                        }
                    }
                } else {
                    if (lP > 1) {
                        // e, l -> l/lp, 1, xC|eP, lp
                        int lC = mL / lP;
                        int lR = mL % lP;
                        for (int yy=0; yy<lC; ++yy) {
                            for (int x=0; x<xC; ++x) {
                                ::memcpy(TA + (yy * eP * lP + x * lP) * core->bytes, (uint8_t*)APtr + ((x+xStart)*mL+yy*lP)*core->bytes, lP * core->bytes);
                            }
                        }
                        if (lR > 0) {
                            int yy = lC;
                            for (int x=0; x<xC; ++x) {
                                ::memset(TA + (yy * eP * lP + x * lP) * core->bytes, 0, lP * core->bytes);
                                ::memcpy(TA + (yy * eP * lP + x * lP) * core->bytes, (uint8_t*)APtr + ((x+xStart)*mL+yy*lP)*core->bytes, xC * core->bytes);
                            }
                        }
                    } else {
                        // e, l -> l, eP
                        int dims[] = {
                            xC,
                            mL,
                            mL,
                            eP
                        };
                        if (core->bytes == 2) {
                            auto S = (const int16_t*)APtr + xStart * mL;
                            auto D = (int16_t*)TA;
                            MNNTranspose16Bit(D, S, dims);
                        } else if (core->bytes == 4) {
                            auto S = (const int32_t*)APtr + xStart * mL;
                            auto D = (int32_t*)TA;
                            MNNTranspose32Bit(D, S, dims);
                        }
                    }
                }
                if (core->matmulBytes != 0) {
                    core->MNNFp32ToLowp((const float*)TA, (int16_t*)TA, eP * lAlign);
                }
                if (xC == eP) {
                    core->MNNPackedMatMul((float*)TC, (float*)TA, (float*)TB, parameters, postPtr, biasPtr, nullptr, nullptr);
                } else {
                    core->MNNPackedMatMulRemain((float*)TC, (float*)TA, (float*)TB, xC, parameters, postPtr, biasPtr, nullptr, nullptr);
                }
                int area[] = {
                    eP,
                    mE
                };
                if (mTransposeC) {
                    // hC4, e, 4 -> e, h
                    auto dst = (uint8_t*)CPtr + xStart * mH * core->bytes;
                    core->MNNUnpackCUnitTranspose((float*)dst, (const float*)TC, xC, mH, area);
                } else {
                    // hC4, e, 4 -> h, e
                    auto dst = (uint8_t*)CPtr + xStart * core->bytes;
                    core->MNNUnpackCUnit((float*)dst, (const float*)TC, xC, mH, area);
                }
            }
        };
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
