//
//  StrassenMatmulComputor.cpp
//  MNN
//
//  Created by MNN on 2019/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StrassenMatmulComputor.hpp"
#include "CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"
#include <string.h>
#include <limits.h>
#include "core/AutoStorage.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "math/Vec.hpp"
#include "math/Matrix.hpp"

namespace MNN {
typedef AutoRelease<Tensor> PTensor;
class StrassenMatrixComputor::AddTensor {
public:
    AddTensor(Tensor* t, Backend* bn, Backend::StorageType storageType = Backend::DYNAMIC) {
        mTensor.reset(t);
        mValid       = bn->onAcquireBuffer(t, storageType);
        mBackend     = bn;
        mStorageType = storageType;
    }
    inline bool valid() const {
        return mValid;
    }
    ~AddTensor() {
        mBackend->onReleaseBuffer(mTensor.get(), mStorageType);
    }

    const Tensor* operator->() const {
        return mTensor.get();
    }
    const Tensor* get() const {
        return mTensor.get();
    }

private:
    AutoRelease<Tensor> mTensor;
    Backend* mBackend;
    bool mValid = false;
    Backend::StorageType mStorageType;
};
StrassenMatrixComputor::StrassenMatrixComputor(Backend* bn, bool multithread, int maxDepth) : mBackend(bn) {
    mMaxDepth = maxDepth;
    mSupportMultiThread = multithread;
};
StrassenMatrixComputor::~StrassenMatrixComputor() {
    // Do nothing
}

ErrorCode StrassenMatrixComputor::_generateTrivalMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT, const Tensor* COT, const std::vector<float>& active) {
    // Generate Trival Matrix Multiply
    auto e = AT->length(1);
    MNN_ASSERT(e > 0);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes    = core->bytes;
    auto packedA = core->MNNPackC4ForMatMul_A;
    auto matmul  = core->MNNPackedMatMul;
    auto matmulr = core->MNNPackedMatMulRemain;
    auto aHost   = AT->host<uint8_t>();
    auto bHost   = BT->host<uint8_t>();
    auto cHost   = CT->host<uint8_t>();
    auto aStride = AT->stride(0);
    auto bStride = BT->stride(0);
    auto cStride = CT->stride(0);
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto l = BT->length(1);
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    auto bExtraStride = bStride - BT->length(1) * BT->length(2);
    AddTensor tileBuffer(Tensor::createDevice<uint8_t>(std::vector<int>{numberThread, UP_DIV(l, lP) * eP * lP * bytes}), backend());
    auto tileHostOrigin  = tileBuffer->host<uint8_t>();
    int unitNumber = e / eP;
    int xCount     = e - unitNumber * eP;
    std::vector<size_t> parameters(6);
    auto hMin = std::min(CT->length(0) * core->pack, BT->length(0) * hP);
    parameters[0] = xCount * bytes;
    parameters[1] = l;
    parameters[2] = hMin;
    parameters[3] = cStride * bytes;
    parameters[4] = 0;
    parameters[5] = bExtraStride * bytes;
    auto eReal = aStride / AT->length(2);
    const float* biasPtr = nullptr;
    if (nullptr != COT) {
        if (COT != CT) {
            biasPtr = COT->host<float>();
        }
    }

    mFunctions.emplace_back(
        std::make_pair([xCount, aHost, bHost, cHost, tileHostOrigin, unitNumber, bExtraStride, numberThread, parameters, eReal, eP, biasPtr, active, packedA, matmul, matmulr, core](int tId) {
            auto tileHost = tileHostOrigin + eP * parameters[1] * tId * core->bytes;
            const float* postParametersPtr = nullptr;
            if (!active.empty()) {
                postParametersPtr = active.data();
            }
            auto packUnit = core->bytes * core->pack;
            int32_t info[4];
            int32_t stride[4];
            stride[0] = eP;
            stride[1] = parameters[1];
            stride[2] = 0;
            stride[3] = 0;
            info[0] = 1;
            info[1] = eReal;
            info[2] = eP;
            info[3] = 1;
            for (int i = tId; i < unitNumber; i+=numberThread) {
                int xStart    = i * eP;
                auto aStart   = aHost + xStart * packUnit;
                packedA((float*)(tileHost), (const float**)(&aStart), info, stride);
                matmul((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, parameters.data(), postParametersPtr, biasPtr);
            }
            if (tId != numberThread -1) {
                return;
            }
            if (xCount > 0) {
                stride[0] = xCount;
                stride[1] = parameters[1];
                info[2] = xCount;

                int xStart    = unitNumber * eP;
                auto aStart   = aHost + xStart * packUnit;
                // Copy
                packedA((float*)(tileHost), (const float**)(&aStart), info, stride);
                matmulr((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, xCount, parameters.data(), postParametersPtr, biasPtr);
            }
        }, numberThread));
    return NO_ERROR;
}

#define MNNMATRIX_SUB_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub, core) \
for (int y = tId; y < lSub; y+=numberThread) {\
core->MNNMatrixSub((float*)(c + y * cStride * core->bytes), (float*)(a + y * aStride * core->bytes), (float*)(b + y * bStride * core->bytes), widthC4, 0, 0, 0, 1);\
}\

#define MNNMATRIX_ADD_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub, core) \
for (int y = tId; y < lSub; y+=numberThread) {\
core->MNNMatrixAdd((float*)(c + y * cStride * core->bytes), (float*)(a + y * aStride * core->bytes), (float*)(b + y * bStride * core->bytes), widthC4, 0, 0, 0, 1);\
}\


ErrorCode StrassenMatrixComputor::_generateMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT, const Tensor* COT, int currentDepth, const std::vector<float>& postParameters) {
    auto l = AT->length(0);
    auto e = AT->length(1);
    auto h = CT->length(0);
    auto lReal = BT->length(1);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto aUnit = core->pack;

    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    MNN_ASSERT(hP % core->pack == 0);
    auto hDiv = hP / core->pack;
    auto eSub = (e / eP) / 2 * eP;
    auto lSub = l / 2;
    auto hSub = (h / hDiv) / 2 * hDiv;
    auto remainH = h - hSub * 2;
    auto remainE = e - eSub * 2;
    auto lMinDiv = std::max(core->pack * 2, 2 * lP);
    if (currentDepth >= mMaxDepth || eSub == 0 || hSub == 0 || lReal % lMinDiv != 0) {
        return _generateTrivalMatMul(AT, BT, CT, COT, postParameters);
    }

    /*
     Compute the memory read / write cost for expand
     */
    auto bLSub = lSub * core->pack;
    auto bHSub = (hSub * core->pack) / hP;
    float AComputeCost = 4 * ((float)eSub * lSub) * aUnit;
    float BComputeCost = 4 * (float)bLSub * bHSub * hP;
    float CComputeCost = 7 * (float)eSub * hSub * aUnit;
    float saveMatMulCost = (e / eP) * (aUnit * eP * hSub + lSub * eP * aUnit + bLSub * bHSub * hP);
    const float penalty = core->penalty;//FIXME: Find beter way to set it
    //MNN_PRINT("%f - %f, %f, %f\n", saveMatMulCost, AComputeCost, BComputeCost, CComputeCost);
    float saveCost = saveMatMulCost - (AComputeCost + BComputeCost + CComputeCost) * penalty;
    if (saveCost <= 0.0f) {
        return _generateTrivalMatMul(AT, BT, CT, COT, postParameters);
    }

    // Strassen Construct
    auto bn = backend();
    currentDepth += 1;
    auto bUnit = hP;
    auto AS                = std::vector<int>{lSub, eSub, aUnit};
    auto BS                = std::vector<int>{bHSub, bLSub, bUnit};
    auto CS                = std::vector<int>{hSub, eSub, aUnit};

    auto ACS = AS;
    if (CS[0] > ACS[0]) {
        ACS[0] = CS[0];
    }

    // Use XReal to contain both AX and CX, that's two cache
    AddTensor XReal(Tensor::createDevice<float>(ACS), bn);
    AddTensor Y(Tensor::createDevice<float>(BS), bn);
    if (!XReal.valid() || !Y.valid()) {
        return OUT_OF_MEMORY;
    }

    PTensor X(Tensor::create<float>(AS, XReal->host<float>()));
    PTensor CX(Tensor::create<float>(CS, XReal->host<float>()));

    auto xAddr = X->host<uint8_t>();
    auto yAddr = Y->host<uint8_t>();

    auto aStride = AT->stride(0);
    auto a11     = AT->host<uint8_t>() + (0 * aUnit * eSub + 0 * aStride * lSub) * core->bytes;
    auto a12     = AT->host<uint8_t>() + (0 * aUnit * eSub + 1 * aStride * lSub) * core->bytes;
    auto a21     = AT->host<uint8_t>() + (1 * aUnit * eSub + 0 * aStride * lSub) * core->bytes;
    auto a22     = AT->host<uint8_t>() + (1 * aUnit * eSub + 1 * aStride * lSub) * core->bytes;

    auto bStride = BT->stride(0);
    auto b11     = BT->host<uint8_t>() + (0 * bUnit * bLSub + 0 * bStride * bHSub) * core->bytes;
    auto b12     = BT->host<uint8_t>() + (0 * bUnit * bLSub + 1 * bStride * bHSub) * core->bytes;
    auto b21     = BT->host<uint8_t>() + (1 * bUnit * bLSub + 0 * bStride * bHSub) * core->bytes;
    auto b22     = BT->host<uint8_t>() + (1 * bUnit * bLSub + 1 * bStride * bHSub) * core->bytes;

    auto cStride = CT->stride(0);
    auto c11     = CT->host<uint8_t>() + (0 * aUnit * eSub + 0 * cStride * hSub) * core->bytes;
    auto c12     = CT->host<uint8_t>() + (0 * aUnit * eSub + 1 * cStride * hSub) * core->bytes;
    auto c21     = CT->host<uint8_t>() + (1 * aUnit * eSub + 0 * cStride * hSub) * core->bytes;
    auto c22     = CT->host<uint8_t>() + (1 * aUnit * eSub + 1 * cStride * hSub) * core->bytes;

    PTensor A11(Tensor::create<float>(AS, a11));
    A11->setStride(0, aStride);
    PTensor A12(Tensor::create<float>(AS, a12));
    A12->setStride(0, aStride);
    PTensor A21(Tensor::create<float>(AS, a21));
    A21->setStride(0, aStride);
    PTensor A22(Tensor::create<float>(AS, a22));
    A22->setStride(0, aStride);

    PTensor B11(Tensor::create<float>(BS, b11));
    B11->setStride(0, bStride);
    PTensor B12(Tensor::create<float>(BS, b12));
    B12->setStride(0, bStride);
    PTensor B21(Tensor::create<float>(BS, b21));
    B21->setStride(0, bStride);
    PTensor B22(Tensor::create<float>(BS, b22));
    B22->setStride(0, bStride);

    PTensor C11(Tensor::create<float>(CS, c11));
    C11->setStride(0, cStride);
    PTensor C12(Tensor::create<float>(CS, c12));
    C12->setStride(0, cStride);
    PTensor C21(Tensor::create<float>(CS, c21));
    C21->setStride(0, cStride);
    PTensor C22(Tensor::create<float>(CS, c22));
    C22->setStride(0, cStride);

    {
        // S3=A11-A21, T3=B22-B12, P7=S3*T3
        auto f = [a11, a21, b22, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub, core](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11, a21, eSub, eSub * core->pack, aStride, aStride, lSub, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, b12, bLSub * bUnit / core->pack, bLSub * bUnit, bStride, bStride, bHSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C21.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [a22, a21, b11, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub, core](int tId) {
            MNNMATRIX_ADD_MULTITHREAD(xAddr, a21, a22, eSub, eSub * core->pack, aStride, aStride, lSub, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b12, b11, bLSub * bUnit / core->pack, bLSub * bUnit, bStride, bStride, bHSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C22.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [a11, b22, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub, core](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, a11, eSub, eSub * core->pack, eSub * core->pack, aStride, lSub, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, yAddr, bLSub * bUnit / core->pack, bLSub * bUnit, bStride, bLSub * bUnit, bHSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C12.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        auto f = [a12, xAddr, eSub, lSub, aStride, numberThread, core](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a12, xAddr, eSub, eSub * core->pack, aStride, eSub * core->pack, lSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), B22.get(), C11.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        code = _generateMatMul(A11.get(), B11.get(), CX.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
        auto f = [c11, c12, c21, c22, b21, xAddr, yAddr, eSub, lSub, hSub, bStride, cStride, numberThread, bUnit, bHSub, bLSub, core](int tId) {
            for (int y = tId; y < hSub; y+=numberThread) {
                core->MNNStrassenMergeCFunction((float*)(c11 + y * cStride * core->bytes), (float*)(c12 + y * cStride * core->bytes), (float*)(c21 + y * cStride * core->bytes), (float*)(c22 + y * cStride * core->bytes), (float*)(xAddr + y * eSub * core->pack * core->bytes), 0, eSub, 1);
            }
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, b21, bLSub * bUnit / core->pack, bLSub * bUnit, bLSub * bUnit, bStride, bHSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(A22.get(), Y.get(), C11.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        auto f0 = [c11, c21, eSub, hSub, cStride, numberThread, core](int tId) {
            auto cw = eSub;
            MNNMATRIX_SUB_MULTITHREAD(c21, c21, c11, cw, cStride, cStride, cStride, hSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f0, numberThread));
        auto code = _generateMatMul(A12.get(), B21.get(), C11.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, xAddr, eSub, hSub, cStride, numberThread, core](int tId) {
            auto cw = eSub;
            MNNMATRIX_ADD_MULTITHREAD(c11, c11, xAddr, cw, cStride, cStride, eSub * core->pack, hSub, core);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
        if (!postParameters.empty() && nullptr != COT) {
            auto biasPtr = COT->host<float>();
            if (1 == numberThread) {
                auto postFunction = [c11, eSub, hSub, cStride, numberThread, biasPtr, postParameters, core](int tId) {
                    auto width = eSub * 2;
                    auto height = hSub * 2;
                    core->MNNAxByClampBroadcastUnit((float*)c11, (float*)c11, biasPtr, width, cStride, cStride, height, postParameters.data());
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            } else {
                auto postFunction = [c11, eSub, hSub, cStride, numberThread, biasPtr, postParameters, core](int tId) {
                    auto width = eSub * 2;
                    auto height = hSub * 2;
                    for (int y = tId; y < height; y+=numberThread) {
                        core->MNNAxByClampBroadcastUnit((float*)(c11 + y * cStride * core->bytes), (float*)(c11 + y * cStride * core->bytes), (const float*)((uint8_t*)biasPtr + y * core->bytes * core->pack), width, 0, 0, 1, postParameters.data());
                    }
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            }
        }
    }
    if (remainH > 0) {
        auto lastH = hSub * 2;
        auto cLast = CT->host<uint8_t>() + cStride * lastH * core->bytes;
        auto lastHB = bHSub * 2;
        auto bLast = BT->host<uint8_t>() + bStride * lastHB * core->bytes;
        PTensor BLast(Tensor::create<float>(std::vector<int>{BT->length(0) - lastHB, BT->length(1), bUnit}, bLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{remainH, eSub * 2, aUnit}, cLast));
        PTensor ALast(Tensor::create<float>(std::vector<int>{l, eSub * 2, aUnit}, AT->host<float>()));
        PTensor biasWrap;
        const Tensor* bias = COT;
        if (nullptr != bias) {
            biasWrap.reset(Tensor::create<float>(std::vector<int>{remainH, 1, aUnit}, COT->host<uint8_t>() + core->bytes * core->pack * lastH));
            bias = biasWrap.get();
        }
        BLast->setStride(0, bStride);
        CLast->setStride(0, cStride);
        ALast->setStride(0, aStride);
        auto code = _generateTrivalMatMul(AT, BLast.get(), CLast.get(), bias, postParameters);
        if (NO_ERROR != code) {
            return code;
        }
    }
    if (remainE > 0) {
        auto aLast = AT->host<uint8_t>() + eSub * 2 * aUnit * core->bytes;
        auto cLast = CT->host<uint8_t>() + eSub * 2 * aUnit * core->bytes;
        PTensor ALast(Tensor::create<float>(std::vector<int>{l, remainE, aUnit}, aLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{h, remainE, aUnit}, cLast));
        ALast->setStride(0, aStride);
        CLast->setStride(0, cStride);
        auto code = _generateTrivalMatMul(ALast.get(), BT, CLast.get(), COT, postParameters);
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}

void StrassenMatrixComputor::onReset() {
    mFunctions.clear();
}

ErrorCode StrassenMatrixComputor::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const std::vector<float>& postParameters) {
    MNN_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    MNN_ASSERT(outputs.size() == 1);
    auto A  = inputs[0];
    auto BT = inputs[1];
    auto C  = outputs[0];
    Tensor* CO = nullptr;
    if (inputs.size() > 2) {
        CO = inputs[2];
    }
    return _generateMatMul(A, BT, C, CO, 0, postParameters);
}
void StrassenMatrixComputor::onExecute() {
    // All is done in onResize, just execute it
    for (auto& f : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId);
        }
        MNN_CONCURRENCY_END();
    }
}
} // namespace MNN
