//
//  StrassenMatmulComputor.cpp
//  MNN
//
//  Created by MNN on 2019/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StrassenMatmulComputor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include <string.h>
#include "ConvOpt.h"
#include <limits.h>
#include "CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "math/Vec.hpp"
#include "math/Matrix.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;
extern "C" {
void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t eSub, size_t hSub);
}

#ifndef MNN_USE_NEON
void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t eSub, size_t hSub) {
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * 4;
        for (int x=0; x<eSub; ++x) {
            auto xv = Vec4::load(xY + 4*x);
            auto c21v = Vec4::load(c21Y + 4*x);
            auto c11v = Vec4::load(c11Y + 4*x);
            auto c22v = Vec4::load(c22Y + 4*x);
            auto c12v = Vec4::load(c12Y + 4*x);
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            Vec4::save(c12Y + 4*x, c12v);
            Vec4::save(c22Y + 4*x, c22v);
            Vec4::save(c21Y + 4*x, c21v);
        }
    }
}
#endif

namespace MNN {
typedef std::shared_ptr<Tensor> PTensor;
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
    std::shared_ptr<Tensor> mTensor;
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
    auto aHost   = AT->host<float>();
    auto bHost   = BT->host<float>();
    auto cHost   = CT->host<float>();
    auto aStride = AT->stride(0);
    auto bStride = BT->stride(0);
    auto cStride = CT->stride(0);
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    auto CONVOLUTION_TILED_NUMBER = eP;
    auto bExtraStride = bStride - BT->length(1) * BT->length(2);
    AddTensor tileBuffer(Tensor::createDevice<float>(std::vector<int>{numberThread, BT->length(1), CONVOLUTION_TILED_NUMBER}), backend());
    std::vector<float*> cachePtr(numberThread, nullptr);
    if (hP % 4 != 0) {
        auto hDiv = MNNGetC4DivNumber(hP);
        AddTensor matmulTempBuffer(Tensor::createDevice<float>(std::vector<int>{numberThread, eP * hDiv * 4 + CT->length(0) * eP * 4}), backend());
        for (int i=0; i<numberThread; ++i) {
            cachePtr[i] = matmulTempBuffer->host<float>() + i * matmulTempBuffer->stride(0);
        }
    }
    auto tileHostOrigin  = tileBuffer->host<float>();
    int unitNumber = e / CONVOLUTION_TILED_NUMBER;
    int xCount     = e - unitNumber * CONVOLUTION_TILED_NUMBER;
    std::vector<size_t> parameters(6);
    auto hMin = std::min(CT->length(0) * 4, BT->length(0) * hP);
    parameters[0] = xCount * sizeof(float);
    parameters[1] = BT->length(1);
    parameters[2] = hMin;
    parameters[3] = cStride * sizeof(float);
    parameters[4] = 0;
    parameters[5] = bExtraStride * sizeof(float);
    auto eReal = aStride / AT->length(2);
    const float* biasPtr = nullptr;
    if (nullptr != COT) {
        if (COT != CT) {
            biasPtr = COT->host<float>();
        }
    }

    mFunctions.emplace_back(
        std::make_pair([xCount, aHost, bHost, cHost, tileHostOrigin, unitNumber, bExtraStride, numberThread, parameters, eReal, CONVOLUTION_TILED_NUMBER, cachePtr, biasPtr, active](int tId) {
            auto tileHost = tileHostOrigin + CONVOLUTION_TILED_NUMBER * parameters[1] * tId;
            const float* postParametersPtr = nullptr;
            if (!active.empty()) {
                postParametersPtr = active.data();
            }
            auto cache = cachePtr[tId];
            for (int i = tId; i < unitNumber; i+=numberThread) {
                int xStart    = i * CONVOLUTION_TILED_NUMBER;
                auto aStart   = aHost + xStart * 4;
                MNNPackC4ForMatMul_A(tileHost, aStart, CONVOLUTION_TILED_NUMBER, parameters[1], eReal);
                MNNPackedMatMul(cHost + 4 * xStart, tileHost, bHost, parameters.data(), cache, postParametersPtr, biasPtr);
            }
            if (tId != numberThread -1) {
                return;
            }
            if (xCount > 0) {
                int xStart    = unitNumber * CONVOLUTION_TILED_NUMBER;
                auto aStart   = aHost + xStart * 4;
                // Copy
                MNNPackC4ForMatMul_A(tileHost, aStart, xCount, parameters[1], eReal);
                MNNPackedMatMulRemain(cHost + 4 * xStart, tileHost, bHost, xCount, parameters.data(), cache, postParametersPtr, biasPtr);
            }
        }, numberThread));
    return NO_ERROR;
}

#define MNNMATRIX_SUB_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub) \
for (int y = tId; y < lSub; y+=numberThread) {\
MNNMatrixSub(c + y * cStride, a + y * aStride, b + y * bStride, widthC4, 0, 0, 0, 1);\
}\

#define MNNMATRIX_ADD_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub) \
for (int y = tId; y < lSub; y+=numberThread) {\
MNNMatrixAdd(c + y * cStride, a + y * aStride, b + y * bStride, widthC4, 0, 0, 0, 1);\
}\


ErrorCode StrassenMatrixComputor::_generateMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT, const Tensor* COT, int currentDepth, const std::vector<float>& postParameters) {
    auto l = AT->length(0);
    auto e = AT->length(1);
    auto h = CT->length(0);
    auto lReal = BT->length(1);
    static const int aUnit = 4;

    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto hDiv = MNNGetC4DivNumber(hP);
    auto eSub = (e / eP) / 2 * eP;
    auto lSub = l / 2;
    auto hSub = (h / hDiv) / 2 * hDiv;
    auto remainH = h - hSub * 2;
    auto remainE = e - eSub * 2;
    if (currentDepth >= mMaxDepth || eSub == 0 || hSub == 0 || lReal % 8 != 0) {
        return _generateTrivalMatMul(AT, BT, CT, COT, postParameters);
    }

    /*
     Compute the memory read / write cost for expand
     */
    auto bLSub = lSub * 4;
    auto bHSub = (hSub * 4) / hP;
    float AComputeCost = 4 * ((float)eSub * lSub) * aUnit;
    float BComputeCost = 4 * (float)bLSub * bHSub * hP;
    float CComputeCost = 7 * (float)eSub * hSub * aUnit;
    float saveMatMulCost = (e / eP) * (aUnit * eP * hSub + lSub * eP * aUnit + bLSub * bHSub * hP);
    const float pernaty = 1.5f;//FIXME: Find beter way to set it
    //MNN_PRINT("%f - %f, %f, %f\n", saveMatMulCost, AComputeCost, BComputeCost, CComputeCost);
    float saveCost = saveMatMulCost - (AComputeCost + BComputeCost + CComputeCost) * pernaty;
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

    auto xAddr = X->host<float>();
    auto yAddr = Y->host<float>();

    auto aStride = AT->stride(0);
    auto a11     = AT->host<float>() + 0 * aUnit * eSub + 0 * aStride * lSub;
    auto a12     = AT->host<float>() + 0 * aUnit * eSub + 1 * aStride * lSub;
    auto a21     = AT->host<float>() + 1 * aUnit * eSub + 0 * aStride * lSub;
    auto a22     = AT->host<float>() + 1 * aUnit * eSub + 1 * aStride * lSub;

    auto bStride = BT->stride(0);
    auto b11     = BT->host<float>() + 0 * bUnit * bLSub + 0 * bStride * bHSub;
    auto b12     = BT->host<float>() + 0 * bUnit * bLSub + 1 * bStride * bHSub;
    auto b21     = BT->host<float>() + 1 * bUnit * bLSub + 0 * bStride * bHSub;
    auto b22     = BT->host<float>() + 1 * bUnit * bLSub + 1 * bStride * bHSub;

    auto cStride = CT->stride(0);
    auto c11     = CT->host<float>() + 0 * aUnit * eSub + 0 * cStride * hSub;
    auto c12     = CT->host<float>() + 0 * aUnit * eSub + 1 * cStride * hSub;
    auto c21     = CT->host<float>() + 1 * aUnit * eSub + 0 * cStride * hSub;
    auto c22     = CT->host<float>() + 1 * aUnit * eSub + 1 * cStride * hSub;

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
        auto f = [a11, a21, b22, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11, a21, eSub * aUnit / 4, eSub * aUnit, aStride, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, b12, bLSub * bUnit / 4, bLSub * bUnit, bStride, bStride, bHSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C21.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [a22, a21, b11, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub](int tId) {
            MNNMATRIX_ADD_MULTITHREAD(xAddr, a21, a22, eSub * aUnit / 4, eSub * aUnit, aStride, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b12, b11, bLSub * bUnit / 4, bLSub * bUnit, bStride, bStride, bHSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C22.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [a11, b22, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, bUnit, bLSub, bHSub](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, a11, eSub * aUnit / 4, eSub * aUnit, eSub * aUnit, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, yAddr, bLSub * bUnit / 4, bLSub * bUnit, bStride, bLSub * bUnit, bHSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C12.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        auto f = [a12, xAddr, eSub, lSub, aStride, numberThread](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a12, xAddr, eSub * aUnit / 4, eSub * aUnit, aStride, eSub * aUnit, lSub);
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
        auto f = [c11, c12, c21, c22, b21, xAddr, yAddr, eSub, lSub, hSub, bStride, cStride, numberThread, bUnit, bHSub, bLSub](int tId) {
            for (int y = tId; y < hSub; y+=numberThread) {
                MNNStrassenMergeCFunction(c11 + y * cStride, c12 + y * cStride, c21 + y * cStride, c22 + y * cStride, xAddr + y * eSub * 4, 0, eSub, 1);
            }
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, b21, bLSub * bUnit / 4, bLSub * bUnit, bLSub * bUnit, bStride, bHSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(A22.get(), Y.get(), C11.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        auto f0 = [c11, c21, eSub, hSub, cStride, numberThread](int tId) {
            auto cw = eSub * aUnit / 4;
            MNNMATRIX_SUB_MULTITHREAD(c21, c21, c11, cw, cStride, cStride, cStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f0, numberThread));
        auto code = _generateMatMul(A12.get(), B21.get(), C11.get(), nullptr, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, xAddr, eSub, hSub, cStride, numberThread](int tId) {
            auto cw = eSub * aUnit / 4;
            MNNMATRIX_ADD_MULTITHREAD(c11, c11, xAddr, cw, cStride, cStride, eSub * aUnit, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
        if (!postParameters.empty() && nullptr != COT) {
            auto biasPtr = COT->host<float>();
            if (1 == numberThread) {
                auto postFunction = [c11, eSub, hSub, cStride, numberThread, biasPtr, postParameters](int tId) {
                    auto width = eSub * 2;
                    auto height = hSub * 2;
                    MNNAxByClampBroadcastC4(c11, c11, biasPtr, width, cStride, cStride, height, postParameters.data());
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            } else {
                auto postFunction = [c11, eSub, hSub, cStride, numberThread, biasPtr, postParameters](int tId) {
                    auto width = eSub * 2;
                    auto height = hSub * 2;
                    for (int y = tId; y < height; y+=numberThread) {
                        MNNAxByClampBroadcastC4(c11 + y * cStride, c11 + y * cStride, biasPtr + y * 4, width, 0, 0, 1, postParameters.data());
                    }
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            }
        }
    }
    if (remainH > 0) {
        auto lastH = hSub * 2;
        auto cLast = CT->host<float>() + cStride * lastH;
        auto lastHB = bHSub * 2;
        auto bLast = BT->host<float>() + bStride * lastHB;
        PTensor BLast(Tensor::create<float>(std::vector<int>{BT->length(0) - lastHB, BT->length(1), bUnit}, bLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{remainH, eSub * 2, aUnit}, cLast));
        PTensor ALast(Tensor::create<float>(std::vector<int>{l, eSub * 2, aUnit}, AT->host<float>()));
        PTensor biasWrap;
        const Tensor* bias = COT;
        if (nullptr != bias) {
            biasWrap.reset(Tensor::create<float>(std::vector<int>{remainH, 1, aUnit}, COT->host<float>() + 4 * lastH));
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
        auto aLast = AT->host<float>() + eSub * 2 * aUnit;
        auto cLast = CT->host<float>() + eSub * 2 * aUnit;
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
