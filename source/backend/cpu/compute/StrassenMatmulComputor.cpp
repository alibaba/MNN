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
#include "core/Macro.h"
#include "core/Concurrency.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "math/Vec4.hpp"
using namespace MNN::Math;
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
            Vec4::save(c11Y + 4*x, c11v);
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

ErrorCode StrassenMatrixComputor::_generateTrivalMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT) {
    // Generate Trival Matrix Multiply
    auto l = AT->length(0);
    auto e = AT->length(1);
    auto h = BT->length(0);
    MNN_ASSERT(l > 0 && e > 0 && h > 0);
    auto aHost   = AT->host<float>();
    auto bHost   = BT->host<float>();
    auto cHost   = CT->host<float>();
    auto aStride = AT->stride(0);
    auto bStride = BT->stride(0);
    auto cStride = CT->stride(0);
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;

    auto bExtraStride = bStride - BT->length(1) * BT->length(2);
    std::shared_ptr<AddTensor> bCopy;
    if (e > CONVOLUTION_TILED_NUMBER && h >= 4 && l >= 4) {
        AddTensor tileBuffer(Tensor::createDevice<float>(std::vector<int>{numberThread, l, CONVOLUTION_TILED_NUMBER, 4}), backend());
        auto tileHostOrigin  = tileBuffer->host<float>();
        int unitNumber = e / CONVOLUTION_TILED_NUMBER;
        int xCount     = e - unitNumber * CONVOLUTION_TILED_NUMBER;
        mFunctions.emplace_back(
            std::make_pair([xCount, aHost, bHost, cHost, l, h, cStride, aStride, tileHostOrigin, unitNumber, bExtraStride, numberThread](int tId) {
                auto tileHost = tileHostOrigin + CONVOLUTION_TILED_NUMBER * 4 * l * tId;
                for (int i = tId; i < unitNumber; i+=numberThread) {
                    int xStart    = i * CONVOLUTION_TILED_NUMBER;
                    int lineCount = CONVOLUTION_TILED_NUMBER * 4;
                    auto aStart   = aHost + xStart * 4;
                    MNNMatrixCopyUnit(tileHost, aStart, lineCount, aStride, l);
                    MNNGemmFloatUnit_4(cHost + 4 * xStart, tileHost, bHost, l, cStride, h, bExtraStride);
                }
                if (tId != numberThread -1) {
                    return;
                }
                if (xCount > 0) {
                    int xStart    = unitNumber * CONVOLUTION_TILED_NUMBER;
                    int lineCount = xCount * 4;
                    auto aStart   = aHost + xStart * 4;
                    // Copy
                    MNNMatrixCopy(tileHost, aStart, xCount, lineCount, aStride, l);
                    if (1 == xCount) {
                        MNNGemmFloatOne_4(cHost + 4 * xStart, tileHost, bHost, l, cStride, h, bExtraStride);
                    } else {
                        MNNGemmFloatCommon_4(cHost + 4 * xStart, tileHost, bHost, l, cStride, h, xCount, bExtraStride);
                    }
                }
            }, numberThread));
        return NO_ERROR;
    }

    std::shared_ptr<AddTensor> aCopy;
    if (AT->length(1) * AT->length(2) != aStride) {
        aCopy.reset(new AddTensor(Tensor::createDevice<float>(AT->shape()), backend()));
        auto tempHost = aCopy->get()->host<float>();
        mFunctions.emplace_back(
            std::make_pair([e, l, aStride, aHost, tempHost](int tId) { MNNMatrixCopy(tempHost, aHost, e * 4 / 4, e * 4, aStride, l); }, 1));
        aHost = tempHost;
    }
    if (e == CONVOLUTION_TILED_NUMBER) {
        mFunctions.emplace_back(std::make_pair([aHost, bHost, cHost, l, h, cStride, bStride, numberThread, bExtraStride](int tId) {
            int yStep = UP_DIV(h, numberThread), yStart = tId * yStep, yNum = ALIMIN(yStart + yStep, h) - yStart;
            if (yNum <= 0) return;
            MNNGemmFloatUnit_4(cHost + cStride * yStart, aHost, bHost + bStride * yStart, l, cStride, yNum, bExtraStride);
        }, numberThread));
    } else if (e == 1) {
        mFunctions.emplace_back(std::make_pair([aHost, bHost, cHost, l, h, cStride, bStride, numberThread, bExtraStride](int tId) {
            int yStep = UP_DIV(h, numberThread), yStart = tId * yStep, yNum = ALIMIN(yStart + yStep, h) - yStart;
            if (yNum <= 0) return;
            MNNGemmFloatOne_4(cHost + yStart * cStride, aHost, bHost + yStart * bStride, l, cStride, yNum, bExtraStride);
        }, numberThread));
    } else {
        mFunctions.emplace_back(std::make_pair([aHost, bHost, cHost, l, e, h, cStride, bStride, numberThread, bExtraStride](int tId) {
            int yStep = UP_DIV(h, numberThread), yStart = tId * yStep, yNum = ALIMIN(yStart + yStep, h) - yStart;
            if (yNum <= 0) return;
            MNNGemmFloatCommon_4(cHost + yStart * cStride, aHost, bHost + bStride * yStart, l, cStride, yNum, e, bExtraStride);
        }, numberThread));
    }
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


ErrorCode StrassenMatrixComputor::_generateMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT,
                                                  int currentDepth) {
    auto l = AT->length(0);
    auto e = AT->length(1);
    auto h = BT->length(0);

    auto eSub = e / 2;
    auto lSub = l / 2;
    auto hSub = h / 2;

    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;

    /*
     Compute the memory read / write cost for expand
     Matrix Mul need eSub*lSub*hSub*(1+1.0/CONVOLUTION_TILED_NUMBER), Matrix Add/Sub need x*y*UNIT*3 (2 read 1 write)
     */
    float saveCost = (eSub * lSub * hSub) * (1.0f + 1.0f / CONVOLUTION_TILED_NUMBER) - 4 * (eSub * lSub) * 3 -
                     4 * (4 * lSub * hSub * 3) - 7 * (eSub * hSub * 3);
    if (currentDepth >= mMaxDepth || e <= CONVOLUTION_TILED_NUMBER || l % 2 != 0 || h % 2 != 0 || saveCost < 0.0f) {
        return _generateTrivalMatMul(AT, BT, CT);
    }

    // Strassen Construct
    auto bn = backend();
    currentDepth += 1;
    static const int aUnit = 4;
    static const int bUnit = 16;
    auto AS                = std::vector<int>{lSub, eSub, aUnit};
    auto BS                = std::vector<int>{hSub, lSub, bUnit};
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
    auto b11     = BT->host<float>() + 0 * bUnit * lSub + 0 * bStride * hSub;
    auto b12     = BT->host<float>() + 0 * bUnit * lSub + 1 * bStride * hSub;
    auto b21     = BT->host<float>() + 1 * bUnit * lSub + 0 * bStride * hSub;
    auto b22     = BT->host<float>() + 1 * bUnit * lSub + 1 * bStride * hSub;

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
        auto f = [a11, a21, b22, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11, a21, eSub * aUnit / 4, eSub * aUnit, aStride, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, b12, lSub * bUnit / 4, lSub * bUnit, bStride, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C21.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [a22, a21, b11, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread](int tId) {
            MNNMATRIX_ADD_MULTITHREAD(xAddr, a21, a22, eSub * aUnit / 4, eSub * aUnit, aStride, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b12, b11, lSub * bUnit / 4, lSub * bUnit, bStride, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C22.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [a11, b22, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, a11, eSub * aUnit / 4, eSub * aUnit, eSub * aUnit, aStride, lSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, yAddr, lSub * bUnit / 4, lSub * bUnit, bStride, lSub * bUnit, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C12.get(), currentDepth);
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
        auto code = _generateMatMul(X.get(), B22.get(), C11.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
        code = _generateMatMul(A11.get(), B11.get(), CX.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
        auto f = [c11, c12, c21, c22, b21, xAddr, yAddr, eSub, lSub, hSub, bStride, cStride, numberThread](int tId) {
            for (int y = tId; y < hSub; y+=numberThread) {
                MNNStrassenMergeCFunction(c11 + y * cStride, c12 + y * cStride, c21 + y * cStride, c22 + y * cStride, xAddr + y * eSub * 4, 0, eSub, 1);
            }
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, b21, lSub * bUnit / 4, lSub * bUnit, lSub * bUnit, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(A22.get(), Y.get(), C11.get(), currentDepth);
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
        auto code = _generateMatMul(A12.get(), B21.get(), C11.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, xAddr, eSub, hSub, cStride, numberThread](int tId) {
            auto cw = eSub * aUnit / 4;
            MNNMATRIX_ADD_MULTITHREAD(c11, c11, xAddr, cw, cStride, cStride, eSub * aUnit, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
    }
    if (e % 2 != 0) {
        auto aLast = AT->host<float>() + eSub * 2 * aUnit;
        auto cLast = CT->host<float>() + eSub * 2 * aUnit;
        PTensor ALast(Tensor::create<float>(std::vector<int>{l, 1, aUnit}, aLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{h, 1, aUnit}, cLast));
        ALast->setStride(0, aStride);
        CLast->setStride(0, cStride);
        _generateMatMul(ALast.get(), BT, CLast.get(), currentDepth);
    }
    return NO_ERROR;
}

void StrassenMatrixComputor::onReset() {
    mFunctions.clear();
    mConstTensor.clear();
}

ErrorCode StrassenMatrixComputor::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 2);
    MNN_ASSERT(outputs.size() == 1);
    auto A  = inputs[0];
    auto BT = inputs[1];
    auto C  = outputs[0];
    return _generateMatMul(A, BT, C, 0);
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
