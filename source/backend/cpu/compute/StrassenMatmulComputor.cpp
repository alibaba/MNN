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
#ifndef MNN_USE_SSE
void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t length, size_t hSub) {
    auto lengthC4 = length / 4;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * length;
        for (int x=0; x<lengthC4; ++x) {
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
        for (int x=lengthC4*4; x<length; ++x) {
            auto xv = xY[x];
            auto c21v = c21Y[x];
            auto c11v = c11Y[x];
            auto c22v = c22Y[x];
            auto c12v = c12Y[x];
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            c12Y[x] = c12v;
            c22Y[x] = c22v;
            c21Y[x] = c21v;
            c11Y[x] = c11v;
        }
    }
}
#endif
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
extern "C" {
void _AVX_MNNGemm16x4(float* C, const float* A, const float* B, const size_t* parameter);
}
static void _packMatMul(float* C, const float* A, const float* B, const size_t* parameter) {
    _AVX_MNNGemm16x4(C, A, B, parameter);
}

ErrorCode StrassenMatrixComputor::_generateTrivalMatMul(const Tensor* A, const Tensor* BT, const Tensor* C) {
    // Generate Trival Matrix Multiply
    auto e = A->length(0);
    auto l = A->length(1);
    auto h = C->length(1);
    MNN_ASSERT(l > 0 && e > 0 && h > 0);
    auto aHost   = A->host<float>();
    auto bHost   = BT->host<float>();
    auto cHost   = C->host<float>();
    std::vector<size_t> parameter(6);
    parameter[0] = e;
    parameter[1] = l;
    parameter[2] = h;
    parameter[3] = C->stride(0) * sizeof(float);
    parameter[4] = A->stride(0) * sizeof(float);
    parameter[5] = (BT->stride(0) - BT->length(1) * BT->length(2)) * sizeof(float);
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    mFunctions.emplace_back(std::make_pair([aHost, bHost, cHost, parameter](int tId) {
        _packMatMul(cHost, aHost, bHost, parameter.data());
    }, numberThread));
    return NO_ERROR;
}

#define MNNMATRIX_SUB_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub) \
for (int y = tId; y < lSub; y+=numberThread) {\
MNNMatrixSubCommon(c + y * cStride, a + y * aStride, b + y * bStride, widthC4, 0, 0, 0, 1);\
}\

#define MNNMATRIX_ADD_MULTITHREAD(c, a, b, widthC4, cStride, aStride, bStride, lSub) \
for (int y = tId; y < lSub; y+=numberThread) {\
MNNMatrixAddCommon(c + y * cStride, a + y * aStride, b + y * bStride, widthC4, 0, 0, 0, 1);\
}\


ErrorCode StrassenMatrixComputor::_generateMatMul(const Tensor* A, const Tensor* BT, const Tensor* C,
                                                  int currentDepth) {
    auto l = A->length(1);
    auto e = A->length(0);
    auto h = C->length(1);
    auto apack = A->length(2);
    auto bpack = BT->length(2);
    auto cpack = C->length(2);

    auto eSub = e / 2;
    auto lSub = l / 2;
    auto hSub = h / 2;

    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;

    /*
     Compute the memory read / write cost for expand
     Matrix Mul need eSub*lSub*hSub*(apack + bpack + cpack), Matrix Add/Sub need x*y*pack*3 (2 read 1 write)
     */
    float saveCost = (eSub * lSub * hSub) * (apack + bpack + cpack) - 4 * (eSub * lSub) * apack * 3 -
                     4 * (bpack * lSub * hSub * 3) - 7 * (eSub * hSub * 3) * cpack;
    if (currentDepth >= mMaxDepth || e <= CONVOLUTION_TILED_NUMBER || l % 2 != 0 || saveCost < 0.0f) {
        return _generateTrivalMatMul(A, BT, C);
    }

    // Strassen Construct
    auto bn = backend();
    currentDepth += 1;
    auto AS                = std::vector<int>{eSub, lSub, apack};
    auto BS                = std::vector<int>{hSub, lSub, bpack};
    auto CS                = std::vector<int>{eSub, hSub, cpack};

    AddTensor X(Tensor::createDevice<float>(AS), bn);
    AddTensor Y(Tensor::createDevice<float>(BS), bn);
    AddTensor CX(Tensor::createDevice<float>(CS), bn);
    if (!X.valid() || !Y.valid() || !CX.valid()) {
        return OUT_OF_MEMORY;
    }

    auto xAddr = X->host<float>();
    auto yAddr = Y->host<float>();
    auto cAddr = CX->host<float>();

    auto aStride = A->stride(0);
    auto a11     = A->host<float>() + 0 * apack * lSub + 0 * aStride * eSub;
    auto a21     = A->host<float>() + 0 * apack * lSub + 1 * aStride * eSub;
    auto a12     = A->host<float>() + 1 * apack * lSub + 0 * aStride * eSub;
    auto a22     = A->host<float>() + 1 * apack * lSub + 1 * aStride * eSub;

    auto bStride = BT->stride(0);
    auto b11     = BT->host<float>() + 0 * bpack * lSub + 0 * bStride * hSub;
    auto b12     = BT->host<float>() + 0 * bpack * lSub + 1 * bStride * hSub;
    auto b21     = BT->host<float>() + 1 * bpack * lSub + 0 * bStride * hSub;
    auto b22     = BT->host<float>() + 1 * bpack * lSub + 1 * bStride * hSub;

    auto cStride = C->stride(0);
    auto c11     = C->host<float>() + 0 * cpack * hSub + 0 * cStride * eSub;
    auto c21     = C->host<float>() + 0 * cpack * hSub + 1 * cStride * eSub;
    auto c12     = C->host<float>() + 1 * cpack * hSub + 0 * cStride * eSub;
    auto c22     = C->host<float>() + 1 * cpack * hSub + 1 * cStride * eSub;

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
        auto f = [a11, a21, b22, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, apack, bpack](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11, a21, lSub * apack, lSub * apack, aStride, aStride, eSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, b12, lSub * bpack, lSub * bpack, bStride, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C21.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [a22, a21, b11, b12, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, apack, bpack](int tId) {
            MNNMATRIX_ADD_MULTITHREAD(xAddr, a21, a22, lSub * apack, lSub * apack, aStride, aStride, eSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b12, b11, lSub * bpack, lSub * bpack, bStride, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C22.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [a11, b22, xAddr, yAddr, eSub, lSub, hSub, aStride, bStride, numberThread, apack, bpack](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, a11, lSub * apack, lSub * apack, lSub * apack, aStride, eSub);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, b22, yAddr, lSub * bpack, lSub * bpack, bStride, lSub * bpack, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(X.get(), Y.get(), C12.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        auto f = [a12, xAddr, eSub, lSub, aStride, numberThread, apack](int tId) {
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a12, xAddr, lSub * apack, lSub * apack, aStride, lSub * apack, eSub);
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
        auto f = [c11, c12, c21, c22, b21, cAddr, yAddr, eSub, lSub, hSub, bStride, cStride, numberThread, bpack, cpack](int tId) {
            for (int y = tId; y < eSub; y+=numberThread) {
                MNNStrassenMergeCFunction(c11 + y * cStride, c12 + y * cStride, c21 + y * cStride, c22 + y * cStride, cAddr + y * hSub * cpack, 0, hSub * cpack, 1);
            }
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, b21, lSub * bpack, lSub * bpack, lSub * bpack, bStride, hSub);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(A22.get(), Y.get(), C11.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        auto f0 = [c11, c21, eSub, hSub, cStride, numberThread, cpack](int tId) {
            auto cw = hSub * cpack;
            MNNMATRIX_SUB_MULTITHREAD(c21, c21, c11, cw, cStride, cStride, cStride, eSub);
        };
        mFunctions.emplace_back(std::make_pair(f0, numberThread));
        auto code = _generateMatMul(A12.get(), B21.get(), C11.get(), currentDepth);
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, cAddr, eSub, hSub, cStride, numberThread, cpack](int tId) {
            auto cw = hSub * cpack;
            MNNMATRIX_ADD_MULTITHREAD(c11, c11, cAddr, cw, cStride, cStride, hSub * cpack, eSub);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
    }
    // Bottom
    if (e % 2 != 0) {
        auto aLast = A->host<float>() + eSub * 2 * aStride;
        auto cLast = C->host<float>() + eSub * 2 * cStride;
        PTensor ALast(Tensor::create<float>(std::vector<int>{1, l, apack}, aLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{1, h, cpack}, cLast));
        ALast->setStride(0, aStride);
        CLast->setStride(0, cStride);
        _generateTrivalMatMul(ALast.get(), BT, CLast.get());
    }
    // Right
    if (h % 2 != 0) {
        auto length = e % 2 == 0 ? e : e-1;
        auto bLast = BT->host<float>() + hSub * 2 * bStride;
        auto cLast = C->host<float>() + hSub * 2 * cpack;
        PTensor ALast(Tensor::create<float>(std::vector<int>{length, l, apack}, A->host<float>()));
        PTensor BLast(Tensor::create<float>(std::vector<int>{1, l, bpack}, bLast));
        PTensor CLast(Tensor::create<float>(std::vector<int>{length, 1, cpack}, cLast));
        ALast->setStride(0, aStride);
        BLast->setStride(0, bStride);
        CLast->setStride(0, cStride);
        _generateTrivalMatMul(ALast.get(), BLast.get(), CLast.get());
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
