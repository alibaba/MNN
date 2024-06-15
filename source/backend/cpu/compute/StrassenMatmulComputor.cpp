//
//  StrassenMatmulComputor.cpp
//  MNN
//
//  Created by MNN on 2019/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StrassenMatmulComputor.hpp"
#include "DenseConvolutionTiledExecutor.hpp"
#include "CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"
#include <string.h>
#include <limits.h>
#include "core/AutoStorage.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "math/Vec.hpp"
#include "math/Matrix.hpp"
#include "core/BufferAllocator.hpp"

namespace MNN {
class AutoMemory {
public:
    AutoMemory(int size, BufferAllocator* allocator) {
        mContent = allocator->alloc(size);
        mAllocator = allocator;
    }
    ~ AutoMemory() {
        if (!mContent.invalid()) {
            mAllocator->free(mContent);
        }
    }
    const MemChunk& get() const {
        return mContent;
    }
private:
    MemChunk mContent;
    BufferAllocator* mAllocator;
};

StrassenMatrixComputor::StrassenMatrixComputor(Backend* bn, bool multithread, int maxDepth, uint8_t* dequantAlpha, uint8_t* dequantBias, int32_t dequantBits) : mBackend(bn) {
    mMaxDepth = maxDepth;
    mSupportMultiThread = multithread;
    mDequantBias = dequantBias;
    mDequantAlpha = dequantAlpha;
    mDequantBits = dequantBits;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mWeightBytes = core->bytes;
    if (mDequantBits == 8 || mDequantBits == 4) {
        mWeightBytes = (float)mDequantBits / 8;
    }
};
StrassenMatrixComputor::~StrassenMatrixComputor() {
    // Do nothing
}

ErrorCode StrassenMatrixComputor::_generateTrivalMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, const std::vector<float>& active) {
    // Generate Trival Matrix Multiply
    MNN_ASSERT(e > 0);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes    = core->bytes;
    auto aStride = AT.lineStrideBytes;
    auto bStride = BT.lineStrideBytes;
    auto cStride = CT.lineStrideBytes;
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    auto bExtraStride = bStride - UP_DIV(l, lP)*lP*hP * mWeightBytes;
    MNN_ASSERT(bExtraStride >= 0);
    auto tileBufferBasic = static_cast<CPUBackend*>(backend())->getBufferAllocator()->alloc(numberThread * UP_DIV(l, lP) * eP * lP * bytes);
    if (tileBufferBasic.invalid()) {
        return OUT_OF_MEMORY;
    }

    int unitNumber = e / eP;
    int xCount     = e - unitNumber * eP;
    auto eReal = aStride / core->bytes / core->pack;
    auto matmulUnit = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;
    const float* dequantAlpha = nullptr;
    const float* dequantBias  = nullptr;
    float weightBytes           = 1;
#ifdef MNN_LOW_MEMORY
    if (nullptr != mDequantAlpha && nullptr != mDequantBias) {
        dequantAlpha = reinterpret_cast<const float*>(mDequantAlpha);
        dequantBias = reinterpret_cast<const float*>(mDequantBias);
        DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(&matmulUnit, &matmulRemain, &weightBytes, mDequantBits, core);
    }
#endif
    mFunctions.emplace_back(
        std::make_pair([cStride, l, h, xCount, AT, BT, CT, COT, tileBufferBasic, unitNumber, bExtraStride, numberThread, eReal, eP, active, matmulUnit, matmulRemain, dequantAlpha, dequantBias, this](int tId) {
            auto core = static_cast<CPUBackend*>(backend())->functions();
            size_t parameters[7];
            parameters[0] = xCount * core->bytes;
            parameters[1] = l;
            parameters[2] = h;
            parameters[3] = cStride;
            parameters[4] = 0;
            parameters[5] = bExtraStride;
            parameters[6] = 0;
            auto tileHost = tileBufferBasic.ptr() + eP * parameters[1] * tId * core->bytes;
            const float* postParametersPtr = nullptr;
            if (!active.empty()) {
                postParametersPtr = active.data();
            }
            auto aHost = mStack[AT.stackIndex].ptr() + AT.offsetBytes;
            auto bHost = mStack[BT.stackIndex].ptr() + BT.offsetBytes;
            auto cHost = mStack[CT.stackIndex].ptr() + CT.offsetBytes;
            const uint8_t* biasPtr = nullptr;
            if (-1 != COT.stackIndex) {
                biasPtr = mStack[COT.stackIndex].ptr() + COT.offsetBytes;
            }
            auto packUnit = core->bytes * core->pack;
            int32_t info[4];
            int32_t stride[4];
            stride[0] = eP;
            stride[1] = (int32_t)parameters[1];
            stride[2] = 0;
            stride[3] = 0;
            info[0] = 1;
            info[1] = eReal;
            info[2] = eP;
            info[3] = 1;
            for (int i = tId; i < unitNumber; i+=numberThread) {
                int xStart    = i * eP;
                auto aStart   = aHost + xStart * packUnit;
                core->MNNPackC4ForMatMul_A((float*)(tileHost), (const float**)(&aStart), info, stride);
                matmulUnit((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, parameters, postParametersPtr, (const float*)biasPtr, dequantAlpha, dequantBias);
            }
            if (tId != numberThread -1) {
                return;
            }
            if (xCount > 0) {
                stride[0] = xCount;
                stride[1] = (int32_t)parameters[1];
                info[2] = xCount;

                int xStart    = unitNumber * eP;
                auto aStart   = aHost + xStart * packUnit;
                // Copy
                core->MNNPackC4ForMatMul_A((float*)(tileHost), (const float**)(&aStart), info, stride);
                matmulRemain((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, xCount, parameters, postParametersPtr, (const float*)biasPtr, dequantAlpha, dequantBias);
            }
        }, numberThread));
    static_cast<CPUBackend*>(backend())->getBufferAllocator()->free(tileBufferBasic);
    return NO_ERROR;
}

#define MNNMATRIX_SUB_MULTITHREAD(c_, a_, b_, widthC4, cStride, aStride, bStride, lSub, core) \
{\
auto c = c_;\
auto b = b_;\
auto a = a_;\
for (int y = tId; y < lSub; y+=numberThread) {\
core->MNNMatrixSub((float*)(c + y * cStride), (float*)(a + y * aStride), (float*)(b + y * bStride), widthC4, 0, 0, 0, 1);\
}\
}

#define MNNMATRIX_ADD_MULTITHREAD(c_, a_, b_, widthC4, cStride, aStride, bStride, lSub, core) \
{\
auto c = c_;\
auto b = b_;\
auto a = a_;\
for (int y = tId; y < lSub; y+=numberThread) {\
core->MNNMatrixAdd((float*)(c + y * cStride), (float*)(a + y * aStride), (float*)(b + y * bStride), widthC4, 0, 0, 0, 1);\
}\
}

ErrorCode StrassenMatrixComputor::_generateBasicMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, const std::vector<float>& postParameters) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    int lLimit = 32768 / (std::min(eP, e) + hP);
    if (l <= lLimit) {
        return _generateTrivalMatMul(e, l, h, AT, BT, CT, COT, postParameters);
    }
    {
        auto lUnit = std::max(lP, core->pack);
        lLimit = lLimit / lUnit * lUnit;
    }
    int unit = UP_DIV(l, lLimit);
    auto allocator = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    AutoMemory CAddr(e * UP_DIV(h, core->pack) * core->pack * core->bytes, allocator);
    MatrixInfo CTemp;
    CTemp.stackIndex = (int)mStack.size();
    CTemp.offsetBytes = 0;
    CTemp.lineStrideBytes = e * core->bytes * core->pack;
    mStack.emplace_back(CAddr.get());

    MatrixInfo Empty;
    Empty.stackIndex = -1;
    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    auto cHeight = UP_DIV(h, core->pack);

    for (int i=0; i<unit; ++i) {
        int lS = i * lLimit;
        int lE = lS + lLimit;
        if (lE > l) {
            lE = l;
        }
        if (0 == i) {
            // First write to output
            auto code = _generateTrivalMatMul(e, lE-lS, h, AT, BT, CT, Empty, {});
            if (NO_ERROR != code) {
                return code;
            }
            continue;
        }
        MatrixInfo tempA = AT;
        MatrixInfo tempB = BT;
        tempA.offsetBytes = AT.offsetBytes + lS / core->pack * AT.lineStrideBytes;
        // tempB.offsetBytes = BT.offsetBytes + lS * hP * core->bytes;
        tempB.offsetBytes = BT.offsetBytes + lS * hP * mWeightBytes;
        auto code = _generateTrivalMatMul(e, lE-lS, h, tempA, tempB, CTemp, Empty, {});
        if (NO_ERROR != code) {
            return code;
        }
        // Add CTemp to C
        auto f1 = [CT, CTemp, e, cHeight, numberThread, core, this](int tId) {
            auto c11Ptr = mStack[CT.stackIndex].ptr() + CT.offsetBytes;
            auto xAddr = mStack[CTemp.stackIndex].ptr() + CTemp.offsetBytes;
            MNNMATRIX_ADD_MULTITHREAD(c11Ptr, c11Ptr, xAddr, e, CT.lineStrideBytes, CT.lineStrideBytes, CTemp.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
    }
    if (!postParameters.empty() && COT.stackIndex >= 0) {
        if (1 == numberThread) {
            auto postFunction = [CT, COT, e, cHeight, numberThread, postParameters, core, this](int tId) {
                auto biasPtr = (const float*)(mStack[COT.stackIndex].ptr() + COT.offsetBytes);
                auto width = e;
                auto height = cHeight;
                auto c11Ptr = mStack[CT.stackIndex].ptr() + CT.offsetBytes;
                core->MNNAxByClampBroadcastUnit((float*)c11Ptr, (float*)c11Ptr, biasPtr, width, CT.lineStrideBytes / core->bytes, CT.lineStrideBytes / core->bytes, height, postParameters.data());
            };
            mFunctions.emplace_back(std::make_pair(postFunction, 1));
        } else {
            auto postFunction = [CT, COT, e, cHeight, numberThread, postParameters, core, this](int tId) {
                auto width = e;
                auto height = cHeight;
                auto c11Ptr = mStack[CT.stackIndex].ptr() + CT.offsetBytes;
                auto biasPtr = mStack[COT.stackIndex].ptr() + COT.offsetBytes;
                for (int y = tId; y < height; y+=numberThread) {
                    core->MNNAxByClampBroadcastUnit((float*)(c11Ptr + y * CT.lineStrideBytes), (float*)(c11Ptr + y * CT.lineStrideBytes), (const float*)(biasPtr + y * core->bytes * core->pack), width, 0, 0, 1, postParameters.data());
                }
            };
            mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
        }
    }
    return NO_ERROR;
}

ErrorCode StrassenMatrixComputor::_generateMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, int currentDepth, const std::vector<float>& postParameters) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto aUnit = core->pack;

    auto numberThread = mSupportMultiThread ? ((CPUBackend*)backend())->threadNumber() : 1;
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    MNN_ASSERT(hP % core->pack == 0 || core->pack % hP == 0);
    auto eSub = (e / eP) / 2 * eP;
    auto lMinDiv = std::max(core->pack * 2, 2 * lP);
    auto hSub = (h / std::max(hP, core->pack)) / 2 * std::max(hP, core->pack);
    auto remainH = h - hSub * 2;
    auto remainE = e - eSub * 2;
    int packHUnit = 1;
    if (core->pack > hP) {
        packHUnit = core->pack / hP;
    }
    if (currentDepth >= mMaxDepth || eSub == 0 || hSub == 0 || l % (2 * core->pack) != 0 || l % (2 * lP) || l % (2 * packHUnit) != 0) {
        return _generateBasicMatMul(e, l, h, AT, BT, CT, COT, postParameters);
    }
    auto lSub = l / 2;
    auto lSubUnit = lSub / core->pack;

    auto bWidth = lSub * hP / core->pack;
    auto aHeight = lSub / core->pack;
    auto cHeight = hSub / core->pack;
    auto bHeight = hSub / hP;
    /*
     Compute the memory read / write cost for expand
     */
    auto bHSub = bHeight;
    float AComputeCost = 4 * ((float)eSub * lSub);
    float BComputeCost = 4 * (float)lSub * bHSub * hP;
    float CComputeCost = 7 * (float)eSub * hSub;
    float saveMatMulCost = (e / eP) * (aUnit * eP * hSub / core->pack + lSubUnit * eP * aUnit + lSub * bHSub * hP);

    const float penalty = core->penalty;//FIXME: Find beter way to set it
    float saveCost = saveMatMulCost - (AComputeCost + BComputeCost + CComputeCost) * penalty;
    if (saveCost <= 0.0f) {
        return _generateBasicMatMul(e, l, h, AT, BT, CT, COT, postParameters);
    }

    // Strassen Construct
    auto bn = backend();
    auto allocator = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    currentDepth += 1;
    auto maxlH = std::max(lSub, hSub);
    AutoMemory YAddr(hSub * lSub * mWeightBytes, allocator);
    AutoMemory XAddr(maxlH * eSub * core->bytes, allocator);
    if (XAddr.get().invalid() || YAddr.get().invalid()) {
        return OUT_OF_MEMORY;
    }
    MatrixInfo Y;
    Y.stackIndex = (int)mStack.size();
    mStack.emplace_back(YAddr.get());
    Y.offsetBytes = 0;
    Y.lineStrideBytes = lSub * mWeightBytes * hP;
    MatrixInfo X;
    X.stackIndex = (int)mStack.size();
    X.offsetBytes = 0;
    X.lineStrideBytes = eSub * core->bytes * core->pack;
    mStack.emplace_back(XAddr.get());

    MatrixInfo CX;
    CX.stackIndex = X.stackIndex;
    CX.offsetBytes = 0;
    CX.lineStrideBytes = eSub * core->bytes * core->pack;
    
    MatrixInfo a11 = AT;
    MatrixInfo a12 = AT;
    a12.offsetBytes = AT.offsetBytes + AT.lineStrideBytes * lSubUnit;
    MatrixInfo a21 = AT;
    a21.offsetBytes = AT.offsetBytes + eSub * core->pack * core->bytes;
    MatrixInfo a22 = AT;
    a22.offsetBytes = AT.offsetBytes + eSub * core->pack * core->bytes + AT.lineStrideBytes * lSubUnit;
    
    MatrixInfo b11 = BT;
    MatrixInfo b12 = BT;
    b12.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * (hSub / hP);
    MatrixInfo b21 = BT;
    b21.offsetBytes = BT.offsetBytes + lSub * hP * mWeightBytes;
    MatrixInfo b22 = BT;
    b22.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * (hSub / hP) + lSub * hP * mWeightBytes;
    
    MatrixInfo c11 = CT;
    MatrixInfo c12 = CT;
    c12.offsetBytes = CT.offsetBytes + CT.lineStrideBytes * (hSub / core->pack);
    MatrixInfo c21 = CT;
    c21.offsetBytes = CT.offsetBytes + eSub * core->pack * core->bytes;
    MatrixInfo c22 = CT;
    c22.offsetBytes = CT.offsetBytes + eSub * core->pack * core->bytes + CT.lineStrideBytes * (hSub / core->pack);

    MatrixInfo Empty;
    Empty.stackIndex = -1;

    {
        // S3=A11-A21, T3=B22-B12, P7=S3*T3
        auto f = [a11, a21, b22, b12, X, Y, eSub, lSub, hSub, numberThread, core, hP, this, bWidth, aHeight, bHeight](int tId) {
            auto xAddr = mStack[X.stackIndex].ptr() + X.offsetBytes;
            auto yAddr = mStack[Y.stackIndex].ptr() + Y.offsetBytes;
            auto a11Ptr = mStack[a11.stackIndex].ptr() + a11.offsetBytes;
            auto a21Ptr = mStack[a21.stackIndex].ptr() + a21.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11Ptr, a21Ptr, eSub, X.lineStrideBytes, a11.lineStrideBytes, a21.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, mStack[b22.stackIndex].ptr() + b22.offsetBytes, mStack[b12.stackIndex].ptr() + b12.offsetBytes, bWidth, Y.lineStrideBytes, b22.lineStrideBytes, b12.lineStrideBytes, bHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c21, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [a22, a21, b11, b12, X, Y, eSub, lSub, hSub, numberThread, hP, core, this, bWidth, aHeight, bHeight](int tId) {
            MNNMATRIX_ADD_MULTITHREAD(mStack[X.stackIndex].ptr() + X.offsetBytes, mStack[a21.stackIndex].ptr() + a21.offsetBytes, mStack[a22.stackIndex].ptr() + a22.offsetBytes , eSub, X.lineStrideBytes, a21.lineStrideBytes, a22.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(mStack[Y.stackIndex].ptr() + Y.offsetBytes, mStack[b12.stackIndex].ptr() + b12.offsetBytes, mStack[b11.stackIndex].ptr() + b11.offsetBytes, bWidth, Y.lineStrideBytes, b12.lineStrideBytes, b11.lineStrideBytes, bHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c22, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [a11, b22, X, Y, eSub, lSub, hSub, numberThread, hP, core, this, bWidth, aHeight, bHeight](int tId) {
            auto xAddr = mStack[X.stackIndex].ptr() + X.offsetBytes;
            auto yAddr = mStack[Y.stackIndex].ptr() + Y.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, mStack[a11.stackIndex].ptr() + a11.offsetBytes, eSub, X.lineStrideBytes, X.lineStrideBytes, a11.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, mStack[b22.stackIndex].ptr() + b22.offsetBytes, yAddr, bWidth, Y.lineStrideBytes, b22.lineStrideBytes, Y.lineStrideBytes, bHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c12, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        auto f = [a12, X, eSub, aHeight, numberThread, core, this](int tId) {
            auto xAddr = mStack[X.stackIndex].ptr() + X.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, mStack[a12.stackIndex].ptr() + a12.offsetBytes, xAddr, eSub, X.lineStrideBytes, a12.lineStrideBytes, X.lineStrideBytes, aHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, X, b22, c11, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        code = _generateMatMul(eSub, lSub, hSub, a11, b11, CX, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
        auto f = [c11, c12, c21, c22, b21, X, Y, eSub, bWidth, cHeight, bHeight, numberThread, core, this](int tId) {
            for (int y = tId; y < cHeight; y+=numberThread) {
                core->MNNStrassenMergeCFunction((float*)(mStack[c11.stackIndex].ptr() + c11.offsetBytes + y * c11.lineStrideBytes), (float*)(mStack[c12.stackIndex].ptr() + c12.offsetBytes + y * c12.lineStrideBytes), (float*)(mStack[c21.stackIndex].ptr() + c21.offsetBytes + y * c21.lineStrideBytes), (float*)(mStack[c22.stackIndex].ptr() + c22.offsetBytes + y * c22.lineStrideBytes), (float*)(mStack[X.stackIndex].ptr() + X.offsetBytes + y * X.lineStrideBytes), 0, eSub, 1);
            }
            auto yAddr = mStack[Y.stackIndex].ptr() + Y.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, mStack[b21.stackIndex].ptr() + b21.offsetBytes, bWidth, Y.lineStrideBytes, Y.lineStrideBytes, b21.lineStrideBytes, bHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, a22, Y, c11, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        auto f0 = [c11, c21, eSub, cHeight, numberThread, core, this](int tId) {
            auto cw = eSub;
            auto c21Addr = mStack[c21.stackIndex].ptr() + c21.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(c21Addr, c21Addr, mStack[c11.stackIndex].ptr() + c11.offsetBytes, cw, c21.lineStrideBytes, c21.lineStrideBytes, c11.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f0, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, a12, b21, c11, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, X, eSub, cHeight, numberThread, core, this](int tId) {
            auto cw = eSub;
            auto c11Ptr = mStack[c11.stackIndex].ptr() + c11.offsetBytes;
            auto xAddr = mStack[X.stackIndex].ptr() + X.offsetBytes;
            MNNMATRIX_ADD_MULTITHREAD(c11Ptr, c11Ptr, xAddr, cw, c11.lineStrideBytes, c11.lineStrideBytes, X.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
        if (!postParameters.empty() && COT.stackIndex >= 0) {
            if (1 == numberThread) {
                auto postFunction = [c11, COT, eSub, cHeight, numberThread, postParameters, core, this](int tId) {
                    auto biasPtr = (const float*)(mStack[COT.stackIndex].ptr() + COT.offsetBytes);
                    auto width = eSub * 2;
                    auto height = cHeight * 2;
                    auto c11Ptr = mStack[c11.stackIndex].ptr() + c11.offsetBytes;
                    core->MNNAxByClampBroadcastUnit((float*)c11Ptr, (float*)c11Ptr, biasPtr, width, c11.lineStrideBytes / core->bytes, c11.lineStrideBytes / core->bytes, height, postParameters.data());
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            } else {
                auto postFunction = [c11, COT, eSub, cHeight, numberThread, postParameters, core, this](int tId) {
                    auto width = eSub * 2;
                    auto height = cHeight * 2;
                    auto c11Ptr = mStack[c11.stackIndex].ptr() + c11.offsetBytes;
                    auto biasPtr = mStack[COT.stackIndex].ptr() + COT.offsetBytes;
                    for (int y = tId; y < height; y+=numberThread) {
                        core->MNNAxByClampBroadcastUnit((float*)(c11Ptr + y * c11.lineStrideBytes), (float*)(c11Ptr + y * c11.lineStrideBytes), (const float*)(biasPtr + y * core->bytes * core->pack), width, 0, 0, 1, postParameters.data());
                    }
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            }
        }
    }
    if (remainH > 0) {
        auto lastH = hSub * 2;
        MatrixInfo CLast = CT;
        CLast.offsetBytes = CT.offsetBytes + CT.lineStrideBytes * (lastH / core->pack);
        MatrixInfo BLast = BT;
        BLast.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * (lastH / hP);
        MatrixInfo Bias = COT;
        if (Bias.stackIndex >= 0) {
            Bias.offsetBytes = COT.offsetBytes + core->bytes * lastH;
        }
        auto code = _generateBasicMatMul(eSub * 2, l, remainH, AT, BLast, CLast, Bias, postParameters);
        if (NO_ERROR != code) {
            return code;
        }
    }
    if (remainE > 0) {
        MatrixInfo CLast = CT;
        CLast.offsetBytes = CT.offsetBytes + eSub * 2 * core->pack * core->bytes;
        MatrixInfo ALast = AT;
        ALast.offsetBytes = AT.offsetBytes + eSub * 2 * core->pack * core->bytes;

        auto code = _generateBasicMatMul(remainE, l, h, ALast, BT, CLast, COT, postParameters);
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}

void StrassenMatrixComputor::onReset() {
    mStack.clear();
    mFunctions.clear();
}

ErrorCode StrassenMatrixComputor::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const std::vector<float>& postParameters, int inputL, int inputH) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mWeightBytes = core->bytes;
    if (mDequantBits == 8 || mDequantBits == 4) {
        mWeightBytes = (float)mDequantBits / 8;
    }
    MNN_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    MNN_ASSERT(outputs.size() == 1);
    auto A  = inputs[0];
    auto B  = inputs[1];
    auto C  = outputs[0];
    auto l = B->length(1);
    if (inputL != 0) {
        l = inputL;
    }
    auto e = A->length(1);
    auto h = std::min(C->length(0) * core->pack, B->length(0) * B->length(2));
    if (inputH != 0) {
        h = inputH;
    }
    int as = A->stride(0);
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    int bs = UP_DIV(l, lP) * lP * hP;
    int cs = C->stride(0);
    MemChunk bias;
    bool useBias = false;
    if (inputs.size() > 2) {
        bias = TensorUtils::getDescribeOrigin(inputs[2])->mem->chunk();
        useBias = true;
    }
    return onEncode(e, l, h, as, bs, cs, TensorUtils::getDescribeOrigin(A)->mem->chunk(), TensorUtils::getDescribeOrigin(B)->mem->chunk(), TensorUtils::getDescribeOrigin(C)->mem->chunk(), useBias, bias, postParameters);
}

ErrorCode StrassenMatrixComputor::onEncode(int e, int l, int h, int as, int bs, int cs, const MemChunk AT, const MemChunk BT, MemChunk CT, bool useBias, const MemChunk Bias, const std::vector<float>& postParameters) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    MatrixInfo a,b,c,bias;
    bias.stackIndex = -1;
    mFunctions.clear();
    mStack = {AT, BT, CT};
    if (useBias) {
        bias.stackIndex = 3;
        bias.offsetBytes = 0;
        mStack.emplace_back(Bias);
    }
    a.stackIndex = 0;
    a.lineStrideBytes = as * core->bytes;
    a.offsetBytes = 0;

    b.stackIndex = 1;
    b.lineStrideBytes = bs * mWeightBytes;
    b.offsetBytes = 0;
    
    c.stackIndex = 2;
    c.lineStrideBytes = cs * core->bytes;
    c.offsetBytes = 0;
    return _generateMatMul(e, l, h, a, b, c, bias, 0, postParameters);
}

void StrassenMatrixComputor::onExecute(const uint8_t* AT, const uint8_t* BT, const uint8_t* COT, uint8_t* CT) {
    if (nullptr != AT) {
        mStack[0] = (uint8_t*)AT;
    }
    if (nullptr != BT) {
        mStack[1] = (uint8_t*)BT;
    }
    if (nullptr != CT) {
        mStack[2] = (uint8_t*)CT;
    }
    if (nullptr != COT) {
        mStack[3] = (uint8_t*)COT;
    }

    // All is done in onResize, just execute it
    for (auto& f : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId);
        }
        MNN_CONCURRENCY_END();
    }
}
} // namespace MNN
