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
#include "core/BufferAllocator.hpp"

namespace MNN {
class AutoMemory {
public:
    AutoMemory(int size, BufferAllocator* allocator) {
        mContent = allocator->alloc(size);
        mAllocator = allocator;
    }
    ~ AutoMemory() {
        if (nullptr != mContent.first) {
            mAllocator->free(mContent);
        }
    }
    const std::pair<void*, int>& get() const {
        return mContent;
    }
private:
    std::pair<void*, int> mContent;
    BufferAllocator* mAllocator;
};

StrassenMatrixComputor::StrassenMatrixComputor(Backend* bn, bool multithread, int maxDepth) : mBackend(bn) {
    mMaxDepth = maxDepth;
    mSupportMultiThread = multithread;
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
    auto bExtraStride = bStride - UP_DIV(l, lP)*lP*hP * core->bytes;
    MNN_ASSERT(bExtraStride >= 0);
    auto tileBufferBasic = static_cast<CPUBackend*>(backend())->getBufferAllocator()->alloc(numberThread * UP_DIV(l, lP) * eP * lP * bytes);
    if (nullptr == tileBufferBasic.first) {
        return OUT_OF_MEMORY;
    }
    auto tileHostOrigin  = (uint8_t*)tileBufferBasic.first + tileBufferBasic.second;
    int unitNumber = e / eP;
    int xCount     = e - unitNumber * eP;
    auto eReal = aStride / core->bytes / core->pack;
    mFunctions.emplace_back(
        std::make_pair([cStride, l, h, xCount, AT, BT, CT, COT, tileHostOrigin, unitNumber, bExtraStride, numberThread, eReal, eP, active, this](int tId) {
            auto core = static_cast<CPUBackend*>(backend())->functions();
            size_t parameters[6];
            parameters[0] = xCount * core->bytes;
            parameters[1] = l;
            parameters[2] = h;
            parameters[3] = cStride;
            parameters[4] = 0;
            parameters[5] = bExtraStride;
            auto tileHost = tileHostOrigin + eP * parameters[1] * tId * core->bytes;
            const float* postParametersPtr = nullptr;
            if (!active.empty()) {
                postParametersPtr = active.data();
            }
            auto aHost = mStack[AT.stackIndex] + AT.offsetBytes;
            auto bHost = mStack[BT.stackIndex] + BT.offsetBytes;
            auto cHost = mStack[CT.stackIndex] + CT.offsetBytes;
            const uint8_t* biasPtr = nullptr;
            if (-1 != COT.stackIndex) {
                biasPtr = mStack[COT.stackIndex] + COT.offsetBytes;
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
                core->MNNPackC4ForMatMul_A((float*)(tileHost), (const float**)(&aStart), info, stride);
                core->MNNPackedMatMul((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, parameters, postParametersPtr, (const float*)biasPtr);
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
                core->MNNPackC4ForMatMul_A((float*)(tileHost), (const float**)(&aStart), info, stride);
                core->MNNPackedMatMulRemain((float*)(cHost + xStart * packUnit), (float*)tileHost, (float*)bHost, xCount, parameters, postParametersPtr, (const float*)biasPtr);
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
    mStack.emplace_back((uint8_t*)CAddr.get().first + CAddr.get().second);

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
        tempB.offsetBytes = BT.offsetBytes + lS * hP * core->bytes;
        auto code = _generateTrivalMatMul(e, lE-lS, h, tempA, tempB, CTemp, Empty, {});
        if (NO_ERROR != code) {
            return code;
        }
        // Add CTemp to C
        auto f1 = [CT, CTemp, e, cHeight, numberThread, core, this](int tId) {
            auto c11Ptr = mStack[CT.stackIndex] + CT.offsetBytes;
            auto xAddr = mStack[CTemp.stackIndex] + CTemp.offsetBytes;
            MNNMATRIX_ADD_MULTITHREAD(c11Ptr, c11Ptr, xAddr, e, CT.lineStrideBytes, CT.lineStrideBytes, CTemp.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
    }
    if (!postParameters.empty() && COT.stackIndex >= 0) {
        if (1 == numberThread) {
            auto postFunction = [CT, COT, e, cHeight, numberThread, postParameters, core, this](int tId) {
                auto biasPtr = (const float*)(mStack[COT.stackIndex] + COT.offsetBytes);
                auto width = e;
                auto height = cHeight;
                auto c11Ptr = mStack[CT.stackIndex] + CT.offsetBytes;
                core->MNNAxByClampBroadcastUnit((float*)c11Ptr, (float*)c11Ptr, biasPtr, width, CT.lineStrideBytes / core->bytes, CT.lineStrideBytes / core->bytes, height, postParameters.data());
            };
            mFunctions.emplace_back(std::make_pair(postFunction, 1));
        } else {
            auto postFunction = [CT, COT, e, cHeight, numberThread, postParameters, core, this](int tId) {
                auto width = e;
                auto height = cHeight;
                auto c11Ptr = mStack[CT.stackIndex] + CT.offsetBytes;
                auto biasPtr = mStack[COT.stackIndex] + COT.offsetBytes;
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
    AutoMemory YAddr(hSub * lSub * core->bytes, allocator);
    AutoMemory XAddr(maxlH * eSub * core->bytes, allocator);
    if (nullptr == XAddr.get().first || nullptr == YAddr.get().first) {
        return OUT_OF_MEMORY;
    }
    MatrixInfo Y;
    Y.stackIndex = (int)mStack.size();
    mStack.emplace_back((uint8_t*)YAddr.get().first + YAddr.get().second);
    Y.offsetBytes = 0;
    Y.lineStrideBytes = lSub * core->bytes * hP;
    MatrixInfo X;
    X.stackIndex = (int)mStack.size();
    X.offsetBytes = 0;
    X.lineStrideBytes = eSub * core->bytes * core->pack;
    mStack.emplace_back((uint8_t*)XAddr.get().first + XAddr.get().second);

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
    b21.offsetBytes = BT.offsetBytes + lSub * hP * core->bytes;
    MatrixInfo b22 = BT;
    b22.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * (hSub / hP) + lSub * hP * core->bytes;
    
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
            auto xAddr = mStack[X.stackIndex] + X.offsetBytes;
            auto yAddr = mStack[Y.stackIndex] + Y.offsetBytes;
            auto a11Ptr = mStack[a11.stackIndex] + a11.offsetBytes;
            auto a21Ptr = mStack[a21.stackIndex] + a21.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, a11Ptr, a21Ptr, eSub, X.lineStrideBytes, a11.lineStrideBytes, a21.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, mStack[b22.stackIndex] + b22.offsetBytes, mStack[b12.stackIndex] + b12.offsetBytes, bWidth, Y.lineStrideBytes, b22.lineStrideBytes, b12.lineStrideBytes, bHeight, core);
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
            MNNMATRIX_ADD_MULTITHREAD(mStack[X.stackIndex] + X.offsetBytes, mStack[a21.stackIndex] + a21.offsetBytes, mStack[a22.stackIndex] + a22.offsetBytes , eSub, X.lineStrideBytes, a21.lineStrideBytes, a22.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(mStack[Y.stackIndex] + Y.offsetBytes, mStack[b12.stackIndex] + b12.offsetBytes, mStack[b11.stackIndex] + b11.offsetBytes, bWidth, Y.lineStrideBytes, b12.lineStrideBytes, b11.lineStrideBytes, bHeight, core);
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
            auto xAddr = mStack[X.stackIndex] + X.offsetBytes;
            auto yAddr = mStack[Y.stackIndex] + Y.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, xAddr, mStack[a11.stackIndex] + a11.offsetBytes, eSub, X.lineStrideBytes, X.lineStrideBytes, a11.lineStrideBytes, aHeight, core);
            MNNMATRIX_SUB_MULTITHREAD(yAddr, mStack[b22.stackIndex] + b22.offsetBytes, yAddr, bWidth, Y.lineStrideBytes, b22.lineStrideBytes, Y.lineStrideBytes, bHeight, core);
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
            auto xAddr = mStack[X.stackIndex] + X.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(xAddr, mStack[a12.stackIndex] + a12.offsetBytes, xAddr, eSub, X.lineStrideBytes, a12.lineStrideBytes, X.lineStrideBytes, aHeight, core);
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
                core->MNNStrassenMergeCFunction((float*)(mStack[c11.stackIndex] + c11.offsetBytes + y * c11.lineStrideBytes), (float*)(mStack[c12.stackIndex] + c12.offsetBytes + y * c12.lineStrideBytes), (float*)(mStack[c21.stackIndex] + c21.offsetBytes + y * c21.lineStrideBytes), (float*)(mStack[c22.stackIndex] + c22.offsetBytes + y * c22.lineStrideBytes), (float*)(mStack[X.stackIndex] + X.offsetBytes + y * X.lineStrideBytes), 0, eSub, 1);
            }
            auto yAddr = mStack[Y.stackIndex] + Y.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(yAddr, yAddr, mStack[b21.stackIndex] + b21.offsetBytes, bWidth, Y.lineStrideBytes, Y.lineStrideBytes, b21.lineStrideBytes, bHeight, core);
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
            auto c21Addr = mStack[c21.stackIndex] + c21.offsetBytes;
            MNNMATRIX_SUB_MULTITHREAD(c21Addr, c21Addr, mStack[c11.stackIndex] + c11.offsetBytes, cw, c21.lineStrideBytes, c21.lineStrideBytes, c11.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f0, numberThread));
        auto code = _generateMatMul(eSub, lSub, hSub, a12, b21, c11, Empty, currentDepth, {});
        if (code != NO_ERROR) {
            return code;
        }
        auto f1 = [c11, X, eSub, cHeight, numberThread, core, this](int tId) {
            auto cw = eSub;
            auto c11Ptr = mStack[c11.stackIndex] + c11.offsetBytes;
            auto xAddr = mStack[X.stackIndex] + X.offsetBytes;
            MNNMATRIX_ADD_MULTITHREAD(c11Ptr, c11Ptr, xAddr, cw, c11.lineStrideBytes, c11.lineStrideBytes, X.lineStrideBytes, cHeight, core);
        };
        mFunctions.emplace_back(std::make_pair(f1, numberThread));
        if (!postParameters.empty() && COT.stackIndex >= 0) {
            if (1 == numberThread) {
                auto postFunction = [c11, COT, eSub, cHeight, numberThread, postParameters, core, this](int tId) {
                    auto biasPtr = (const float*)(mStack[COT.stackIndex] + COT.offsetBytes);
                    auto width = eSub * 2;
                    auto height = cHeight * 2;
                    auto c11Ptr = mStack[c11.stackIndex] + c11.offsetBytes;
                    core->MNNAxByClampBroadcastUnit((float*)c11Ptr, (float*)c11Ptr, biasPtr, width, c11.lineStrideBytes / core->bytes, c11.lineStrideBytes / core->bytes, height, postParameters.data());
                };
                mFunctions.emplace_back(std::make_pair(postFunction, numberThread));
            } else {
                auto postFunction = [c11, COT, eSub, cHeight, numberThread, postParameters, core, this](int tId) {
                    auto width = eSub * 2;
                    auto height = cHeight * 2;
                    auto c11Ptr = mStack[c11.stackIndex] + c11.offsetBytes;
                    auto biasPtr = mStack[COT.stackIndex] + COT.offsetBytes;
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
    mFunctions.clear();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    MNN_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    MNN_ASSERT(outputs.size() == 1);
    auto A  = inputs[0];
    auto B  = inputs[1];
    auto C  = outputs[0];
    Tensor* CO = nullptr;
    auto l = B->length(1);
    if (inputL != 0) {
        l = inputL;
    }
    auto e = A->length(1);
    auto h = std::min(C->length(0) * core->pack, B->length(0) * B->length(2));
    if (inputH != 0) {
        h = inputH;
    }
    mStack = {A->host<uint8_t>(), B->host<uint8_t>(), C->host<uint8_t>()};
    MatrixInfo a,b,c,bias;
    bias.stackIndex = -1;
    if (inputs.size() > 2) {
        CO = inputs[2];
        bias.stackIndex = 3;
        bias.offsetBytes = 0;
        mStack.emplace_back(CO->host<uint8_t>());
    }
    a.stackIndex = 0;
    a.lineStrideBytes = A->stride(0) * core->bytes;
    a.offsetBytes = 0;
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);

    b.stackIndex = 1;
    b.lineStrideBytes = UP_DIV(l, lP) * lP * hP * core->bytes;
    b.offsetBytes = 0;
    
    c.stackIndex = 2;
    c.lineStrideBytes = C->stride(0) * core->bytes;
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
