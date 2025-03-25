//
//  StrassenMatmulComputor.cpp
//  MNN
//
//  Created by MNN on 2024/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/StrassenMatmulOpenCLComputor.hpp"
#include "core/TensorUtils.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {
namespace OpenCL {
    
class AutoMemory {
public:
    AutoMemory(int size, OpenCLBackend* backend) {
        mOpenCLBackend = backend;
        mTempTensor.reset(Tensor::createDevice<float>({size}));
        bool res = mOpenCLBackend->onAcquireBuffer(mTempTensor.get(), Backend::DYNAMIC);
        if (!res) {
            MNN_ERROR("Strassen out of memory\n");
        }
        mAddrPtr = openCLBuffer(mTempTensor.get());
    }
    ~ AutoMemory() {
        mOpenCLBackend->onReleaseBuffer(mTempTensor.get(), Backend::DYNAMIC);
    }
    const cl::Buffer& get() const {
        return mAddrPtr;
    }
private:
    cl::Buffer mAddrPtr;
    OpenCLBackend* mOpenCLBackend;
    std::shared_ptr<Tensor> mTempTensor;
};

StrassenMatrixComputor::StrassenMatrixComputor(Backend* bn, int maxDepth) {
    mMaxDepth = maxDepth;
    mOpenCLBackend = static_cast<OpenCLBackend*>(bn);
    mBytes = (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? 2 : 4);
    onReset();
};
StrassenMatrixComputor::~StrassenMatrixComputor() {
    // Do nothing
}
    
ErrorCode StrassenMatrixComputor::_generateCFunction(cl::Buffer ptrC, int offsetC, int elementStrideC, cl::Buffer ptrA, int width, int height, Unit& unit) {
    std::set<std::string> buildOptions;
    int vec_h = 1;
    buildOptions.emplace("-DVEC_H=" + std::to_string(vec_h));
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("strassen_binary_buf", "binary_cfunction_buf", buildOptions);
    auto maxWorkGroupSize      = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

    std::vector<uint32_t> globalWorkSize =  {(uint32_t)UP_DIV(width, 8), (uint32_t)UP_DIV(height, vec_h)};
    
    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, globalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, globalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, ptrC);
    ret |= unit.kernel->get().setArg(index++, offsetC);
    ret |= unit.kernel->get().setArg(index++, elementStrideC);
    ret |= unit.kernel->get().setArg(index++, ptrA);
    ret |= unit.kernel->get().setArg(index++, ptrC);
    ret |= unit.kernel->get().setArg(index++, width);
    ret |= unit.kernel->get().setArg(index++, height);

    MNN_CHECK_CL_SUCCESS(ret, "Strassen setArg BinaryCFunctionExecution");

    std::string name = "binary_cfunction_buf";
    auto localWorkSize = localWS2DDefault(globalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
    
    globalWorkSize[0] = ROUND_UP(globalWorkSize[0], std::max((uint32_t)1, localWorkSize[0]));
    globalWorkSize[1] = ROUND_UP(globalWorkSize[1], std::max((uint32_t)1, localWorkSize[1]));
    
    unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
    unit.localWorkSize  = {localWorkSize[0], localWorkSize[1]};
    mOpenCLBackend->recordKernel2d(unit.kernel, globalWorkSize, localWorkSize);
    return NO_ERROR;
    
}

ErrorCode StrassenMatrixComputor::_generateBinary(cl::Buffer ptrC, cl::Buffer ptrA, cl::Buffer ptrB, int offsetC, int offsetA, int offsetB, int elementStrideC, int elementStrideA, int elementStrideB, int width, int height, bool isAdd, Unit& unit) {
    std::set<std::string> buildOptions;
    if(isAdd) {
        buildOptions.emplace("-DOPERATOR=in0+in1");
    } else {
        buildOptions.emplace("-DOPERATOR=in0-in1");
    }
    int vec_h = 1;
    buildOptions.emplace("-DVEC_H=" + std::to_string(vec_h));
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("strassen_binary_buf", "binary_function_buf", buildOptions);
    auto maxWorkGroupSize      = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

    std::vector<uint32_t> globalWorkSize =  {(uint32_t)UP_DIV(width, 8), (uint32_t)UP_DIV(height, vec_h)};
    int baseOffset[4] = {offsetA, offsetB, offsetC, 0};
    int elementStride[4] = {elementStrideA, elementStrideB, elementStrideC, 0};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, globalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, globalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, ptrA);
    ret |= unit.kernel->get().setArg(index++, ptrB);
    ret |= unit.kernel->get().setArg(index++, ptrC);
    ret |= unit.kernel->get().setArg(index++, baseOffset);
    ret |= unit.kernel->get().setArg(index++, elementStride);

    MNN_CHECK_CL_SUCCESS(ret, "Strassen setArg BinaryExecution");

    std::string name = "binary_function_buf";
    auto localWorkSize = localWS2DDefault(globalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
    
    globalWorkSize[0] = ROUND_UP(globalWorkSize[0], std::max((uint32_t)1, localWorkSize[0]));
    globalWorkSize[1] = ROUND_UP(globalWorkSize[1], std::max((uint32_t)1, localWorkSize[1]));
    
    unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
    unit.localWorkSize  = {localWorkSize[0], localWorkSize[1]};
    mOpenCLBackend->recordKernel2d(unit.kernel, globalWorkSize, localWorkSize);
    return NO_ERROR;
}

ErrorCode StrassenMatrixComputor::_generateBasicMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, int postType, Unit& unit) {
        
    std::set<std::string> buildOptions;
    
    uint32_t layout = 4;
    uint32_t batch = 1;
    
    std::vector<uint32_t> param;
    if(COT.stackIndex < 0 || postType == 0) {
        param = getGemmParams({(uint32_t)e, (uint32_t)h, (uint32_t)l, layout, batch, (uint32_t)0}, {mStack[AT.stackIndex], mStack[BT.stackIndex], mStack[CT.stackIndex]}, mOpenCLBackend->getOpenCLRuntime());
    } else {
        param = getGemmParams({(uint32_t)e, (uint32_t)h, (uint32_t)l, layout, batch, (uint32_t)postType}, {mStack[AT.stackIndex], mStack[BT.stackIndex], mStack[CT.stackIndex], mStack[COT.stackIndex]}, mOpenCLBackend->getOpenCLRuntime());
    }
    int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
    buildOptions.emplace("-DKWG=" + std::to_string(KWG));
    buildOptions.emplace("-DKWI=" + std::to_string(KWI));
    buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
    buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
    buildOptions.emplace("-DMWG=" + std::to_string(MWG));
    buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
    buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
    buildOptions.emplace("-DNWG=" + std::to_string(NWG));
    buildOptions.emplace("-DSA=" + std::to_string(SA));
    buildOptions.emplace("-DSB=" + std::to_string(SB));
    buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
    buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
    buildOptions.emplace("-DVWM=" + std::to_string(VWM));
    buildOptions.emplace("-DVWN=" + std::to_string(VWN));
    if(layout >= 4) {
        buildOptions.emplace("-DOUTPUTMN");
    }

    if(postType > 0) {
        buildOptions.emplace(" -DBIAS_TYPE=" + std::to_string(postType));
    }

    int tileM = MWG;
    int tileN = NWG;
    int localM = MDIMC;
    int localN = NDIMC;
    int alignM = e;
    int alignN = h;
    int alignK = l;
    if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
        buildOptions.emplace("-DUSE_CL_MAD=1");
        buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
    }

    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "Xgemm", buildOptions);
    
    int out_per_thread_m = tileM / localM;
    int out_per_thread_n = tileN / localN;
    
    std::vector<uint32_t> globalWorkSize = {static_cast<uint32_t>(alignM/out_per_thread_m), static_cast<uint32_t>(alignN/out_per_thread_n)};
    std::vector<uint32_t> localWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN)};
    
    float alpha = 1.0;
    float beta = 0.0f;
    // offset_a, offset_b, offset_c, offset_bias
    int offset[4] = {AT.offsetBytes / mBytes, BT.offsetBytes / mBytes, CT.offsetBytes / mBytes, COT.offsetBytes / mBytes};
    // stride_a, stride_b, stride_c, stride_bias
    int stride[4] = {AT.lineStrideBytes / mBytes, BT.lineStrideBytes / mBytes, CT.lineStrideBytes / mBytes, COT.lineStrideBytes / mBytes};

    int idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignM));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignN));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignK));
    ret |= unit.kernel->get().setArg(idx++, alpha);
    ret |= unit.kernel->get().setArg(idx++, beta);
    ret |= unit.kernel->get().setArg(idx++, mStack[AT.stackIndex]);
    ret |= unit.kernel->get().setArg(idx++, mStack[BT.stackIndex]);
    if(postType > 0) {
        ret |= unit.kernel->get().setArg(idx++, mStack[COT.stackIndex]);
    }
    ret |= unit.kernel->get().setArg(idx++, mStack[CT.stackIndex]);
    ret |= unit.kernel->get().setArg(idx++, offset);
    ret |= unit.kernel->get().setArg(idx++, stride);
    
    MNN_CHECK_CL_SUCCESS(ret, "setArg Conv1x1Buf Strassen Kernel Select");
    
    unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
    unit.localWorkSize  = {localWorkSize[0], localWorkSize[1]};
    mOpenCLBackend->recordKernel2d(unit.kernel, globalWorkSize, localWorkSize);

    return NO_ERROR;
}
    
static int getMaxMultiple(int number) {
    if(number % 128 == 0) {
        return 128;
    } else if(number % 64 == 0) {
        return 64;
    } else if(number % 32 == 0) {
        return 32;
    } else if(number % 16 == 0) {
        return 16;
    }
    return 1;
}

ErrorCode StrassenMatrixComputor::_generateMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, int currentDepth, int postType) {

    bool isAligned = (e % 32 == 0 && l % 4 == 0 && h % 32 == 0);
    bool enoughComputation = (e >= 512 && l >= 512 && h >= 512) && (1.0 * e / 1024 * l / 1024 * h / 1024 >= 4.0);
    
    if (currentDepth >= mMaxDepth || !isAligned || !enoughComputation) {// not align or not enough computation
        Unit unit;
        auto res = _generateBasicMatMul(e, l, h, AT, BT, CT, COT, postType, unit);
        mUnits.emplace_back(unit);
        return res;
    }
    int eSub = e / 2;
    int hSub = h / 2;
    int lSub = l / 2;
    
    // Compute expand the memory read and write cost
    float AComputeCost = 1.0 * eSub * lSub * 12 * mBytes;// 4 times, 3 matrix each time
    float BComputeCost = 1.0 * lSub * hSub * 12 * mBytes;// 4 times, 3 matrix each time
    float CComputeCost = 1.0 * eSub * hSub * (8 + 3 * 2) * mBytes;// 3 times, 8 matrix first time, 3 matrix last two times
    // Compute save compute time
    float saveMatMulCost =  1.0 * eSub * lSub * hSub * 2;// 2 for Mul_ADD
    
    // devices peak compute value / memory bandwidth
    const float penalty = 30.0;//FIXME: Find beter way to set it
    float saveCost = saveMatMulCost - (AComputeCost + BComputeCost + CComputeCost) * penalty;
    
    if (saveCost <= 0.0f) {
        Unit unit;
        auto res = _generateBasicMatMul(e, l, h, AT, BT, CT, COT, postType, unit);
        mUnits.emplace_back(unit);
        return res;
    }
    
    // sub_matrix cannot own sufficient tile
    if(getMaxMultiple(e) != getMaxMultiple(eSub)  || getMaxMultiple(h) != getMaxMultiple(eSub) || (lSub % 4 != 0)) {
        Unit unit;
        auto res = _generateBasicMatMul(e, l, h, AT, BT, CT, COT, postType, unit);
        mUnits.emplace_back(unit);
        return res;
    }
    
    // Strassen Construct
    currentDepth += 1;
    
    auto maxlH = std::max(lSub, hSub);
    
    AutoMemory YAddr(hSub * lSub, mOpenCLBackend);
    AutoMemory XAddr(maxlH * eSub, mOpenCLBackend);

    MatrixInfo Y;
    Y.stackIndex = (int)mStack.size();
    mStack.emplace_back(YAddr.get());
    Y.offsetBytes = 0;
    Y.lineStrideBytes = hSub * mBytes;
    MatrixInfo X;
    X.stackIndex = (int)mStack.size();
    X.offsetBytes = 0;
    X.lineStrideBytes = eSub * mBytes;
    mStack.emplace_back(XAddr.get());
    
    MatrixInfo CX;
    CX.stackIndex = X.stackIndex;
    CX.offsetBytes = 0;
    CX.lineStrideBytes = hSub * mBytes;
    
    MatrixInfo a11 = AT;
    MatrixInfo a12 = AT;
    a12.offsetBytes = AT.offsetBytes + AT.lineStrideBytes * lSub;
    MatrixInfo a21 = AT;
    a21.offsetBytes = AT.offsetBytes + eSub * mBytes;
    MatrixInfo a22 = AT;
    a22.offsetBytes = AT.offsetBytes + eSub * mBytes + AT.lineStrideBytes * lSub;
    
    MatrixInfo b11 = BT;
    MatrixInfo b12 = BT;
    b12.offsetBytes = BT.offsetBytes + hSub * mBytes;
    MatrixInfo b21 = BT;
    b21.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * lSub;
    MatrixInfo b22 = BT;
    b22.offsetBytes = BT.offsetBytes + BT.lineStrideBytes * lSub + hSub * mBytes;
    
    MatrixInfo c11 = CT;
    MatrixInfo c12 = CT;
    c12.offsetBytes = CT.offsetBytes + hSub * mBytes;
    MatrixInfo c21 = CT;
    c21.offsetBytes = CT.offsetBytes + CT.lineStrideBytes * eSub;
    MatrixInfo c22 = CT;
    c22.offsetBytes = CT.offsetBytes + CT.lineStrideBytes * eSub + hSub * mBytes;
    
    MatrixInfo Empty;
    Empty.stackIndex = -1;
    
    {
        // S3=A11-A21, T3=B22-B12, P7=S3*T3
        {
            Unit unit;
            _generateBinary(mStack[X.stackIndex], mStack[a11.stackIndex], mStack[a21.stackIndex], X.offsetBytes/mBytes, a11.offsetBytes/mBytes, a21.offsetBytes/mBytes, X.lineStrideBytes/mBytes, a11.lineStrideBytes/mBytes, a21.lineStrideBytes/mBytes, eSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }
        {
            Unit unit;
            _generateBinary(mStack[Y.stackIndex], mStack[b22.stackIndex], mStack[b12.stackIndex], Y.offsetBytes/mBytes, b22.offsetBytes/mBytes, b12.offsetBytes/mBytes, Y.lineStrideBytes/mBytes, b22.lineStrideBytes/mBytes, b12.lineStrideBytes/mBytes, hSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }

        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c21, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        {
            Unit unit;
            _generateBinary(mStack[X.stackIndex], mStack[a21.stackIndex], mStack[a22.stackIndex], X.offsetBytes/mBytes, a21.offsetBytes/mBytes, a22.offsetBytes/mBytes, X.lineStrideBytes/mBytes, a21.lineStrideBytes/mBytes, a22.lineStrideBytes/mBytes, eSub, lSub, true, unit);
            mUnits.emplace_back(unit);
        }
        {
            Unit unit;
            _generateBinary(mStack[Y.stackIndex], mStack[b12.stackIndex], mStack[b11.stackIndex], Y.offsetBytes/mBytes, b12.offsetBytes/mBytes, b11.offsetBytes/mBytes, Y.lineStrideBytes/mBytes, b12.lineStrideBytes/mBytes, b11.lineStrideBytes/mBytes, hSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }
  
        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c22, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        {
            Unit unit;
            _generateBinary(mStack[X.stackIndex], mStack[X.stackIndex], mStack[a11.stackIndex], X.offsetBytes/mBytes, X.offsetBytes/mBytes, a11.offsetBytes/mBytes, X.lineStrideBytes/mBytes, X.lineStrideBytes/mBytes, a11.lineStrideBytes/mBytes, eSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }
        {
            Unit unit;
            _generateBinary(mStack[Y.stackIndex], mStack[b22.stackIndex], mStack[Y.stackIndex], Y.offsetBytes/mBytes, b22.offsetBytes/mBytes, Y.offsetBytes/mBytes, Y.lineStrideBytes/mBytes, b22.lineStrideBytes/mBytes, Y.lineStrideBytes/mBytes, hSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }
  
        auto code = _generateMatMul(eSub, lSub, hSub, X, Y, c12, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        {
            Unit unit;
            _generateBinary(mStack[X.stackIndex], mStack[a12.stackIndex], mStack[X.stackIndex], X.offsetBytes/mBytes, a12.offsetBytes/mBytes, X.offsetBytes/mBytes, X.lineStrideBytes/mBytes, a12.lineStrideBytes/mBytes, X.lineStrideBytes/mBytes, eSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }

        auto code = _generateMatMul(eSub, lSub, hSub, X, b22, c11, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
        code = _generateMatMul(eSub, lSub, hSub, a11, b11, CX, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
    }
    {
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
        {
            Unit unit;
            _generateCFunction(mStack[CT.stackIndex], CT.offsetBytes/mBytes, CT.lineStrideBytes/mBytes, mStack[CX.stackIndex], hSub, eSub, unit);
            mUnits.emplace_back(unit);
        }

        {
            Unit unit;
            _generateBinary(mStack[Y.stackIndex], mStack[Y.stackIndex], mStack[b21.stackIndex], Y.offsetBytes/mBytes, Y.offsetBytes/mBytes, b21.offsetBytes/mBytes, Y.lineStrideBytes/mBytes, Y.lineStrideBytes/mBytes, b21.lineStrideBytes/mBytes, hSub, lSub, false, unit);
            mUnits.emplace_back(unit);
        }
    }
    {
        auto code = _generateMatMul(eSub, lSub, hSub, a22, Y, c11, Empty, currentDepth, 0);
        if (code != NO_ERROR) {
            return code;
        }
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        {
            Unit unit;
            _generateBinary(mStack[c21.stackIndex], mStack[c21.stackIndex], mStack[c11.stackIndex], c21.offsetBytes/mBytes, c21.offsetBytes/mBytes, c11.offsetBytes/mBytes, c21.lineStrideBytes/mBytes, c21.lineStrideBytes/mBytes, c11.lineStrideBytes/mBytes, hSub, eSub, false, unit);
            mUnits.emplace_back(unit);
        }
        
        {
            auto code = _generateMatMul(eSub, lSub, hSub, a12, b21, c11, Empty, currentDepth, 0);
            if (code != NO_ERROR) {
                return code;
            }
            Unit unit;
            _generateBinary(mStack[c11.stackIndex], mStack[c11.stackIndex], mStack[CX.stackIndex], c11.offsetBytes/mBytes, c11.offsetBytes/mBytes, CX.offsetBytes/mBytes, c11.lineStrideBytes/mBytes, c11.lineStrideBytes/mBytes, CX.lineStrideBytes/mBytes, hSub, eSub, true, unit);
            mUnits.emplace_back(unit);
        }

    }
    return NO_ERROR;
}

void StrassenMatrixComputor::onReset() {
    mStack.clear();
    mUnits.clear();
}

ErrorCode StrassenMatrixComputor::onEncode(int e, int l, int h, int as, int bs, int cs, const cl::Buffer AT, const cl::Buffer BT, cl::Buffer CT, bool useBias, const cl::Buffer Bias) {
    mM = e;
    mN = h;
    mK = l;
    MatrixInfo a,b,c,bias;
    bias.stackIndex = -1;
    mUnits.clear();
    mStack = {AT, BT, CT};
    if (useBias) {
        bias.stackIndex = 3;
        bias.offsetBytes = 0;
        mStack.emplace_back(Bias);
    }
    a.stackIndex = 0;
    a.lineStrideBytes = as * mBytes;
    a.offsetBytes = 0;
    
    b.stackIndex = 1;
    b.lineStrideBytes = bs * mBytes;
    b.offsetBytes = 0;
    
    c.stackIndex = 2;
    c.lineStrideBytes = cs * mBytes;
    c.offsetBytes = 0;
    return _generateMatMul(e, l, h, a, b, c, bias, 0, useBias);
}

void StrassenMatrixComputor::onExecute() {
    // All is done in onResize, just execute it
    auto res = CL_SUCCESS;
    int count = 0;
    for (auto &unit : mUnits) {
        if(unit.localWorkSize[0] == 0 || unit.localWorkSize[1] == 0) {
            unit.localWorkSize = cl::NullRange;
        }
#ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                cl::NullRange,
                                                unit.globalWorkSize,
                                                unit.localWorkSize,
                                                nullptr,
                                                &event);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Strassen-" + std::to_string(count++) + "-m" + std::to_string(mM) + "-n" + std::to_string(mN) + "-k" + std::to_string(mK), event});
#else
        res = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                                                      cl::NullRange,
                                                                                      unit.globalWorkSize,
                                                                                      unit.localWorkSize);
#endif
        MNN_CHECK_CL_SUCCESS(res, "Strassen execute");
    }
}
} // namespace MNN
}
#endif
