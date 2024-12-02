//
//  MatmulBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"

namespace MNN {
namespace OpenCL {

MatMulBufExecution::MatMulBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                 bool transposeA, bool transposeB) : CommonExecution(backend, op)
                                 , mTransposeA(transposeA), mTransposeB(transposeB){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}
ErrorCode MatMulBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);

    std::set<std::string> buildOptions;
    int M = input0Shape[0];
    int K = input0Shape[3];
    if(mTransposeA) {
        M = input0Shape[3];
        K = input0Shape[0];
    }
    int N = mTransposeB ? input1Shape[0]: input1Shape[3];
    
    const int K_4 = UP_DIV(K, 4);
    const int N_4 = UP_DIV(N, 4);
    const int M_4 = UP_DIV(M, 4);
    
    // set large tile
    unsigned int tileM = 128;
    unsigned int tileN = 128;
    unsigned int tileK = 32;
    unsigned int localM = 32;
    unsigned int localN = 8;
    
    if(inputs.size() > 2) {
        buildOptions.emplace("-DBIAS");
    }
    
    bool canUseTile = (M % tileM == 0) && \
        (N % tileN == 0) && \
        (K % tileK == 0);
    bool canUseLargeTile = canUseTile && mTransposeA && !mTransposeB;
    if (!canUseLargeTile) {
        // set small tile
        tileM = 64;
        tileN = 128;
        tileK = 8;
        localM = 16;
        localN = 16;
        
        canUseTile = (M % tileM == 0) && (N % tileN == 0) && (K % tileK == 0);
    }
    
    if(canUseLargeTile) {
        // Match with Large tileM->MWG tileN->NWG tileK->KWG localM->MDIMA localN->NDIMC
        uint32_t layout = 4;
        uint32_t batch = 1;
        std::vector<uint32_t> param;
        if(inputs.size() == 2) {
            param = getGemmParams({(uint32_t)M, (uint32_t)N, (uint32_t)K, layout, batch, (uint32_t)0}, {openCLBuffer(input0), openCLBuffer(input1), openCLBuffer(output)}, mOpenCLBackend->getOpenCLRuntime());
        } else {
            param = getGemmParams({(uint32_t)M, (uint32_t)N, (uint32_t)K, layout, batch, (uint32_t)1}, {openCLBuffer(input0), openCLBuffer(input1), openCLBuffer(output), openCLBuffer(inputs[2])}, mOpenCLBackend->getOpenCLRuntime());
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

        if(inputs.size() > 2) {
            buildOptions.emplace(" -DBIAS_TYPE=1");
        }
        if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
            buildOptions.emplace("-DUSE_CL_MAD=1");
            buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
        }

        unit.kernel       = runtime->buildKernel("matmul_params_buf", "Xgemm", buildOptions);

     } else if(canUseTile) {
        if(mTransposeA) {
            buildOptions.emplace(" -DTRANSPOSE_A");
        }
        if(mTransposeB) {
            buildOptions.emplace(" -DTRANSPOSE_B");
        }
        // Match with Small tileM->OPWM tileN->OPWN tileK->CPWK localM->OPWM/OPTM localN->OPWN/OPTN
        buildOptions.emplace(" -DOPWM=64 -DOPWN=128 -DCPWK=8 -DOPTM=4 -DOPTN=8");
        
        unit.kernel       = runtime->buildKernel("matmul_local_buf", "matmul_local_buf", buildOptions);
    } else {
        if(mTransposeA) {
            buildOptions.emplace(" -DTRANSPOSE_A");
        }
        if(mTransposeB) {
            buildOptions.emplace(" -DTRANSPOSE_B");
        }
        if(M % 4 != 0) {
            buildOptions.emplace(" -DM_LEAVE");
            buildOptions.emplace(" -DM_LEAVE_NUM=" + std::to_string(M % 4));
        }
        if(N % 4 != 0) {
            buildOptions.emplace(" -DN_LEAVE");
            buildOptions.emplace(" -DN_LEAVE_NUM=" + std::to_string(N % 4));
        }
        if(K % 4 != 0) {
            buildOptions.emplace(" -DK_LEAVE");
        }
        unit.kernel       = runtime->buildKernel("matmul_buf", "matmul_buf", buildOptions);
    }
    
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    cl_int ret = CL_SUCCESS;
    if(canUseLargeTile) {
        
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSize = {static_cast<uint32_t>(M/out_per_thread_m), static_cast<uint32_t>(N/out_per_thread_n)};
        mLocalWorkSize = {localM, localN};
        
        float alpha = 1.0;
        float beta = 0.0f;
        int offset[4] = {0, 0, 0, 0};
        int stride[4] = {M, N, N, N};
        
        int idx            = 0;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(N));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(K));
        ret |= unit.kernel->get().setArg(idx++, alpha);
        ret |= unit.kernel->get().setArg(idx++, beta);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input1));
        if (inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, offset);
        ret |= unit.kernel->get().setArg(idx++, stride);
        
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulBufExecution use large tile opt");
        
    } else if(canUseTile) {
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSize = {static_cast<uint32_t>(M/out_per_thread_m), static_cast<uint32_t>(N/out_per_thread_n)};
        mLocalWorkSize = {localM, localN};

        int idx            = 0;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(N));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(K));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));

        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulBufExecution use tile opt");

    } else {
        mGlobalWorkSize = {static_cast<uint32_t>(N_4), static_cast<uint32_t>(M_4)};
        int idx            = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(N));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(K));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulBufExecution mTransposeA");
            
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, unit.kernel).first;
    }
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return NO_ERROR;
}

class MatMulBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto param = op->main_as_MatMul();
        return new MatMulBufExecution(inputs, op, backend, param->transposeA(), param->transposeB());
    }
};

REGISTER_OPENCL_OP_CREATOR(MatMulBufCreator, OpType_MatMul, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
