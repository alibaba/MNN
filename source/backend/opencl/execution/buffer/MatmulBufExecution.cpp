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
                                 bool transposeA, bool transposeB) : Execution(backend)
                                 , mTransposeA(transposeA), mTransposeB(transposeB){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}
ErrorCode MatMulBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);
    
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        if(mTransposeA) {
            mKernelName = mTransposeB ? "matmul_transA_transB_buf":"matmul_transA_buf";
        } else {
            mKernelName = mTransposeB ? "matmul_transB_buf":"matmul_buf";
        }

        if(inputs.size() > 2) {
            buildOptions.emplace("-DBIAS");
        }
        mKernel           = runtime->buildKernel("matmul_buf", mKernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘
    cl_int ret = CL_SUCCESS;
    if(mTransposeA) {
        const int height        = input0Shape.at(3);//input0 H
        const int outputChannel = input0Shape.at(0);//input0 W
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);//input1 WW
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        const int heightblocks        = UP_DIV(height, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(heightblocks)};
        int idx            = 0;
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel.setArg(idx++, openCLBuffer(input0));
        ret |= mKernel.setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= mKernel.setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= mKernel.setArg(idx++, openCLBuffer(output));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannel));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(height));
        ret |= mKernel.setArg(idx++, static_cast<int>(heightblocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(widthblocks));
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, mKernel).first;
    }
    else {
        const int height        = input0Shape.at(0);//input0 H
        const int outputChannel = input0Shape.at(3);//input0 W
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);//input1 W
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(height)};
        int idx            = 0;
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
        ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
        ret |= mKernel.setArg(idx++, openCLBuffer(input0));
        ret |= mKernel.setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= mKernel.setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= mKernel.setArg(idx++, openCLBuffer(output));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannel));
        ret |= mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= mKernel.setArg(idx++, static_cast<int>(widthblocks));
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, mKernel).first;
    }
    MNN_CHECK_CL_SUCCESS(ret, "matmul_buf");
    return NO_ERROR;
}

ErrorCode MatMulBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

#ifdef LOG_VERBOSE
    MNN_PRINT("Start MatMulBufExecution onExecute... \n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);
        
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us MatmulBuf\n",costTime);
    #else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, nullptr);
    #endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End MatMulBufExecution onExecute... \n");
#endif
    return NO_ERROR;
}

class MatMulBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulBufExecution(inputs, op, backend, param->transposeA(), param->transposeB());
    }
};

OpenCLCreatorRegister<MatMulBufCreator> __matmulBuf_op(OpType_MatMul, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
