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
    if(mTransposeA) {
        mKernelName = mTransposeB ? "matmul_transA_transB_buf":"matmul_transA_buf";
    } else {
        mKernelName = mTransposeB ? "matmul_transB_buf":"matmul_buf";
    }

    if(inputs.size() > 2) {
        buildOptions.emplace("-DBIAS");
    }
    unit.kernel       = runtime->buildKernel("matmul_buf", mKernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

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
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(height));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(heightblocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(widthblocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulBufExecution mTransposeA");

        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, unit.kernel).first;
    }
    else {
        const int height        = input0Shape.at(0);//input0 H
        const int outputChannel = input0Shape.at(3);//input0 W
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);//input1 W
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(height)};
        int idx            = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(widthblocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(width));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulBufExecution");
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
