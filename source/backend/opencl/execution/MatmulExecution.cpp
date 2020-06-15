//
//  MatMulExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "backend/opencl/execution/MatmulExecution.hpp"

namespace MNN {
namespace OpenCL {

MatMulExecution::MatMulExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                 bool transposeA, bool transposeB) : Execution(backend)
                                 , mTransposeA(transposeA), mTransposeB(transposeB){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mAreadySetArg  = false;
}
ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    if (mKernel.get() == nullptr) {

        std::set<std::string> buildOptions;
        MNN_ASSERT(!mTransposeA)
        std::string kernelName = mTransposeB ? "matmul_transB":"matmul";

        mKernel           = runtime->buildKernel("matmul", kernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);
    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘
    const int height        = input0Shape.at(0);
    const int outputChannel = input0Shape.at(3);
    const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);
    const int outputChannelBlocks = UP_DIV(outputChannel, 4);
    const int widthblocks         = UP_DIV(width, 4);
    mGlobalWorkSize[0] = static_cast<uint32_t>(widthblocks);
    mGlobalWorkSize[1] = static_cast<uint32_t>(height);
    int idx            = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, openCLImage(input0));
    mKernel.setArg(idx++, openCLImage(input1));
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, static_cast<int>(outputChannel));
    mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
    mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};

    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

#ifdef LOG_VERBOSE
    MNN_PRINT("Start MatMulExecution onExecute... \n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    run2DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime);

#ifdef LOG_VERBOSE
    MNN_PRINT("End MatMulExecution onExecute... \n");
#endif
    return NO_ERROR;
}

class MatMulCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulExecution(inputs, op, backend, param->transposeA(), param->transposeB());
    }
};

OpenCLCreatorRegister<MatMulCreator> __matmul_op(OpType_MatMul);

} // namespace OpenCL
} // namespace MNN
