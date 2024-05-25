//
//  MatmulExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/MatmulExecution.hpp"

namespace MNN {
namespace OpenCL {

MatMulExecution::MatMulExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                 bool transposeA, bool transposeB) : CommonExecution(backend, op)
                                 , mTransposeA(transposeA), mTransposeB(transposeB){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mAreadySetArg  = false;
}
ErrorCode MatMulExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];
    
    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);
    
    std::vector<uint32_t> mGlobalWorkSize{1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    
    
    std::string kernelName;
    std::set<std::string> buildOptions;
    if(mTransposeA) {
        kernelName = mTransposeB ? "matmul_transA_transB":"matmul_transA";
    } else {
        kernelName = mTransposeB ? "matmul_transB":"matmul";
    }
    
    if(inputs.size() > 2) {
        buildOptions.emplace("-DBIAS");
    }
    unit.kernel           = runtime->buildKernel("matmul", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘
    if(mTransposeA) {
        const int height        = input0Shape.at(3);
        const int outputChannel = input0Shape.at(0);
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        const int heightblocks        = UP_DIV(height, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(heightblocks)};
        cl_int ret = CL_SUCCESS;
        int idx            = 0;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(height));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulExecution transposeA");

        mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};
    }
    else {
        const int height        = input0Shape.at(0);
        const int outputChannel = input0Shape.at(3);
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        
        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(height)};
        int idx            = 0;
        cl_int ret = CL_SUCCESS;

        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input0));
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input1));
        if(inputs.size() > 2) {
            ret |= unit.kernel->get().setArg(idx++, openCLImage(inputs[2]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MatMulExecution transposeA");

        mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};
    }

    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
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

REGISTER_OPENCL_OP_CREATOR(MatMulCreator, OpType_MatMul, IMAGE);

} // namespace OpenCL
} // namespace MNN
