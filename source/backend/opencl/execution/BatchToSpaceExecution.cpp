//
//  BatchToSpaceExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/BatchToSpaceExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

BatchToSpaceExecution::BatchToSpaceExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start BatchToSpaceExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param     = op->main_as_SpaceBatch();
    mPaddings[1]   = param->padding()->int32s()->data()[0];
    mPaddings[0]   = param->padding()->int32s()->data()[2];
    mBlockShape[1] = param->blockShape()->int32s()->data()[0];
    mBlockShape[0] = param->blockShape()->int32s()->data()[1];
    std::set<std::string> buildOptions;
    std::string kernelName = "batch_to_space";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();
    mKernel                = runtime->buildKernel("batch_to_space", kernelName, buildOptions);

#ifdef LOG_VERBOSE
    MNN_PRINT("end BatchToSpaceExecution init !\n");
#endif
}

ErrorCode BatchToSpaceExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start BatchToSpaceExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    int inputSize[4]  = {input->width(), input->height(), UP_DIV(input->channel(), 4), input->batch()};
    int outputSize[4] = {output->width(), output->height(), UP_DIV(output->channel(), 4), output->batch()};

    uint32_t idx = 0;
    mKernel.setArg(idx++, inputSize[2]);
    mKernel.setArg(idx++, inputSize[0]);
    mKernel.setArg(idx++, inputSize[1]*inputSize[3]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, sizeof(inputSize), inputSize);
    mKernel.setArg(idx++, sizeof(outputSize), outputSize);
    mKernel.setArg(idx++, sizeof(mPaddings), mPaddings);
    mKernel.setArg(idx++, sizeof(mBlockShape), mBlockShape);
#ifdef LOG_VERBOSE
    MNN_PRINT("end BatchToSpaceExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode BatchToSpaceExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start BatchToSpaceExecution onExecute !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];

    int inputSize[4]  = {input->width(), input->height(), UP_DIV(input->channel(), 4), input->batch()};
    int outputSize[4] = {output->width(), output->height(), UP_DIV(output->channel(), 4), output->batch()};

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    runtime->commandQueue().enqueueNDRangeKernel(
        mKernel, cl::NullRange,
        cl::NDRange(UP_DIV(inputSize[2], 16) * 16, UP_DIV(inputSize[0], 16) * 16, inputSize[1] * inputSize[3]),
        cl::NDRange(16, 16, 1));

#ifdef LOG_VERBOSE
    MNN_PRINT("end BatchToSpaceExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<BatchToSpaceExecution>> __batch_to_space_op(OpType_BatchToSpaceND);

} // namespace OpenCL
} // namespace MNN
