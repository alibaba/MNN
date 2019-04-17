//
//  ConvertExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/ConvertExecution.hpp"
#include <Macro.h>
#include "CPUTensorConvert.hpp"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ConvertExecution::ConvertExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(backend) {
}

ErrorCode ConvertExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input  = inputs[0];
    Tensor* output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);
    mWidth                  = width * channelBlocks;
    mHeight                 = height * batch;

    return NO_ERROR;
}

ErrorCode ConvertExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvertExecution onExecute... \n");
#endif
    Tensor* input  = inputs[0];
    Tensor* output = outputs[0];

    auto runtime = ((OpenCLBackend*)backend())->getOpenCLRuntime();
    runtime->commandQueue().enqueueCopyImage(openCLImage(input), openCLImage(output), {0, 0, 0}, {0, 0, 0},
                                             {mWidth, mHeight, 1});
#ifdef LOG_VERBOSE
    MNN_PRINT("End ConvertExecution onExecute... \n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ConvertExecution>> __ConvertExecution(OpType_ConvertTensor);

} // namespace OpenCL
} // namespace MNN
