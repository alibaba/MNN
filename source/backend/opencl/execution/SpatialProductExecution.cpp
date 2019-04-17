//
//  SpatialProductExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SpatialProductExecution.hpp"
#include <Macro.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

SpatialProductExecution::SpatialProductExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                                 Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SpatialProductExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mAreadySetArg  = false;
#ifdef LOG_VERBOSE
    MNN_PRINT("end SpatialProductExecution init !\n");
#endif
}

ErrorCode SpatialProductExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName = "spatial_product";
        mKernel                = runtime->buildKernel("spatial_product", kernelName, buildOptions);
        mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return NO_ERROR;
}

ErrorCode SpatialProductExecution::onExecute(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SpatialProductExecution onExecute !\n");
#endif
    Tensor *input  = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);

    if (!mAreadySetArg) {
        int batch        = outputShape.at(0);
        int outputHeight = outputShape.at(1);
        int outputWidth  = outputShape.at(2);
        int channels     = outputShape.at(3);

        int channelBlocks = (channels + 3) / 4;

        mGlobalWorkSize = {
            static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(batch * outputHeight),
        };

        uint32_t idx = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, mGlobalWorkSize[2]);
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(input1));
        mKernel.setArg(idx++, static_cast<int>(outputHeight));
        mKernel.setArg(idx++, openCLImage(output));

        mAreadySetArg = true;
    }

    const std::vector<uint32_t> lws =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());
    run3DKernelDefault(mKernel, mGlobalWorkSize, lws, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end SpatialProductExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<SpatialProductExecution>> __spatial_product_op(OpType_SpatialProduct);

} // namespace OpenCL
} // namespace MNN
