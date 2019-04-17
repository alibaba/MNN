//
//  InterpExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/InterpExecution.hpp"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

InterpExecution::InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    mAlignCorners  = op->main_as_Interp()->alignCorners();

    std::set<std::string> buildOptions;
    std::string kernelName = "interp";
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    } else {
        mKernel                = runtime->buildKernel("interp", kernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    mAreadySetArg = false;
}

std::vector<uint32_t> InterpExecution::interpLocalWS(const std::vector<uint32_t> &gws,
                                                     const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    if (gpuType == GpuType::ADRENO) {
        lws[0] = deviceComputeUnits * 4;
        lws[1] = 4;
        lws[2] = 1;
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    return lws;
}

static float resizeScale(int inputSize, int outputSize, bool isAlign) {
    int corner = 0;
    if (isAlign) {
        corner = 1;
    }
    return (float)(inputSize - corner) / (float)(outputSize - corner);
}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpExecution onExecute... \n");
#endif
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = input->batch();
    const int inputHeight   = input->height();
    const int inputWidth    = input->width();
    const int inputChannels = input->channel();

    const int channelBlocks = UP_DIV(inputChannels, 4);

    const int outputHeight = output->height();
    const int outputWidth  = output->width();

    const std::vector<uint32_t> gws = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
                                       static_cast<uint32_t>(outputHeight * inputBatch)};

    if (!mAreadySetArg) {
        MNN_ASSERT(outputHeight > 0 && outputWidth > 0);

        float height_scale = resizeScale(inputHeight, outputHeight, mAlignCorners);
        float width_scale  = resizeScale(inputWidth, outputWidth, mAlignCorners);

        uint32_t idx = 0;
        mKernel.setArg(idx++, gws[0]);
        mKernel.setArg(idx++, gws[1]);
        mKernel.setArg(idx++, gws[2]);
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(output));
        mKernel.setArg(idx++, height_scale);
        mKernel.setArg(idx++, width_scale);
        mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
        mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
        mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
        mAreadySetArg = true;
    }

    const std::vector<uint32_t> lws = interpLocalWS(gws, mMaxWorkGroupSize);
    run3DKernelDefault(mKernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpExecution onExecute... \n");
#endif

    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<InterpExecution>> __Interp_op_(OpType_Interp);

} // namespace OpenCL
} // namespace MNN
