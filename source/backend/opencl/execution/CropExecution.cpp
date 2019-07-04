//
//  CropExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/CropExecution.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

CropExecution::CropExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CropExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto cropParam = op->main_as_Crop();
    mAxis          = cropParam->axis();
    int offsetSize = cropParam->offset()->size();

    mOffsets.resize(offsetSize);
    for (int i = 0; i < offsetSize; ++i) {
        mOffsets[i] = cropParam->offset()->data()[i];
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName = "crop";
        mKernel                = runtime->buildKernel("crop", kernelName, buildOptions);
        mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end CropExecution init !\n");
#endif
}

ErrorCode CropExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CropExecution onResize !\n");
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end CropExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode CropExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CropExecution onExecute !\n");
#endif
    Tensor *input  = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch   = inputShape.at(0);
    const int inputHeight  = inputShape.at(1);
    const int inputWidth   = inputShape.at(2);
    const int inputChannel = inputShape.at(3);

    const int outputBatch   = outputShape.at(0);
    const int outputHeight  = outputShape.at(1);
    const int outputWidth   = outputShape.at(2);
    const int outputChannel = outputShape.at(3);

    const int inputDim = input->buffer().dimensions;
    std::vector<int> offsets(inputDim, 0);
    for (int i = 0; i < inputDim; ++i) {
        int cropOffset = 0;
        if (i >= mAxis) {
            if (mOffsets.size() == 1) {
                cropOffset = mOffsets[0];
            } else if (mOffsets.size() > 1) {
                cropOffset = mOffsets[i - mAxis];
            }
            MNN_ASSERT(input->buffer().dim[i].extent - cropOffset >= input1->buffer().dim[i].extent);
        }
        offsets[i] = cropOffset;
    }

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputChannel, 4) * outputWidth),
                                        static_cast<uint32_t>(outputBatch * outputHeight)};

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    {
        uint32_t idx = 0;
        mKernel.setArg(idx++, outputGlobalWorkSize[0]);
        mKernel.setArg(idx++, outputGlobalWorkSize[1]);
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(output));
        mKernel.setArg(idx++, inputHeight);
        mKernel.setArg(idx++, inputWidth);
        mKernel.setArg(idx++, offsets[0]); // offset n
        mKernel.setArg(idx++, offsets[2]); // offset h
        mKernel.setArg(idx++, offsets[3]); // offset w
        mKernel.setArg(idx++, offsets[1]); // offset c4
        mKernel.setArg(idx++, outputHeight);
        mKernel.setArg(idx++, outputWidth);
    }

    const std::vector<uint32_t> lws = {16, mMaxWorkGroupSize / 16};

    cl::Event event;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], std::max((uint32_t)1, lws[i]));
    }

    runtime->commandQueue().enqueueNDRangeKernel(mKernel, cl::NullRange,
                                                 cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                 cl::NDRange(lws[0], lws[1]), nullptr, &event);
#ifdef LOG_VERBOSE
    MNN_PRINT("end CropExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<CropExecution>> __crop_op(OpType_Crop);

} // namespace OpenCL
} // namespace MNN
