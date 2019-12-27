//
//  ConvertExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ConvertExecution.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
    namespace OpenCL {

        ConvertExecution::ConvertExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
        : Execution(backend) {
            mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
            std::string kernelName;
            std::set<std::string> buildOptions;

            kernelName = "convert";
            mKernel    = mOpenCLBackend->getOpenCLRuntime()->buildKernel(kernelName, kernelName, buildOptions);
            mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
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

            const std::vector<uint32_t> gws = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
                static_cast<uint32_t>(height * batch)};

            int idx = 0;
            mKernel.setArg(idx++, gws[0]);
            mKernel.setArg(idx++, gws[1]);
            mKernel.setArg(idx++, gws[2]);

            mKernel.setArg(idx++, openCLImage(input));
            mKernel.setArg(idx++, openCLImage(output));

            auto runtime                    = mOpenCLBackend->getOpenCLRuntime();
            mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(width),
                static_cast<uint32_t>(height * batch)};

            mLocalWorkSize = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());
            return NO_ERROR;
        }

        ErrorCode ConvertExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
            MNN_PRINT("Start ConvertExecution onExecute... \n");
#endif

            run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
            MNN_PRINT("End ConvertExecution onExecute... \n");
#endif
            return NO_ERROR;
        }

        OpenCLCreatorRegister<TypedCreator<ConvertExecution>> __ConvertExecution(OpType_ConvertTensor);
        OpenCLCreatorRegister<TypedCreator<ConvertExecution>> __SqueezeExecution(OpType_Squeeze);

    } // namespace OpenCL
} // namespace MNN
