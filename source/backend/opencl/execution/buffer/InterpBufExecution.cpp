//
//  InterpBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/InterpBufExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

InterpBufExecution::InterpBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interpParam = op->main_as_Interp();
    mCordTransform[0] = interpParam->widthScale();
    mCordTransform[1] = interpParam->widthOffset();
    mCordTransform[2] = interpParam->heightScale();
    mCordTransform[3] = interpParam->heightOffset();

    std::set<std::string> buildOptions;
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernelName = "nearest_buf";
        mKernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    } else if(op->main_as_Interp()->resizeType() == 4) {
        mKernelName = "nearest_buf";
        buildOptions.emplace("-DUSE_ROUND");
        mKernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }else {
        mKernelName = "bilinear_buf";
        mKernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode InterpBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = input->batch();
    const int inputHeight   = input->height();
    const int inputWidth    = input->width();
    const int inputChannels = input->channel();

    const int channelBlocks = UP_DIV(inputChannels, 4);

    const int outputHeight = output->height();
    const int outputWidth  = output->width();

    mGWS = {static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * inputBatch)};

    MNN_ASSERT(outputHeight > 0 && outputWidth > 0);

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGWS[0]);
    ret |= mKernel.setArg(idx++, mGWS[1]);
    ret |= mKernel.setArg(idx++, mGWS[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, mCordTransform[2]);
    ret |= mKernel.setArg(idx++, mCordTransform[0]);
    ret |= mKernel.setArg(idx++, mCordTransform[3]);
    ret |= mKernel.setArg(idx++, mCordTransform[1]);
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(channelBlocks));
    MNN_CHECK_CL_SUCCESS(ret, "setArg InterpBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, mKernelName, mKernel).first;
    return NO_ERROR;

}

ErrorCode InterpBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpBufExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Interp", event});
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpBufExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class InterpBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~InterpBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        if(op->main_as_Interp()->resizeType() == 3) {
            MNN_PRINT("openCL buffer not support interp type:%d, fallback to cpu\n", op->main_as_Interp()->resizeType());
            return nullptr;
        }
        return new InterpBufExecution(inputs, op, backend);
    }
};
    
OpenCLCreatorRegister<InterpBufCreator> __InterpBuf_op_(OpType_Interp, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
