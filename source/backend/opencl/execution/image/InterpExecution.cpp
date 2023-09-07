//
//  InterpExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/InterpExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

InterpExecution::InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interpParam = op->main_as_Interp();
    mCordTransform[0] = interpParam->widthScale();
    mCordTransform[1] = interpParam->widthOffset();
    mCordTransform[2] = interpParam->heightScale();
    mCordTransform[3] = interpParam->heightOffset();

    std::set<std::string> buildOptions;
    std::string kernelName = "interp";
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    }else if (op->main_as_Interp()->resizeType() == 4) {
        buildOptions.emplace("-DUSE_ROUND");
        mKernel                = runtime->buildKernel("nearest", kernelName, buildOptions);
    }else {
        mKernel                = runtime->buildKernel("interp", kernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode InterpExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    startRecord(runtime, mRecording);

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
    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, openCLImage(output));
    ret |= mKernel.setArg(idx++, mCordTransform[2]);
    ret |= mKernel.setArg(idx++, mCordTransform[0]);
    ret |= mKernel.setArg(idx++, mCordTransform[3]);
    ret |= mKernel.setArg(idx++, mCordTransform[1]);
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    MNN_CHECK_CL_SUCCESS(ret, "setArg InterpExecution");

    std::string name = "interp";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, name, mKernel).first;
    recordKernel3d(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
    endRecord(runtime, mRecording);
    return NO_ERROR;

}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Interp", event});
#else
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End InterpExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class InterpCreator : public OpenCLBackend::Creator {
public:
    virtual ~InterpCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend) const override {
        if(op->main_as_Interp()->resizeType() == 3) {
            MNN_PRINT("openCL not support interp type:%d, fallback to cpu\n", op->main_as_Interp()->resizeType());
            return nullptr;
        }
        return new InterpExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<InterpCreator> __Interp_op_(OpType_Interp, IMAGE);

} // namespace OpenCL
} // namespace MNN
