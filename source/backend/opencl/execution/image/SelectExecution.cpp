//
//  SelectExecution.cpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/SelectExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

SelectExecution::SelectExecution(Backend* backend) : Execution(backend) {
    // Do nothing
}
ErrorCode SelectExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inSize1 = inputs[1]->elementSize();
    auto inSize2 = inputs[2]->elementSize();
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    openCLBackend->startRecord(mRecording);
    if(inSize1 == 1)
        mBuildOptions.emplace("-DINSIZE1_EUQAL_1");
    if(inSize2 == 1)
        mBuildOptions.emplace("-DINSIZE2_EUQAL_1");
    mKernel = runtime->buildKernel("select", "select_img", mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

    std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);
    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks * outputWidth),
        static_cast<uint32_t>(batch * outputHeight)
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[0]);
    ret |= mKernel.setArg(idx++, mGlobalWorkSize[1]);
    ret |= mKernel.setArg(idx++, openCLImage(inputs[0]));
    ret |= mKernel.setArg(idx++, openCLImage(inputs[1]));
    ret |= mKernel.setArg(idx++, openCLImage(inputs[2]));
    ret |= mKernel.setArg(idx++, openCLImage(outputs[0]));
    MNN_CHECK_CL_SUCCESS(ret, "setArg SelectExecution");

    std::string kernelName = "select_img";
    mLocalSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    openCLBackend->recordKernel2d(mKernel, mGlobalWorkSize, mLocalSize);
    openCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode SelectExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SelectExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Select", event});
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        if(mOpenCLBackend->isDevideOpRecord())
            mOpenCLBackend->addRecord(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End SelectExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    runKernel2D(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end SelectExecution onExecute...");
#endif
    return NO_ERROR;
}

class SelectCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new SelectExecution(backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(SelectCreator, OpType_Select, IMAGE);
} // namespace OpenCL
} // namespace MNN
