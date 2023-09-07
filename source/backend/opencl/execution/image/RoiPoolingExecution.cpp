//
//  RoiPoolingExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/RoiPoolingExecution.hpp"
#include "core/Macro.h"
#include <float.h>
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

RoiPooling::RoiPooling(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RoiPooling init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto roi       = op->main_as_RoiParameters();
    mPooledWidth   = roi->pooledWidth();
    mPooledHeight  = roi->pooledHeight();
    mSpatialScale  = roi->spatialScale();
    mAreadySetArg  = false;
    std::set<std::string> buildOptions;
    std::string kernelName = "roi_pooling";
    std::vector<int> roiShape    = tensorShapeFormat(inputs[1]);
    const int roiHeight   = roiShape.at(1);
    const int roiWidth    = roiShape.at(2);
    const int roiChannels = roiShape.at(3);
    if (roiWidth == 5) {
        buildOptions.emplace("-DROI_C1H1W5");
    }else if(roiChannels == 5){
        buildOptions.emplace("-DROI_C5H1W1");
    }
    mKernel                = mOpenCLBackend->getOpenCLRuntime()->buildKernel("roi_pooling", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
#ifdef LOG_VERBOSE
    MNN_PRINT("end RoiPooling init !\n");
#endif
}

ErrorCode RoiPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    Tensor *roi    = inputs[1];

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    startRecord(runtime, mRecording);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    std::vector<int> roiShape    = tensorShapeFormat(roi);

    const int batch        = outputShape.at(0);
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int channels     = outputShape.at(3);

    const int inputBatch    = inputShape.at(0);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    int channelBlocks = (channels + 3) / 4;

    mGWS = {static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(batch * outputHeight),
            };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGWS[0]);
    ret |= mKernel.setArg(idx++, mGWS[1]);
    ret |= mKernel.setArg(idx++, mGWS[2]);

    ret |= mKernel.setArg(idx++, openCLImage(input));
    ret |= mKernel.setArg(idx++, openCLImage(roi));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputBatch));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputWidth));
    ret |= mKernel.setArg(idx++, static_cast<float>(mSpatialScale));
    ret |= mKernel.setArg(idx++, openCLImage(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg RoiPoolExecution");

    mLWS = roiPoolingLocalWS(mGWS, mMaxWorkGroupSize);
    recordKernel3d(mKernel, mGWS, mLWS, runtime);
    endRecord(runtime, mRecording);
    return NO_ERROR;
}

std::vector<uint32_t> RoiPooling::roiPoolingLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    int coreNum = deviceComputeUnits;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
                int remain = gws[i] % groupSize;
                if (remain == 0 && (i > 0 || groupSize <= maxWorkGroupSize)) {
                    lws[i] = groupSize;
                    break;
                }
                --groupSize;
            }
        }
        lws[i] = std::max<uint32_t>(std::min<uint32_t>(lws[i], maxWorkGroupSize / totalSizeNow), 1);
        totalSizeNow *= lws[i];
    }
    return lws;
}

ErrorCode RoiPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RoiPooling onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"RoiPooling", event});
#else
    if(mOpenCLBackend->getOpenCLRuntime()->isUseRecordQueue()){
        if(mOpenCLBackend->getOpenCLRuntime()->isDevideOpRecord())
            mOpenCLBackend->getOpenCLRuntime()->getRecordings()->emplace_back(mRecording);
#ifdef LOG_VERBOSE
        MNN_PRINT("End RoiPooling onExecute... \n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end RoiPooling onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<RoiPooling>> __roi_pooling_op(OpType_ROIPooling, IMAGE);

} // namespace OpenCL
} // namespace MNN
