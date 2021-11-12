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

    mKernel.setArg(idx++, mGWS[0]);
    mKernel.setArg(idx++, mGWS[1]);
    mKernel.setArg(idx++, mGWS[2]);

    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(roi));
    mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    mKernel.setArg(idx++, static_cast<int32_t>(channels));
    mKernel.setArg(idx++, static_cast<int32_t>(roiShape.at(1)));
    mKernel.setArg(idx++, static_cast<float>(mSpatialScale));
    mKernel.setArg(idx++, openCLImage(output));
    
    mLWS = roiPoolingLocalWS(mGWS, mMaxWorkGroupSize);

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
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us RoiPooling\n",costTime);
#else
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
