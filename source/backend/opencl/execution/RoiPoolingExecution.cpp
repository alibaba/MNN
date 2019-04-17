//
//  RoiPoolingExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RoiPoolingExecution.hpp"
#include <Macro.h>
#include <float.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

RoiPooling::RoiPooling(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RoiPooling init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto roi       = op->main_as_RoiPooling();
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
    mAreadySetArg = false;
    return NO_ERROR;
}

std::vector<uint32_t> RoiPooling::roiPoolingLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    if (gpuType == GpuType::ADRENO) {
        int coreNum   = deviceComputeUnits;
        int remain    = gws[0] % coreNum;
        int groupSize = gws[0] / coreNum;
        if (remain == 0) {
            lws[0] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[0] % groupSize;
                if (remain == 0 && groupSize <= maxWorkGroupSize) {
                    lws[0] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

        remain    = gws[1] % coreNum;
        groupSize = gws[1] / coreNum;
        if (remain == 0) {
            lws[1] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[1] % groupSize;
                if (remain == 0) {
                    lws[1] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[1] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / lws[0], lws[1]), 1);

        remain    = gws[2] % coreNum;
        groupSize = gws[2] / coreNum;
        if (remain == 0) {
            lws[2] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[2] % groupSize;
                if (remain == 0) {
                    lws[2] = groupSize;
                    break;
                }
                groupSize--;
            }
        }

        lws[2] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / (lws[0] * lws[1]), lws[2]), 1);
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    return lws;
}

ErrorCode RoiPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RoiPooling onExecute !\n");
#endif
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

    std::vector<uint32_t> gws;

    int channelBlocks = (channels + 3) / 4;

    gws = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    if (!mAreadySetArg) {
        uint32_t idx = 0;

        mKernel.setArg(idx++, gws[0]);
        mKernel.setArg(idx++, gws[1]);
        mKernel.setArg(idx++, gws[2]);

        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(roi));
        mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
        mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
        mKernel.setArg(idx++, static_cast<int32_t>(channels));
        mKernel.setArg(idx++, static_cast<int32_t>(roiShape.at(1)));
        mKernel.setArg(idx++, static_cast<float>(mSpatialScale));
        mKernel.setArg(idx++, openCLImage(output));
        mAreadySetArg = true;
    }

    const std::vector<uint32_t> lws = roiPoolingLocalWS(gws, mMaxWorkGroupSize);

    run3DKernelDefault(mKernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());
#ifdef LOG_VERBOSE
    MNN_PRINT("end RoiPooling onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<RoiPooling>> __roi_pooling_op(OpType_ROIPooling);

} // namespace OpenCL
} // namespace MNN
