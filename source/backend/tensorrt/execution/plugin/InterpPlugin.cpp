//
//  InterpPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "InterpPlugin.hpp"
namespace MNN {

InterpPlugin::InterpPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
    auto interp = op->main_as_Interp();
    mResizeType = interp->resizeType();
    MNN_ASSERT(mResizeType == 1 || mResizeType == 2);

    auto InterpInfo = plugin->main_as_InterpInfo();
    mInputChannel   = InterpInfo->inputChannel();
    mChannelBlocks  = InterpInfo->channelBlocks();
    mOutputWidth    = InterpInfo->outputWidth();
    mInputHeight    = InterpInfo->inputHeight();
    mInputWidth     = InterpInfo->inputWidth();
    mOutputHeight   = InterpInfo->outputHeight();

    mWidthOffset  = interp->widthOffset();
    mHeightOffset = interp->heightOffset();
    mWidthScale   = interp->widthScale();
    mHeightScale  = interp->heightScale();

    // printf("%d-%d-%d-%f-%f-%d-%d\n", mChannelBlocks, mOutputHeight, mOutputWidth, mHeightScale, mWidthScale,
    // mInputWidth, mInputHeight);
}
InterpPlugin::~InterpPlugin() {
}

int InterpPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float* top_data          = reinterpret_cast<float*>(outputs[0]);
    int count                = mChannelBlocks * mOutputWidth * mOutputHeight;
    return InterpExecute(dataType, count, mHeightScale, mWidthScale, mHeightOffset, mWidthOffset, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, bottom_data,
                  top_data, stream);
}

} // namespace MNN