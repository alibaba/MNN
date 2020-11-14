//
//  ScalePlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ScalePlugin.hpp"
namespace MNN {
ScalePlugin::ScalePlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
    auto shape = plugin->outputs()->GetAs<MNNTRTPlugin::Shape>(0);
    mBatch     = shape->dim()->data()[0];
    mChannel   = shape->dim()->data()[1];
    mArea      = 1;
    for (int i = 2; i < shape->dim()->size(); ++i) {
        mArea *= shape->dim()->data()[i];
    }
    auto scale = op->main_as_Scale();
    cudaMalloc(&mDeviceScale, mChannel * sizeof(float));
    MNN_ASSERT(nullptr != mDeviceScale);
    cudaMalloc(&mDeviceBias, mChannel * sizeof(float));
    MNN_ASSERT(nullptr != mDeviceBias);
    mInputCount = mBatch * mChannel * mArea;
    {
        auto alphaData = scale->scaleData()->data();
        cudaMemcpy(mDeviceScale, alphaData, mChannel * sizeof(float), cudaMemcpyHostToDevice);
    }
    {
        auto biasData = scale->biasData()->data();
        if (nullptr != biasData) {
            cudaMemcpy(mDeviceBias, biasData, mChannel * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(mDeviceBias, 0, mChannel * sizeof(float));
        }
    }
}
ScalePlugin::~ScalePlugin() {
    cudaFree(mDeviceBias);
    cudaFree(mDeviceScale);
}
int ScalePlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float* top_data          = reinterpret_cast<float*>(outputs[0]);

    const int count    = batchSize * mInputCount;
    const int dim      = mArea;
    const int channels = mChannel;

    return ScaleExecute(dataType, count, channels, dim, bottom_data, top_data, (const float*)mDeviceScale, (const float*)mDeviceBias,
                 stream);
}

} // namespace MNN