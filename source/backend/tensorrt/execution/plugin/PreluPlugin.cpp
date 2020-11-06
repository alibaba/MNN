//
//  PreluPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PreluPlugin.hpp"
#include <MNN/MNNDefine.h>
namespace MNN {

PreluPlugin::PreluPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
    auto shape  = plugin->outputs()->GetAs<MNNTRTPlugin::Shape>(0);
    mInputB     = shape->dim()->data()[0];
    mInputC     = shape->dim()->data()[1];
    mInputPlane = 1;
    for (int i = 2; i < shape->dim()->size(); ++i) {
        mInputPlane *= shape->dim()->data()[i];
        // printf("Prelu: %d\n", mInputPlane);
    }
    mInputCount = mInputB * mInputC * mInputPlane;
    if (op->type() == OpType_ReLU) {
        float slope = op->main_as_Relu()->slope();
        cudaMalloc(&mDeviceKernel, 1 * sizeof(float));
        MNN_ASSERT(nullptr != mDeviceKernel);
        cudaMemcpy(mDeviceKernel, &slope, 1 * sizeof(float), cudaMemcpyHostToDevice);
        mIsChannelShared = true;
    } else {
        auto slopCount = op->main_as_PRelu()->slope()->size();
        auto alphaData = op->main_as_PRelu()->slope()->data();
        cudaMalloc(&mDeviceKernel, slopCount * sizeof(float));
        MNN_ASSERT(nullptr != mDeviceKernel);
        cudaMemcpy(mDeviceKernel, alphaData, slopCount * sizeof(float), cudaMemcpyHostToDevice);
        mIsChannelShared = slopCount == 1;
    }
}

PreluPlugin::~PreluPlugin() {
    cudaFree(mDeviceKernel);
}

int PreluPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float* top_data          = reinterpret_cast<float*>(outputs[0]);

    const int count    = batchSize * mInputCount;
    const int dim      = mInputPlane;
    const int channels = mInputC;
    const int div_factor = mIsChannelShared ? channels : 1; // mIsChannelShared default is false

    return PReLUExecute(dataType, count, channels, dim, bottom_data, top_data, mDeviceKernel, div_factor, stream);

}

}; // namespace MNN
