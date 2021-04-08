//
//  LayerNormPlugin.cpp
//  MNN
//
//  Created by MNN on b'2021/02/08'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "LayerNormPlugin.hpp"
namespace MNN {
LayerNormPlugin::LayerNormPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {

    const auto* layer_norm_param = op->main_as_LayerNorm();
    int axis_size = layer_norm_param->axis()->size();
    mAxis.resize(axis_size);
    for (int i = 0; i < axis_size; ++i) {
        mAxis[i] = layer_norm_param->axis()->Get(i);
    }
    mEpsilon = layer_norm_param->epsilon();

    int size = layer_norm_param->gamma()->size();
    cudaMalloc(&mGamma, size * sizeof(float));
    MNN_ASSERT(nullptr != mGamma);
    const float* gamma_data = layer_norm_param->gamma()->data();
    auto status = cudaMemcpy(mGamma, gamma_data, size * sizeof(float), cudaMemcpyHostToDevice);
    MNN_ASSERT(0 == status);

    cudaMalloc(&mBeta, size * sizeof(float));
    MNN_ASSERT(nullptr != mBeta);

    const float* beta_data = layer_norm_param->beta()->data();
    status = cudaMemcpy(mBeta, beta_data, size * sizeof(float), cudaMemcpyHostToDevice);
    MNN_ASSERT(0 == status);

    auto Info = plugin->main_as_OneHotInfo();
    mOutterSize   = Info->outerSize();
    mInnerSize  = Info->innerSize();

}
LayerNormPlugin::~LayerNormPlugin() {
    cudaFree(mBeta);
    cudaFree(mGamma);
}
int LayerNormPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float* top_data          = reinterpret_cast<float*>(outputs[0]);

    return LayerNormExecute(dataType, mOutterSize, mInnerSize, bottom_data, top_data, (const float*)mGamma, (const float*)mBeta,
                 stream);
}

} // namespace MNN