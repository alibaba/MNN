//
//  OneHotPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OneHotPlugin.hpp"
namespace MNN {

OneHotPlugin::OneHotPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {

    auto Info = plugin->main_as_OneHotInfo();
    mOuterSize   = Info->outerSize();
    mInnerSize  = Info->innerSize();
}

OneHotPlugin::~OneHotPlugin() {
}

int OneHotPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    float* output          = reinterpret_cast<float*>(outputs[0]);
    const float* indices        = reinterpret_cast<const float*>(inputs[0]);
    auto depthTensor    = reinterpret_cast<const float*>(inputs[1]);
    auto onValueTensor    = reinterpret_cast<const float*>(inputs[2]);
    auto offValueTensor    = reinterpret_cast<const float*>(inputs[3]);
    return OneHotExecute(dataType, mOuterSize, depthTensor, mInnerSize, indices, onValueTensor, offValueTensor, output, stream);
}

} // namespace MNN