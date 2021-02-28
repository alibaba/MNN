//
//  CastPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CastPlugin.hpp"
#include <MNN/MNNDefine.h>
namespace MNN {

CastPlugin::CastPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
    auto Info = plugin->main_as_OneHotInfo();
    mCount   = Info->outerSize();
}

CastPlugin::~CastPlugin() {
}

int CastPlugin::onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, cudaStream_t stream) {
    const int* bottom_data = reinterpret_cast<const int*>(inputs[0]);
    float* top_data          = reinterpret_cast<float*>(outputs[0]);
    return CastInt32ToFloatExecute(dataType, mCount, bottom_data, top_data, stream);

}

}; // namespace MNN
