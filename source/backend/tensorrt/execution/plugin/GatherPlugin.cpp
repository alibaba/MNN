//
//  GatherPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GatherPlugin.hpp"

namespace MNN {

GatherPlugin::GatherPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    auto GatherInfo = plugin->main_as_GatherInfo();
    mOutside = GatherInfo->outside();
    mInpNum  = GatherInfo->inpNum();
    mInside  = GatherInfo->inside();
    mOutNum  = GatherInfo->outNum();
    mCount = mOutside * mOutNum * mInside;
}

GatherPlugin::~GatherPlugin() {
    // Do nothgin
}

int GatherPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void *, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
    const float *indices = reinterpret_cast<const float *>(inputs[1]);
    float *top_data          = reinterpret_cast<float *>(outputs[0]);
    return GatherExecute(dataType, mCount, bottom_data, indices, top_data, stream);
}

}; // namespace MNN
