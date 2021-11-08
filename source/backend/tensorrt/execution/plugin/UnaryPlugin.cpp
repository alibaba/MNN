//
//  UnaryPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "UnaryPlugin.hpp"

namespace MNN {

UnaryPlugin::UnaryPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    mType      = op->main_as_UnaryOp()->opType();
    auto shape = plugin->outputs()->GetAs<MNNTRTPlugin::Shape>(0);
    int count  = 1;
    for (int i = 0; i < shape->dim()->size(); ++i) {
        count *= shape->dim()->data()[i];
    }
    mCount = count;
}

UnaryPlugin::~UnaryPlugin() {
    // Do nothgin
}

int UnaryPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void *, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
    float *top_data           = reinterpret_cast<float *>(outputs[0]);
    return UnaryExecute(dataType, mCount, bottom_data, top_data, stream);
}

}; // namespace MNN
