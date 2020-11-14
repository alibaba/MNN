//
//  BinaryPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryPlugin.hpp"

namespace MNN {

BinaryPlugin::BinaryPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    mType      = op->main_as_BinaryOp()->opType();
    auto shape = plugin->outputs()->GetAs<MNNTRTPlugin::Shape>(0);
    int count  = 1;
    for (int i = 0; i < shape->dim()->size(); ++i) {
        count *= shape->dim()->data()[i];
    }
    mCount = count;
    mS0    = plugin->main_as_BroadCastInfo()->input0() ? 0 : 1;
    mS1    = plugin->main_as_BroadCastInfo()->input1() ? 0 : 1;
}

BinaryPlugin::~BinaryPlugin() {
    // Do nothgin
}

int BinaryPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void *, nvinfer1::DataType dataType, cudaStream_t stream) {
    return BinaryExecute(dataType, mCount, inputs, outputs, mS0, mS1, stream);
}

}; // namespace MNN
