//
//  Input.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class ShuffleChannel : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);

    virtual MNN::OpType opType() {
        return MNN::OpType_PLUGIN;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Plugin;
    }
};

void ShuffleChannel::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters,
                         const caffe::LayerParameter& weight) {
    auto plugin  = new MNN::PluginT;
    plugin->type = "ShuffleChannel";
    plugin->buffer.resize(1);
    plugin->buffer[0].reset(new MNN::BlobT);
    auto blob    = plugin->buffer[0].get();
    blob->int32s = {1};
    if (parameters.has_shuffle_channel_param()) {
        blob->int32s = {(int)parameters.shuffle_channel_param().group()};
    }
    dstOp->main.value = plugin;
}

static OpConverterRegister<ShuffleChannel> a("ShuffleChannel");
