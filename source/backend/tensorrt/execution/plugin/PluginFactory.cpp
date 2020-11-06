//
//  PluginFactory.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PluginFactory.hpp"
#include "BinaryPlugin.hpp"
#include "CommonPlugin.hpp"
#include "MNN_generated.h"
#include "PreluPlugin.hpp"

using namespace MNN;

void* MNNTRTCreatePlugion(const void* opRaw, const void* extraInfo) {
    auto op     = (MNN::Op*)opRaw;
    auto plugin = (const MNNTRTPlugin::PluginT*)extraInfo;
    return new CommonPlugin(op, plugin);
}

void* MNNTRTCreatePlugionSerial(const char* layerName, const void* serialData, size_t serialLength) {
    return new CommonPlugin(serialData, serialLength);
}
