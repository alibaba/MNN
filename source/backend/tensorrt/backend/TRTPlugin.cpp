//
//  TRTPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTPlugin.hpp"

namespace MNN {

nvinfer1::IPlugin* TRTPlugin::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
#ifdef TRT_LOG
    printf("CUDA Plugin type : %s %d\n", layerName, serialLength);
#endif
    auto pluginLayer       = (nvinfer1::IPluginExt*)MNNTRTCreatePlugionSerial(layerName, serialData, serialLength);
    mPluginsMap[layerName] = pluginLayer;
    return pluginLayer;
}

void TRTPlugin::destroyPlugin() {
    for (auto it = mPluginsMap.begin(); it != mPluginsMap.end(); it++) {
        delete it->second;
        mPluginsMap.erase(it);
    }
}

}; // namespace MNN
