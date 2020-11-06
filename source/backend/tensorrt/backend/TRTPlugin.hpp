//
//  TRTPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TRTPlugin_hpp
#define TRTPlugin_hpp

#include <map>
#include "../execution/plugin/PluginFactory.hpp"
#include "NvInfer.h"

namespace MNN {

class TRTPlugin : public nvinfer1::IPluginFactory {
public:
    // deserialization plugin implementation
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData,
                                            size_t serialLength) override;

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin();

private:
    std::map<std::string, nvinfer1::IPluginExt*> mPluginsMap;
};

} // namespace MNN
#endif /* TRTPlugin_hpp */
