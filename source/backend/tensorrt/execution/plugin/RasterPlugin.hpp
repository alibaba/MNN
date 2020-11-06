//
//  RasterPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RasterPlugin_hpp
#define RasterPlugin_hpp
#include "CommonPlugin.hpp"
using namespace std;
namespace MNN {
class RasterPlugin : public CommonPlugin::Enqueue {
public:
    RasterPlugin(const MNNTRTPlugin::Plugin* plugin) {
        mPlugin = plugin;
    }
    virtual ~RasterPlugin() = default;

    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType, 
                          cudaStream_t stream) override;

private:
    const MNNTRTPlugin::Plugin* mPlugin;
};

} // namespace MNN
#endif // RasterPlugin_hpp
