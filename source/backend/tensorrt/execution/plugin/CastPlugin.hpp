//
//  CastPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CastPlugin_hpp
#define CastPlugin_hpp
#include "CommonPlugin.hpp"

using namespace std;
namespace MNN {

class CastPlugin : public CommonPlugin::Enqueue {
public:
    CastPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~CastPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
private:
    int mCount;
};

} // namespace MNN
#endif /* CastPlugin_hpp */
