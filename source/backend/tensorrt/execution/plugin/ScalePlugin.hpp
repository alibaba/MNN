//
//  ScalePlugin.hpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScalePlugin_hpp
#define ScalePlugin_hpp
#include <MNN/MNNDefine.h>
#include "CommonPlugin.hpp"
namespace MNN {
class ScalePlugin : public CommonPlugin::Enqueue {
public:
    ScalePlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~ScalePlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t ScaleExecute(nvinfer1::DataType dataType, const int count, const int channels, const int dim, const float* bottom_data,
                             float* top_data, const float* scale, const float* bias, cudaStream_t stream);

private:
    int mArea;
    int mChannel;
    int mBatch;
    int mInputCount;
    void* mDeviceScale = nullptr;
    void* mDeviceBias  = nullptr;
};
} // namespace MNN

#endif