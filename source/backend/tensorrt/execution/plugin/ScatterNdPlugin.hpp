//
//  ScatterNdPlugin.hpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScatterNdPlugin_hpp
#define ScatterNdPlugin_hpp
#include <MNN/MNNDefine.h>
#include "CommonPlugin.hpp"
namespace MNN {
class ScatterNdPlugin : public CommonPlugin::Enqueue {
public:
    ScatterNdPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~ScatterNdPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t ScatterNdExecute(nvinfer1::DataType dataType, const int count, const int outElementSize, const int indicesLastDim, const int accNumber,
                          const float* indice, const void* update, void* top_data, const int32_t* dimsToCount,
                          cudaStream_t stream);

private:
    int mIndicesLastDim;
    int mIndexes;
    int mAccNumber;
    int mOutElementSize;
    void* mDeviceScatterNd = nullptr;
};
} // namespace MNN

#endif