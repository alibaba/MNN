//
//  ScatterNdPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ScatterNdPlugin.hpp"
namespace MNN {
ScatterNdPlugin::ScatterNdPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    auto scatterNdInfo = plugin->main_as_ScatterNdInfo();
    mIndicesLastDim    = scatterNdInfo->indicesLastDim();
    mIndexes           = scatterNdInfo->indexes();
    mAccNumber         = scatterNdInfo->accNumber();
    mOutElementSize    = scatterNdInfo->outElementSize();

    cudaMalloc(&mDeviceScatterNd, mIndicesLastDim * sizeof(int32_t));
    MNN_ASSERT(nullptr != mDeviceScatterNd);
    {
        auto data = scatterNdInfo->dimsToCount()->data();
        cudaMemcpy(mDeviceScatterNd, data, mIndicesLastDim * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
}
ScatterNdPlugin::~ScatterNdPlugin() {
    cudaFree(mDeviceScatterNd);
}
int ScatterNdPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void *, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float *indice = reinterpret_cast<const float *>(inputs[0]);
    const void *update  = reinterpret_cast<const void *>(inputs[1]);
    void *top_data      = reinterpret_cast<void *>(outputs[0]);

    const int count = mIndexes;

    return ScatterNdExecute(dataType, count, mOutElementSize, mIndicesLastDim, mAccNumber, indice, update, top_data,
                     (const int32_t *)mDeviceScatterNd, stream);
}

} // namespace MNN