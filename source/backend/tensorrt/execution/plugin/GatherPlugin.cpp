//
//  GatherPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GatherPlugin.hpp"

namespace MNN {

GatherPlugin::GatherPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    auto GatherInfo = plugin->main_as_GatherInfo();
    mLimit = GatherInfo->limit();
    mInsideStride  = GatherInfo->insideStride();
    mN  = GatherInfo->N();
    mOutputOutsideStride  = GatherInfo->outputOutsideStride();
    mInputOutsideStride  = GatherInfo->inputOutsideStride();
    mCount = GatherInfo->outside()*mN*mInsideStride;
    mInput3 = GatherInfo->input3();
}

GatherPlugin::~GatherPlugin() {
    // Do nothgin
}

int GatherPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void *, nvinfer1::DataType dataType, cudaStream_t stream) {
    const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
    const float *indices = reinterpret_cast<const float *>(inputs[1]);
    float *top_data          = reinterpret_cast<float *>(outputs[0]);
    if(mInput3){
        int axis;
        auto status = cudaMemcpy(&axis, reinterpret_cast<const int *>(inputs[2]), sizeof(int), cudaMemcpyDeviceToHost);
        MNN_ASSERT(0 == status);
        MNN_ASSERT(0 == axis);
    }
    return GatherExecute(dataType, mCount, bottom_data, indices, top_data, stream);
}

}; // namespace MNN
