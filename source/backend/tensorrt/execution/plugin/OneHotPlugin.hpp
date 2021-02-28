//
//  OneHotPlugin.hpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OneHotPlugin_hpp
#define OneHotPlugin_hpp
#include <MNN/MNNDefine.h>
#include "CommonPlugin.hpp"
namespace MNN {
class OneHotPlugin : public CommonPlugin::Enqueue {
public:
    OneHotPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~OneHotPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t OneHotExecute(nvinfer1::DataType dataType, const int count, const float* depth, int innerSize, const float* indices, const float* onValueTensor,
                                    const float* offValueTensor, float* outputTensor, cudaStream_t stream);

private:
    int mDepth;
    int mInnerSize;
    int mOuterSize;
};
} // namespace MNN

#endif