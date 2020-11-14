//
//  InterpPlugin.hpp
//  MNN
//
//  Created by MNN on b'2020/08/14'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpPlugin_hpp
#define InterpPlugin_hpp
#include <MNN/MNNDefine.h>
#include "CommonPlugin.hpp"
namespace MNN {
class InterpPlugin : public CommonPlugin::Enqueue {
public:
    InterpPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~InterpPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t InterpExecute(nvinfer1::DataType dataType, const int count, const float mHeightScale, const float mWidthScale, const int mInputHeight,
                       const int mInputWidth, const int mOutputHeight, const int mOutputWidth, const float* bottom_data,
                       float* top_data, cudaStream_t stream);

private:
    int mInputChannel;
    int mChannelBlocks;
    int mOutputWidth;
    int mOutputH_N;
    float mHeightScale;
    float mWidthScale;
    int mInputHeight;
    int mInputWidth;
    int mOutputHeight;
    int mResizeType;
};
} // namespace MNN

#endif