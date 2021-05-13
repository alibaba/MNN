//
//  GatherPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GatherPlugin_hpp
#define GatherPlugin_hpp
#include "CommonPlugin.hpp"

using namespace std;
namespace MNN {

class GatherPlugin : public CommonPlugin::Enqueue {
public:
    GatherPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~GatherPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t GatherExecute(nvinfer1::DataType dataType, const int count, const float* bottom_data, const float* indices, float* top_data, cudaStream_t stream);

private:
    int mCount;
    int mLimit;
    int mInsideStride;
    int mN;
    int mOutputOutsideStride;
    int mInputOutsideStride;
    bool mInput3{false};
};

} // namespace MNN
#endif /* GatherPlugin_hpp */
