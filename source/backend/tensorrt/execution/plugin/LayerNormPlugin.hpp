//
//  LayerNormPlugin.hpp
//  MNN
//
//  Created by MNN on b'2021/02/08'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LayerNormPlugin_hpp
#define LayerNormPlugin_hpp
#include <MNN/MNNDefine.h>
#include "CommonPlugin.hpp"
namespace MNN {
class LayerNormPlugin : public CommonPlugin::Enqueue {
public:
    LayerNormPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~LayerNormPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t LayerNormExecute(nvinfer1::DataType dataType, const int outter_size_, const int inner_size_, const float* bottom_data,
                                      float* top_data, const float* gamma, const float* beta, cudaStream_t stream);

private:
    int mInnerSize = 1;
    int mOutterSize = 1;
    float mEpsilon = 0.001;
    void* mGamma = nullptr;
    void* mBeta  = nullptr;
    std::vector<int> mAxis;
};
} // namespace MNN

#endif