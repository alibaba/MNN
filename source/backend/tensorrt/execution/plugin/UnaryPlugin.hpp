//
//  UnaryPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UnaryPlugin_hpp
#define UnaryPlugin_hpp
#include "CommonPlugin.hpp"

using namespace std;
namespace MNN {

class UnaryPlugin : public CommonPlugin::Enqueue {
public:
    UnaryPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~UnaryPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t UnaryExecute(nvinfer1::DataType dataType, const int count, const float* bottom_data, float* top_data, cudaStream_t stream);

private:
    int mCount;
    int mType;
};

} // namespace MNN
#endif /* UnaryPlugin_hpp */
