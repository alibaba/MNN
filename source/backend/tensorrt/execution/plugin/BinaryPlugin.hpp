//
//  BinaryPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BinaryPlugin_hpp
#define BinaryPlugin_hpp
#include "CommonPlugin.hpp"

using namespace std;
namespace MNN {

class BinaryPlugin : public CommonPlugin::Enqueue {
public:
    BinaryPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    ~BinaryPlugin();
    cudaError_t BinaryExecute(nvinfer1::DataType dataType, const int count, const void *const *inputs, void **outputs,
                              int s0, int s1, cudaStream_t stream);
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;

private:
    int mType;
    int mS0;
    int mS1;
    int mCount;
};

} // namespace MNN
#endif /* BinaryPlugin_hpp */
