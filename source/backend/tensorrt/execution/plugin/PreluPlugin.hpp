//
//  PreluPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PreluPlugin_hpp
#define PreluPlugin_hpp
#include "CommonPlugin.hpp"

using namespace std;
namespace MNN {

class PreluPlugin : public CommonPlugin::Enqueue {
public:
    PreluPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    virtual ~PreluPlugin();
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;
    cudaError_t PReLUExecute(nvinfer1::DataType dataType, const int count, const int channels, const int dim, const float* bottom_data,
                             float* top_data, void* mDeviceKernel, const int div_factor, cudaStream_t stream);

private:
    int mInputB;
    int mInputC;
    int mInputPlane;
    int mInputCount;
    bool mIsChannelShared;
    void* mDeviceKernel{nullptr};
};

} // namespace MNN
#endif /* PreluPlugin_hpp */
