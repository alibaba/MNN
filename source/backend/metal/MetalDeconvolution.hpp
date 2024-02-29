//
//  MetalDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalDeconvolution_hpp
#define MetalDeconvolution_hpp

#import "MetalExecution.hpp"
#include "MNN_generated.h"
#if MNN_METAL_ENABLED
namespace MNN {

class MetalDeconvolution : public MetalExecution {
public:
    MetalDeconvolution(Backend *backend, const MNN::Op *op);
    virtual ~MetalDeconvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    bool mDepthwise  = false;
    int mGroup       = 0;
    int mKernelX     = 0;
    int mKernelY     = 0;
    PadMode mPadMode = PadMode_CAFFE;
    int mPadX        = 0;
    int mPadY        = 0;
    int mStrideX     = 0;
    int mStrideY     = 0;
    int mDilateX     = 0;
    int mDilateY     = 0;
    int mActivationType = 0;

    const MNN::Op *mOp = nullptr;

    std::shared_ptr<MNN::Tensor> mWeight;
    std::shared_ptr<MNN::Tensor> mBias;
    id<MTLBuffer> mConstBuffer = nil;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;

};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalDeconvolution_hpp */
