//
//  MetalDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalDeconvolution_hpp
#define MetalDeconvolution_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalDeconvolution : public Execution {
public:
    MetalDeconvolution(Backend *backend, const MNN::Op *op);
    virtual ~MetalDeconvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

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

    id<MTLBuffer> mWeight      = nil;
    id<MTLBuffer> mBias        = nil;
    id<MTLBuffer> mConstBuffer = nil;

private:
    ErrorCode onDepthwise(const Tensor *input, const Tensor *output);
    ErrorCode onDeconv(const Tensor *input, const Tensor *output);
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalDeconvolution_hpp */
