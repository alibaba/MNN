//
//  MetalEltwise.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalEltwise_hpp
#define MetalEltwise_hpp
#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalEltwise : public MetalExecution {
public:
    MetalEltwise(Backend *backend, EltwiseType type);
    virtual ~MetalEltwise() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void encode(const Tensor *input0, const Tensor *input1, const Tensor *output, id<MTLComputeCommandEncoder> encoder);
    id<MTLComputePipelineState> mPipeline;
    id<MTLBuffer> mConst;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalEltwise_hpp */
