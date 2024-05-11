//
//  MetalReLU6.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReLU6_hpp
#define MetalReLU6_hpp

#import "MetalExecution.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalReLU6 : public MetalExecution {
public:
    MetalReLU6(Backend *backend, float minValue, float maxValue, bool isRelu);
    virtual ~MetalReLU6() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
private:
    id<MTLBuffer> mConst;
    id<MTLComputePipelineState> mPipeline;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReLU6_hpp */
