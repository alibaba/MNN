//
//  MetalReLU.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReLU_hpp
#define MetalReLU_hpp

#import "MetalExecution.hpp"
#if MNN_METAL_ENABLED
namespace MNN {

class MetalReLU : public MetalExecution {
public:
    MetalReLU(Backend *backend, float slope);
    virtual ~MetalReLU() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    id<MTLBuffer> mSlope;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReLU_hpp */
