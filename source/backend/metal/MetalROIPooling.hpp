//
//  MetalROIPooling.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalROIPooling_hpp
#define MetalROIPooling_hpp

#import "MetalExecution.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalROIPooling : public MetalExecution {
public:
    MetalROIPooling(Backend *backend, float spatialScale);
    virtual ~MetalROIPooling() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    float mSpatialScale;
    id<MTLBuffer> mShape;
    id<MTLComputePipelineState> mPipeline;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalROIPooling_hpp */
