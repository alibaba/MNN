//
//  MetalROIAlign.hpp
//  MNN
//

#ifndef MetalROIAlign_hpp
#define MetalROIAlign_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalROIAlign : public MetalExecution {
public:
    MetalROIAlign(Backend *backend, const MNN::Op *op);
    virtual ~MetalROIAlign() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs,
                          id<MTLComputeCommandEncoder> encoder) override;

private:
    id<MTLComputePipelineState> mPipeline;
    id<MTLBuffer> mConstBuffer;
    std::pair<MTLSize, MTLSize> mThreads;

    int mPooledWidth;
    int mPooledHeight;
    float mSpatialScale;
    int mSamplingRatio;
    bool mAligned;
    PoolType mPoolType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalROIAlign_hpp */