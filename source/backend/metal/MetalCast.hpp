//
//  MetalCast.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalCast_hpp
#define MetalCast_hpp

#import "MetalExecution.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalCast : public MetalExecution {
public:
    MetalCast(Backend *backend, id<MTLComputePipelineState> pipeline);
    virtual ~MetalCast() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    static NSString* getScalarType(const halide_type_t& srcType, bool useFp16);
    static NSString* getVecType(const halide_type_t& srcType, bool useFp16);

protected:
    id<MTLBuffer> mConstBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalCast_hpp */
