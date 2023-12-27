//
//  MetalReduction.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReduction_hpp
#define MetalReduction_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalReduction : public MetalExecution {
public:
    MetalReduction(Backend *backend, const ReductionParam *reduction, halide_type_t type);
    virtual ~MetalReduction() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    int mAxis;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    id<MTLBuffer> mConst;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReduction_hpp */
