//
//  MetalBinary.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBinary_hpp
#define MetalBinary_hpp

#import "MetalExecution.hpp"
#include <string>
#if MNN_METAL_ENABLED
namespace MNN {

class MetalBinary : public MetalExecution {
public:
    MetalBinary(Backend *backend, std::string type, const MNN::Op *op);
    virtual ~MetalBinary() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mConstBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    int mActivationType = 0;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalBinary_hpp */
