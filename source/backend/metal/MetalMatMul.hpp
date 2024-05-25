//
//  MetalMatMul.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalMatMul_hpp
#define MetalMatMul_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalMatMul : public MetalExecution {
public:
    MetalMatMul(Backend *backend, const MatMul *matmul, bool withBias);
    virtual ~MetalMatMul();
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mConstBuffer = nil;
    bool mTransposeA = false;
    bool mTransposeB = false;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalMatMul_hpp */
