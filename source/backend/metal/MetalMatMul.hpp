//
//  MetalMatMul.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalMatMul_hpp
#define MetalMatMul_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalMatMul : public Execution {
public:
    MetalMatMul(Backend *backend, const MatMul *matmul);
    virtual ~MetalMatMul() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mBias   = nil;
    id<MTLBuffer> mWeight = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalMatMul_hpp */
