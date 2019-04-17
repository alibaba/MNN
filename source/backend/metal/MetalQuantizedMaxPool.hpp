//
//  MetalQuantizedMaxPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalQuantizedMaxPool_hpp
#define MetalQuantizedMaxPool_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalQuantizedMaxPool : public Execution {
public:
    MetalQuantizedMaxPool(Backend *backend, const QuantizedMaxPool *pool);
    virtual ~MetalQuantizedMaxPool() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    PoolPadType mPadType;
    int mPadX;
    int mPadY;
    int mActivationMin;
    int mActivationMax;
    id<MTLBuffer> mConstBuffer;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalQuantizedMaxPool_hpp */
