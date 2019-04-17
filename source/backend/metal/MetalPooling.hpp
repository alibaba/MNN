//
//  MetalPooling.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPooling_hpp
#define MetalPooling_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalPooling : public Execution {
public:
    MetalPooling(Backend *backend, const Pool *pooling);
    virtual ~MetalPooling() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mGlobal;
    PoolType mPoolType;
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    int mPadX;
    int mPadY;
    id<MTLBuffer> mConstBuffer;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPooling_hpp */
