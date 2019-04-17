//
//  MetalReLU.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReLU_hpp
#define MetalReLU_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalReLU : public Execution {
public:
    MetalReLU(Backend *backend, float slope);
    virtual ~MetalReLU() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mSlope;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReLU_hpp */
