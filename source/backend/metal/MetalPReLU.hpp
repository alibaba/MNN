//
//  MetalPReLU.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPReLU_hpp
#define MetalPReLU_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalPReLU : public Execution {
public:
    MetalPReLU(Backend *backend, const float *slope, int count);
    virtual ~MetalPReLU() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mSlope;
    bool mShareChannel = false;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPReLU_hpp */
