//
//  MetalScale.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalScale_hpp
#define MetalScale_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalScale : public Execution {
public:
    MetalScale(Backend *backend, const Scale *scale);
    virtual ~MetalScale() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mScale;
    id<MTLBuffer> mBias;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalScale_hpp */
