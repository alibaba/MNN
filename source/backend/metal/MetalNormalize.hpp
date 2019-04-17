//
//  MetalNormalize.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalNormalize_hpp
#define MetalNormalize_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalNormalize : public Execution {
public:
    MetalNormalize(Backend *backend, const Normalize *normalize);
    virtual ~MetalNormalize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAcrossSpatial;
    int mChannelShared;
    float mEps;
    id<MTLBuffer> mScale;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalNormalize_hpp */
