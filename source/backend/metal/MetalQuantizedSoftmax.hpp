//
//  MetalQuantizedSoftmax.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalQuantizedSoftmax_hpp
#define MetalQuantizedSoftmax_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalQuantizedSoftmax : public Execution {
public:
    MetalQuantizedSoftmax(Backend *backend, float beta, float inputScale);
    virtual ~MetalQuantizedSoftmax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mConst = nil;

    int mInputMultiplier;
    int mInputLeftShift;
    int mDiffMin;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalQuantizedSoftmax_hpp */
