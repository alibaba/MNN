//
//  MetalResize.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalResize_hpp
#define MetalResize_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalResize : public Execution {
public:
    MetalResize(Backend *backend, float xScale, float yScale);
    virtual ~MetalResize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mXScale;
    float mYScale;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalResize_hpp */
