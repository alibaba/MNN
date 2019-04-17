//
//  MetalCrop.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalCrop_hpp
#define MetalCrop_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalCrop : public Execution {
public:
    MetalCrop(Backend *backend, const Crop *crop);
    virtual ~MetalCrop() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    int mOffsetX;
    int mOffsetY;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalCrop_hpp */
