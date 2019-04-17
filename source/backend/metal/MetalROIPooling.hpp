//
//  MetalROIPooling.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalROIPooling_hpp
#define MetalROIPooling_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalROIPooling : public Execution {
public:
    MetalROIPooling(Backend *backend, float spatialScale);
    virtual ~MetalROIPooling() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mSpatialScale;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalROIPooling_hpp */
