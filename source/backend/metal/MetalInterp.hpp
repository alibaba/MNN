//
//  MetalInterp.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalInterp_hpp
#define MetalInterp_hpp

#include "Execution.hpp"
#include "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalInterp : public Execution {
public:
    MetalInterp(Backend *backend, float widthScale, float heightScale, int32_t outputWidth, int32_t outputHeight,
                int32_t reiszeType, bool alignCorner);
    virtual ~MetalInterp() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mWidthScale;
    float mHeightScale;
    int32_t mOutputWidth;
    int32_t mOutputHeight;
    int32_t mReiszeType;
    bool mAlignCorner;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalInterp_hpp */
