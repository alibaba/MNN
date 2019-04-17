//
//  MetalCropAndResize.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalCropAndResize_hpp
#define MetalCropAndResize_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalCropAndResize : public Execution {
public:
    MetalCropAndResize(Backend *backend, float extrapolation, CropAndResizeMethod method);
    virtual ~MetalCropAndResize() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mExtrapolation;
    CropAndResizeMethod mMethod;
    id<MTLBuffer> mShape;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalCropAndResize_hpp */
