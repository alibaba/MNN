//
//  MetalGridSample.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalGridSample_hpp
#define MetalGridSample_hpp

#import "core/Execution.hpp"
#import "MNN_generated.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalGridSample : public Execution {
public:
    MetalGridSample(Backend *backend, const GridSample* gridSample);
    virtual ~MetalGridSample() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mParams;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;

    SampleMode mMode;
    BorderMode mPaddingMode;
    bool mAlignCorners;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalGridSample_hpp */
