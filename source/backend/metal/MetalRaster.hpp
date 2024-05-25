//
//  MetalRaster.hpp
//  MNN
//
//  Created by MNN on 2020/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalRaster_hpp
#define MetalRaster_hpp

#import "MetalExecution.hpp"
#include <map>

#if MNN_METAL_ENABLED
namespace MNN {

class MetalRaster : public MetalExecution {
public:
    MetalRaster(Backend *backend);
    virtual ~MetalRaster();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    static id<MTLComputePipelineState> getBlitPipeline(int bytes, Backend* backend, bool multiRegion);
    struct BlitInfo {
        std::pair<void*, size_t> blit;
        MTLSize local;
        MTLSize global;
    };
private:
    std::map<Tensor*, std::shared_ptr<Tensor>> mTempInput;
    std::map<Tensor*, BlitInfo> mTempInputCopy;
    std::shared_ptr<Tensor> mTempOutput;
    bool mNeedZero = false;
    Tensor* mOutputPtr = nullptr;
    id<MTLComputePipelineState> mBlitPipeline;
    std::vector<id<MTLBuffer>> mShapeTemp;
    id<MTLBuffer> mZeroCopy = nil;
    id<MTLComputePipelineState> mZeroPipeline;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalRaster_hpp */
