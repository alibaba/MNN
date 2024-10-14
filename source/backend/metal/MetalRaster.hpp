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
    std::map<Tensor*, BlitInfo> mTempInputCopy;
    bool mNeedZero = false;
    Tensor* mOutputPtr = nullptr;
    std::vector<id<MTLComputePipelineState>> mBlitPipeline;
    id<MTLBuffer> mZeroCopy = nil;
    id<MTLComputePipelineState> mZeroPipeline;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalRaster_hpp */
