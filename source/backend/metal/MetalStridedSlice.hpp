//
//  MetalStridedSlice.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalStridedSlice_hpp
#define MetalStridedSlice_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalStridedSlice : public Execution {
public:
    MetalStridedSlice(Backend *backend, const StridedSliceParam *s);
    virtual ~MetalStridedSlice() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const StridedSliceParam *mParam = NULL;
    id<MTLBuffer> mDims             = nil;
    id<MTLBuffer> mMask             = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalStridedSlice_hpp */
