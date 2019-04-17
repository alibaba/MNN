//
//  MetalTranspose.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalTranspose_hpp
#define MetalTranspose_hpp

#import "Execution.hpp"
#import "MetalDefine.h"
#import "Type_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalTranspose : public Execution {
public:
    MetalTranspose(Backend *backend, DataType type);
    virtual ~MetalTranspose() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mType;
    id<MTLBuffer> mInDims;
    id<MTLBuffer> mOutStrides;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalTranspose_hpp */
