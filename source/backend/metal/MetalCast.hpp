//
//  MetalCast.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalCast_hpp
#define MetalCast_hpp

#import "Execution.hpp"
#import "MetalDefine.h"
#import "Type_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalCast : public Execution {
public:
    MetalCast(Backend *backend, DataType srcType, DataType dstType);
    virtual ~MetalCast() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mSrcType;
    DataType mDstType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalCast_hpp */
