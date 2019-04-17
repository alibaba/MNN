//
//  MetalBinary.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBinary_hpp
#define MetalBinary_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalBinary : public Execution {
public:
    MetalBinary(Backend *backend, int binaryType);
    virtual ~MetalBinary() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mBinaryType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalBinary_hpp */
