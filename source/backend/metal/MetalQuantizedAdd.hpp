//
//  MetalQuantizedAdd.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalQuantizedAdd_hpp
#define MetalQuantizedAdd_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalQuantizedAdd : public Execution {
public:
    MetalQuantizedAdd(Backend *backend, const MNN::QuantizedAdd *add);
    virtual ~MetalQuantizedAdd() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN

#endif /* MNN_METAL_ENABLED */
#endif /* MetalQuantizedAdd_hpp */
