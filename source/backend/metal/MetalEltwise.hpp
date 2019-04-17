//
//  MetalEltwise.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalEltwise_hpp
#define MetalEltwise_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalEltwise : public Execution {
public:
    MetalEltwise(Backend *backend, EltwiseType type);
    virtual ~MetalEltwise() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    EltwiseType mType;
    void encode(NSString *kernel, const Tensor *input0, const Tensor *input1, const Tensor *output);
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalEltwise_hpp */
