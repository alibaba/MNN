//
//  MetalUnary.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalUnary_hpp
#define MetalUnary_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalUnary : public Execution {
public:
    MetalUnary(Backend *backend, UnaryOpOperation optype);
    virtual ~MetalUnary() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    UnaryOpOperation mOpType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalUnary_hpp */
