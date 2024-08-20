//
//  MetalExecution.hpp
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalExecution_hpp
#define MetalExecution_hpp

#include "core/Execution.hpp"
#import "MetalDefine.h"
#include <string>
#if MNN_METAL_ENABLED
namespace MNN {

class MetalExecution : public Execution {
public:
    MetalExecution(Backend *backend);
    virtual ~MetalExecution() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) = 0;

};
} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif
