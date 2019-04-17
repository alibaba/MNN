//
//  MetalFill.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalFill_hpp
#define MetalFill_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalFill : public Execution {
public:
    MetalFill(Backend *backend);
    virtual ~MetalFill() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalFill_hpp */
