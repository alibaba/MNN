//
//  MetalGather.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalGather_hpp
#define MetalGather_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalGather : public Execution {
public:
    MetalGather(Backend *backend);
    virtual ~MetalGather() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalGather_hpp */
