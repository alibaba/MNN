//
//  MetalGatherV2.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalGatherV2_hpp
#define MetalGatherV2_hpp

#import "Execution.hpp"
#import "MetalDefine.h"
#import "Type_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalGatherV2 : public Execution {
public:
    MetalGatherV2(Backend *backend, DataType type);
    virtual ~MetalGatherV2() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalGatherV2_hpp */
