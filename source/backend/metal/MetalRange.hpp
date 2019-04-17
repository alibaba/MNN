//
//  MetalRange.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalRange_hpp
#define MetalRange_hpp

#import "Execution.hpp"
#import "MetalDefine.h"
#import "Type_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalRange : public Execution {
public:
    MetalRange(Backend *backend, DataType type);
    virtual ~MetalRange() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalRange_hpp */
