//
//  MetalTensorConverter.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalTensorConverter_hpp
#define MetalTensorConverter_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalTensorConverter : public Execution {
public:
    MetalTensorConverter(Backend *backend);
    virtual ~MetalTensorConverter() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalTensorConverter_hpp */
