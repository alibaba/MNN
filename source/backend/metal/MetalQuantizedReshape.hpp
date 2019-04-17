//
//  MetalQuantizedReshape.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalQuantizedReshape_hpp
#define MetalQuantizedReshape_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalQuantizedReshape : public Execution {
public:
    MetalQuantizedReshape(Backend *backend);
    virtual ~MetalQuantizedReshape() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalQuantizedReshape_hpp */
