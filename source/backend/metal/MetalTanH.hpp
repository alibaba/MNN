//
//  MetalTanH.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalTanH_hpp
#define MetalTanH_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalTanH : public Execution {
public:
    MetalTanH(Backend *backend);
    virtual ~MetalTanH() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalTanH_hpp */
