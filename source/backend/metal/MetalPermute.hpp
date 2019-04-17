//
//  MetalPermute.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPermute_hpp
#define MetalPermute_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalPermute : public Execution {
public:
    MetalPermute(Backend *backend, const Permute *permute);
    virtual ~MetalPermute() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mDims;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPermute_hpp */
