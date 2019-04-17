//
//  MetalSeLU.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalSeLU_hpp
#define MetalSeLU_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalSeLU : public Execution {
public:
    MetalSeLU(Backend *backend, float scale, float alpha);
    virtual ~MetalSeLU() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    id<MTLBuffer> mConst;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalSeLU_hpp */
