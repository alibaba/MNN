//
//  MetalSlice.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalSlice_hpp
#define MetalSlice_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalSlice : public Execution {
public:
    MetalSlice(Backend *backend, int axis);
    virtual ~MetalSlice() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalSlice_hpp */
