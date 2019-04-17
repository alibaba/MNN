//
//  MetalPack.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPack_hpp
#define MetalPack_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalPack : public Execution {
public:
    MetalPack(Backend *backend, DataType type, int axis);
    virtual ~MetalPack() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mType;
    int mAxis;
    id<MTLBuffer> mBlits;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPack_hpp */
