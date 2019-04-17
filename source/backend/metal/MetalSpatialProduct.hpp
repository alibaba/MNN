//
//  MetalSpatialProduct.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalSpatialProduct_hpp
#define MetalSpatialProduct_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalSpatialProduct : public Execution {
public:
    MetalSpatialProduct(Backend *backend);
    virtual ~MetalSpatialProduct() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalSpatialProduct_hpp */
