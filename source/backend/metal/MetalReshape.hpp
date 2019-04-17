//
//  MetalReshape.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReshape_hpp
#define MetalReshape_hpp

#include <memory>
#include "Execution.hpp"
#include "MetalDefine.h"
#include "Tensor_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalReshape : public Execution {
public:
    MetalReshape(Backend *backend, MNN_DATA_FORMAT dimType);
    virtual ~MetalReshape() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    MNN_DATA_FORMAT mDimType;
    std::shared_ptr<Tensor> mMiddle;
    std::shared_ptr<Tensor> mCarbon;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReshape_hpp */
