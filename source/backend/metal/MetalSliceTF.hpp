//
//  MetalSliceTF.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalSliceTF_hpp
#define MetalSliceTF_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalSliceTF : public Execution {
public:
    MetalSliceTF(Backend *backend, DataType type);
    virtual ~MetalSliceTF() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType mType;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalSliceTF_hpp */
