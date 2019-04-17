//
//  MetalReduction.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalReduction_hpp
#define MetalReduction_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalReduction : public Execution {
public:
    MetalReduction(Backend *backend, const ReductionParam *reduction);
    virtual ~MetalReduction() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    NSString *mKernel;
    std::vector<std::shared_ptr<Tensor>> mMiddles;
    std::vector<int> mDims;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalReduction_hpp */
