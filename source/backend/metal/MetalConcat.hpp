//
//  MetalConcat.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConcat_hpp
#define MetalConcat_hpp

#import <memory>
#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConcat : public Execution {
public:
    MetalConcat(Backend *backend, int axis);
    virtual ~MetalConcat() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    id<MTLBuffer> mBlits;
    std::vector<std::shared_ptr<Tensor>> mTempInputs;
    std::shared_ptr<Tensor> mTempOutput;
    bool mFastMode = false;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalConcat_hpp */
