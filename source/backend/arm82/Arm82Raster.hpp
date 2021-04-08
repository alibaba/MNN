//
//  Arm82Raster.hpp
//  MNN
//
//  Created by MNN on 2020/5/25.
//  Copyright © 2018 Alibaba. All rights reserved.
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Raster_hpp
#define Arm82Raster_hpp
#include "Arm82Backend.hpp"
#include "core/Execution.hpp"
#include <map>
#include "core/TensorUtils.hpp"
namespace MNN {
class Arm82Raster : public Execution {
public:
    Arm82Raster(Backend* bn) : Execution(bn) {
        // Do nothing
    }
    virtual ~ Arm82Raster() {
        // Do nothing
    }
    
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    std::map<Tensor*, std::shared_ptr<Tensor>> mTempInput;
    std::vector<std::pair<void*, Tensor::InsideDescribe::Region*>> mTempInputCopy;
    std::vector<std::pair<void*, Tensor::InsideDescribe::Region>> mFastBlit;
    std::shared_ptr<Tensor> mTempOutput;
    void* mOutputPtr;
    bool mNeedZero = false;
    bool mFast = false;
};
}
#endif /* Arm82Raster_hpp */
#endif
