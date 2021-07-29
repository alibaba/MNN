//
//  CPURaster.hpp
//  MNN
//
//  Created by MNN on b'2020/04/02'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPURaster_hpp
#define CPURaster_hpp
#include "CPUBackend.hpp"
#include <map>
#include <set>
#include "core/TensorUtils.hpp"
namespace MNN {
class CPURaster : public Execution {
public:
    CPURaster(Backend* bn) : Execution(bn) {
        // Do nothing
    }
    virtual ~ CPURaster() {
        // Do nothing
    }
    
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    void executeFaster(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const;
    void tensorConvert(Tensor* input, Tensor* output, int bytes);
private:
    std::map<Tensor*, std::shared_ptr<Tensor>> mTempInput;
    std::vector<std::pair<void*, Tensor::InsideDescribe::Region*>> mTempInputCopy;
    std::vector<std::pair<void*, Tensor::InsideDescribe::Region>> mFastBlit;
    std::shared_ptr<Tensor> mTempOutput;
    void* mOutputPtr;
    bool mNeedZero = false;
    bool mFast = false;
    int mSingleConvert = 0;
    std::vector<Tensor::InsideDescribe::Region> mCacheRegions;
};
}
#endif
