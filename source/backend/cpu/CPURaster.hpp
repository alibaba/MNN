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
#include "core/OpCommonUtils.hpp"
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
    std::map<Tensor*, Tensor*> mTempInput;
    std::vector<std::pair<const Tensor*, Tensor::InsideDescribe::Region*>> mTempInputCopy;
    std::vector<std::pair<const Tensor*, Tensor::InsideDescribe::Region>> mFastBlit;
    std::shared_ptr<Tensor> mTempOutput;
    bool mNeedZero = false;
    bool mFast = false;
    OpCommonUtils::TensorConvertParameter mSingleConvert;
    std::vector<std::shared_ptr<Tensor::InsideDescribe::Region>> mCacheRegions;
    int32_t mZeroPoint = 0;
};
}
#endif
