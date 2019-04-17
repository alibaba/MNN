//
//  CPUConcat.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConcat_hpp
#define CPUConcat_hpp

#include "Execution.hpp"

namespace MNN {
class CPUConcat : public Execution {
public:
    CPUConcat(Backend *b, int axis) : Execution(b), mOriginAxis(axis) {
        mAxis = mOriginAxis;
    }
    virtual ~CPUConcat() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mOriginAxis = 1;
    int mAxis       = 1;
    std::shared_ptr<Tensor> mTempOutput;
    bool mUseSlowMethod = false;
};

} // namespace MNN

#endif /* CPUConcat_hpp */
