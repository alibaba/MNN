//
//  CPUSoftmax.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSoftmax_hpp
#define CPUSoftmax_hpp

#include "Execution.hpp"

namespace MNN {
class CPUSoftmax : public Execution {
public:
    CPUSoftmax(Backend *b, int axis);
    virtual ~CPUSoftmax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    Tensor mStorage;
    Tensor mMaxValue;
    Tensor mSumValue;
    bool mNeedUnpackC4;
};
} // namespace MNN

#endif /* CPUSoftmax_hpp */
