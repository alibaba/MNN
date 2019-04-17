//
//  CPUArgMax.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUArgMax_hpp
#define CPUArgMax_hpp

#include "Execution.hpp"

namespace MNN {

class CPUArgMax : public Execution {
public:
    CPUArgMax(Backend *backend, int topk, int outMaxVal, int softmaxThreshold);
    virtual ~CPUArgMax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mBuffer;
    Tensor mInputBuffer;
    int mTopk;
    int mOutMaxVal;
    int mSoftmaxThreshold;
};

} // namespace MNN

#endif /* CPUArgMax_hpp */
