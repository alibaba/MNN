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
    CPUArgMax(Backend *backend, int topk, int outMaxVal, int softmaxThreshold, int axis);
    virtual ~CPUArgMax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mInputBuffer;
    Tensor mOutputBuffer;
    int mTopk;
    int mOutMaxVal;
    int mSoftmaxThreshold;
    int mAxis;
    int mNum;
    int mDim;
    int mKeyExtent;
    bool mFromNHWC;
};

} // namespace MNN

#endif /* CPUArgMax_hpp */
