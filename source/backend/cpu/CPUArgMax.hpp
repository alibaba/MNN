//
//  CPUArgMax.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUArgMax_hpp
#define CPUArgMax_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUArgMax : public Execution {
public:
    enum ArgMinOrMax {
        ARGMIN,
        ARGMAX
    };
    CPUArgMax(Backend *backend, ArgMinOrMax mode, int topk, int outMaxVal, int softmaxThreshold, int axis);
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
    ArgMinOrMax mMode;
};

} // namespace MNN

#endif /* CPUArgMax_hpp */
