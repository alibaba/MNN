//
//  CPUHistogram.hpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUHistogram_hpp
#define CPUHistogram_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUHistogram : public Execution {
public:
    CPUHistogram(Backend *backend, const Op* op);
    virtual ~CPUHistogram() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    template <typename T> ErrorCode histogram(Tensor* input, Tensor* output);
    int mChannel, mBinNum, mMin, mMax, mSize, mStride;
    float mAlpha, mBeta;
};

} // namespace MNN

#endif /* CPUHistogram_hpp */
