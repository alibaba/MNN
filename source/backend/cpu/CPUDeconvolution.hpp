//
//  CPUDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolution_hpp
#define CPUDeconvolution_hpp

#include <stdio.h>
#include <mutex>
#include "AutoStorage.h"
#include "CPUConvolution.hpp"

namespace MNN {
class CPUDeconvolutionCommon : public CPUConvolution {
public:
    CPUDeconvolutionCommon(const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionCommon();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::shared_ptr<Tensor> mBias;
    int mSrcCount;
};

class CPUDeconvolution : public CPUDeconvolutionCommon {
public:
    CPUDeconvolution(const Op *convOp, Backend *b);
    virtual ~CPUDeconvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mTempSrcBuffer;
    std::shared_ptr<Tensor> mTempColBuffer;
    std::mutex mLock;
    std::function<void(const float *, float *, int)> mFunction;
    int mThreadNumber = 1;
};
} // namespace MNN
#endif /* CPUDeconvolution_hpp */
