//
//  CPUDeconvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDeconvolutionDepthwise_hpp
#define CPUDeconvolutionDepthwise_hpp

#include "AutoStorage.h"
#include "CPUDeconvolution.hpp"

namespace MNN {
class CPUDeconvolutionDepthwise : public CPUDeconvolutionCommon {
public:
    CPUDeconvolutionDepthwise(const Op *convOp, Backend *b);
    virtual ~CPUDeconvolutionDepthwise();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::function<void(const float *, float *)> mFunction;
};
} // namespace MNN

#endif /* CPUDeconvolutionDepthwise_hpp */
