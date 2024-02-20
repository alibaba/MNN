//
//  NPUDeconvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUDeconvolutionDepthwise_HPP
#define MNN_NPUDeconvolutionDepthwise_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUDeconvolutionDepthwise : public NPUCommonExecution {
public:
    NPUDeconvolutionDepthwise(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUDeconvolutionDepthwise() = default;

private:
    hiai::op::Const mConst_w;
    hiai::op::Const mConst_b;

    shared_ptr<hiai::op::Activation> mRelu_conv;
};

} // namespace MNN

#endif // MNN_NPUDeconvolutionDepthwise_HPP
