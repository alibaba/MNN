//
//  NPUConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUCONVOLUTIONDEPTHWISE_HPP
#define NPUDEMO_NPUCONVOLUTIONDEPTHWISE_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUConvolutionDepthwise : public NPUCommonExecution {
public:
    NPUConvolutionDepthwise(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUConvolutionDepthwise() = default;
private:
    hiai::op::Const mConst_w;
    hiai::op::Const mConst_b;

    shared_ptr<hiai::op::Activation> mRelu_conv;
};

} // namespace MNN

#endif // NPUDEMO_NPUCONVOLUTIONDEPTHWISE_HPP
