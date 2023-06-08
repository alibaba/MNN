//
//  NPUConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUCONVOLUTION_HPP
#define MNN_NPUCONVOLUTION_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUConvolution : public NPUCommonExecution {
public:
    NPUConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUConvolution() = default;

private:
    hiai::op::Const mConst_w;
    hiai::op::Const mConst_b;

    shared_ptr<hiai::op::Activation> mRelu_conv;
};

} // namespace MNN

#endif // MNN_NPUCONVOLUTION_HPP
