//
//  NNAPIActivation.hpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIACTIVATION_HPP
#define MNN_NNAPIACTIVATION_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIActivation : public NNAPICommonExecution {
public:
    NNAPIActivation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIActivation() = default;
};
} // namespace MNN

#endif // MNN_NNAPIACTIVATION_HPP
