//
//  NNAPIArgMax.hpp
//  MNN
//
//  Created by MNN on 2022/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPARGMAX_HPP
#define MNN_NNAPARGMAX_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIArgMax : public NNAPICommonExecution {
public:
    NNAPIArgMax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIArgMax() = default;
};
} // namespace MNN

#endif // MNN_NNAPARGMAX_HPP
