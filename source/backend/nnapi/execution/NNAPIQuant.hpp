//
//  NNAPIQuant.hpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIQUANT_HPP
#define MNN_NNAPIQUANT_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIQuant : public NNAPICommonExecution {
public:
    NNAPIQuant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIQuant() = default;
};

class NNAPIDequant : public NNAPICommonExecution {
public:
    NNAPIDequant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIDequant() = default;
};
} // namespace MNN

#endif // MNN_NNAPIQUANT_HPP
