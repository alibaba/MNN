//
//  NNAPIUnary.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIUNARY_HPP
#define MNN_NNAPIUNARY_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"

namespace MNN {

class NNAPIUnary : public NNAPICommonExecution {
public:
    NNAPIUnary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIUnary() = default;
};
} // namespace MNN

#endif // MNN_NNAPIUNARY_HPP
