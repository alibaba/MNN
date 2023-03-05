//
//  NNAPIReduction.hpp
//  MNN
//
//  Created by MNN on 2022/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIREDUCTION_HPP
#define MNN_NNAPIREDUCTION_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"

namespace MNN {

class NNAPIReduction : public NNAPICommonExecution {
public:
    NNAPIReduction(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIReduction() = default;
};
} // namespace MNN

#endif // MNN_NNAPIREDUCTION_HPP
