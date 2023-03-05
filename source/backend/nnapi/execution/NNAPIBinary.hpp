//
//  NNAPIBinary.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIBINARY_HPP
#define MNN_NNAPIBINARY_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"

namespace MNN {

class NNAPIBinary : public NNAPICommonExecution {
public:
    NNAPIBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIBinary() = default;
};
} // namespace MNN

#endif // MNN_NNAPIBINARY_HPP
