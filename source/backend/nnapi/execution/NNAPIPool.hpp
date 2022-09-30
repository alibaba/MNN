//
//  NNAPIPool.hpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIPOOL_HPP
#define MNN_NNAPIPOOL_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIPool : public NNAPICommonExecution {
public:
    NNAPIPool(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIPool() = default;
};
} // namespace MNN

#endif // MNN_NNAPIPOOL_HPP
