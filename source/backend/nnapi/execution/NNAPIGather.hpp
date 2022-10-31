//
//  NNAPIGather.hpp
//  MNN
//
//  Created by MNN on 2022/10/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPGATHER_HPP
#define MNN_NNAPGATHER_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIGather : public NNAPICommonExecution {
public:
    NNAPIGather(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIGather() = default;
};
} // namespace MNN

#endif // MNN_NNAPGATHER_HPP
