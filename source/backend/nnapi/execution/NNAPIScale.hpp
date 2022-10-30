//
//  NNAPIScale.hpp
//  MNN
//
//  Created by MNN on 2022/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPISCALE_HPP
#define MNN_NNAPISCALE_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIScale : public NNAPICommonExecution {
public:
    NNAPIScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIScale() = default;
};
} // namespace MNN

#endif // MNN_NNAPISCALE_HPP
