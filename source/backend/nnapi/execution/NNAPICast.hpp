//
//  NNAPICast.hpp
//  MNN
//
//  Created by MNN on 2023/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPICAST_HPP
#define MNN_NNAPICAST_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPICast : public NNAPICommonExecution {
public:
    NNAPICast(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPICast() = default;
};
} // namespace MNN

#endif // MNN_NNAPICAST_HPP
