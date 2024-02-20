//
//  NPUSoftmax.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUSOFTMAX_HPP
#define MNN_NPUSOFTMAX_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {
class NPUSoftmax : public NPUCommonExecution {
public:
    NPUSoftmax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUSoftmax() = default;
};

} // namespace MNN

#endif // MNN_NPUSOFTMAX_HPP
