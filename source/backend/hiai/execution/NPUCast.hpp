//
//  NPUCast.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUCast_HPP
#define NPUDEMO_NPUCast_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUCast : public NPUCommonExecution {
public:
    NPUCast(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUCast() = default;
};

} // namespace MNN

#endif // NPUDEMO_NPUCast_HPP
