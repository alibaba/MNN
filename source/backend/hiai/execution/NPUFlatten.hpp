//
//  NPUFlatten.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUFlatten_HPP
#define NPUDEMO_NPUFlatten_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUFlatten : public NPUCommonExecution {
public:
    NPUFlatten(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUFlatten() = default;
};
} // namespace MNN

#endif // NPUDEMO_NPUFlatten_HPP
