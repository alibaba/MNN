//
//  NPUEltwise.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUELTWISE_HPP
#define NPUDEMO_NPUELTWISE_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUEltwise : public NPUCommonExecution {
public:
    NPUEltwise(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUEltwise() = default;
   
private:
};
} // namespace MNN

#endif // NPUDEMO_NPUELTWISE_HPP
