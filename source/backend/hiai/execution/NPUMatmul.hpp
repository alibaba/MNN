//
//  NPUMatmul.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUMatmul_HPP
#define NPUDEMO_NPUMatmul_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUMatmul : public NPUCommonExecution {
public:
    NPUMatmul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUMatmul() = default;
   
private:
    ge::op::Const mConst;
};
} // namespace MNN

#endif // NPUDEMO_NPUMatmul_HPP
