//
//  NPUBatchMatMul.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUBatchMatMul_HPP
#define NPUDEMO_NPUBatchMatMul_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUBatchMatMul : public NPUCommonExecution {
public:
    NPUBatchMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUBatchMatMul() = default;
   
private:
    ge::op::Const mConst;
};
} // namespace MNN

#endif // NPUDEMO_NPUBatchMatMul_HPP
