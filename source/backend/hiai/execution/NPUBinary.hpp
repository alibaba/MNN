//
//  NPUBinary.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUBinary_HPP
#define NPUDEMO_NPUBinary_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUBinary : public NPUCommonExecution {
public:
    void OpInsert(int binary_type, string opName, 
                  ge::Operator& input0, ge::Operator& input1,
                  const std::vector<Tensor *> &outputs, int activationType);
    NPUBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUBinary() = default;
   
private:
    ge::op::Const mConst;

};
} // namespace MNN

#endif // NPUDEMO_NPUBinary_HPP
