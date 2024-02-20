//
//  NPUReduction.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUReduction_HPP
#define NPUDEMO_NPUReduction_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUReduction : public NPUCommonExecution {
public:
    NPUReduction(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUReduction() = default;

private:
    hiai::op::Const mConstAxis;
    hiai::op::Const shapeConst;
};
} // namespace MNN

#endif // NPUDEMO_NPUReduction_HPP
