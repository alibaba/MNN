//
//  NPUArgMax.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUArgMax_HPP
#define NPUDEMO_NPUArgMax_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUArgMax : public NPUCommonExecution {
public:
    NPUArgMax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUArgMax() = default;
private:
    hiai::op::Const mConst_axis;
};

} // namespace MNN

#endif // NPUDEMO_NPUArgMax_HPP
