//
//  NPUActivation.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUActivation_HPP
#define NPUDEMO_NPUActivation_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUActivation : public NPUCommonExecution {
public:
    NPUActivation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int type);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUActivation() = default;
private:
    hiai::op::Const mConst_w;
    hiai::op::Const shapeConst;
    int mType;
};

} // namespace MNN

#endif // NPUDEMO_NPUActivation_HPP
