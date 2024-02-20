//
//  NPUScale.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUSCALE_HPP
#define NPUDEMO_NPUSCALE_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUScale : public NPUCommonExecution {
public:
    NPUScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUScale() = default;
    
private:
    hiai::op::Const mConst_fliter;
    hiai::op::Const mConst_bias;
    hiai::op::Const shapeConst;
};

} // namespace MNN

#endif // NPUDEMO_NPUSCALE_HPP
