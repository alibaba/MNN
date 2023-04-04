//
//  NPUExpandDims.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUExpandDims_HPP
#define NPUDEMO_NPUExpandDims_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUExpandDims : public NPUCommonExecution {
public:
    NPUExpandDims(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUExpandDims() = default;
 
private:
    hiai::op::Const mConst_d;
    hiai::op::Const shapeConst;
};
} // namespace MNN

#endif // NPUDEMO_NPUExpandDims_HPP
