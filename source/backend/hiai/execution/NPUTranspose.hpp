//
//  NPUTranspose.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUTranspose_HPP
#define NPUDEMO_NPUTranspose_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUTranspose : public NPUCommonExecution {
public:
    NPUTranspose(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUTranspose() = default;
private:
    std::vector<int64_t> permutation;
    hiai::op::Const shapeConst;
};

} // namespace MNN

#endif // NPUDEMO_NPUTranspose_HPP
