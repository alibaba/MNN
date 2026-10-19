//
//  NPULayerNorm.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPULayerNorm_HPP
#define NPUDEMO_NPULayerNorm_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPULayerNorm : public NPUCommonExecution {
public:
    NPULayerNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
    virtual ~NPULayerNorm() = default;

private:
    hiai::op::Const constw;
    hiai::op::Const constb;
    hiai::op::Const constReshapeShape;
    hiai::op::Const constOutputShape;
    hiai::op::Const constNormGamma;
    hiai::op::Const constNormBeta;
};
} // namespace MNN

#endif // NPUDEMO_NPULayerNorm_HPP
