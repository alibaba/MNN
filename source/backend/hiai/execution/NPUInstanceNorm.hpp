//
//  NPUInstanceNorm.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUInstanceNorm_HPP
#define NPUDEMO_NPUInstanceNorm_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUInstanceNorm : public NPUCommonExecution {
public:
    NPUInstanceNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
    virtual ~NPUInstanceNorm() = default;

private:
    hiai::op::Const mScale;
    hiai::op::Const mBias;
};
} // namespace MNN

#endif // NPUDEMO_NPUBatchnorm_HPP
