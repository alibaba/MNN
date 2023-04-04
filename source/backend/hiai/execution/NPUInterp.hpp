//
//  NPUInterp.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUInterp_HPP
#define NPUDEMO_NPUInterp_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUInterp : public NPUCommonExecution {
public:
    NPUInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
    virtual ~NPUInterp() = default;

private:
    hiai::op::Const mConstShape;
};
} // namespace MNN

#endif // NPUDEMO_NPUInterp_HPP
