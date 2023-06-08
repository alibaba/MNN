//
//  NPUSliceTf.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUSliceTf_HPP
#define NPUDEMO_NPUSliceTf_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUSliceTf : public NPUCommonExecution {
public:
    NPUSliceTf(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUSliceTf() = default;

private:
    hiai::op::Const mConst_start;
    hiai::op::Const mConst_size;

};
} // namespace MNN

#endif // NPUDEMO_NPUSliceTf_HPP
