//
//  NPULRN.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPULRN_HPP
#define NPUDEMO_NPULRN_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPULRN : public NPUCommonExecution {
public:
    NPULRN(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPULRN() = default;
};
} // namespace MNN

#endif // NPUDEMO_NPULRN_HPP
