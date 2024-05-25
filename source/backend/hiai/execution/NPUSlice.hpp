//
//  NPUSlice.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUSlice_HPP
#define NPUDEMO_NPUSlice_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUSlice : public NPUCommonExecution {
public:
    NPUSlice(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUSlice() = default;
};
} // namespace MNN

#endif // NPUDEMO_NPUSlice_HPP
