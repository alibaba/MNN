//
//  NPUConvertTensor.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUConvertTensor_HPP
#define NPUDEMO_NPUConvertTensor_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUConvertTensor : public NPUCommonExecution {
public:
    NPUConvertTensor(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUConvertTensor() = default;
};
} // namespace MNN

#endif // NPUDEMO_NPUConvertTensor_HPP
