//
//  NPUNonMaxSuppressionV2.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUNonMaxSuppressionV2_HPP
#define NPUDEMO_NPUNonMaxSuppressionV2_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUNonMaxSuppressionV2 : public NPUCommonExecution {
public:
    NPUNonMaxSuppressionV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUNonMaxSuppressionV2() = default;
   
private:
};
} // namespace MNN

#endif // NPUDEMO_NPUNonMaxSuppressionV2_HPP
