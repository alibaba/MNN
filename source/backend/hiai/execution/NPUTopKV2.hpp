//
//  NPUTopKV2.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUTopKV2_HPP
#define MNN_NPUTopKV2_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {
class NPUTopKV2 : public NPUCommonExecution {
public:
    NPUTopKV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUTopKV2() = default;

private:
    hiai::op::Const mConst_w;
};

} // namespace MNN

#endif // MNN_NPUTopKV2_HPP
