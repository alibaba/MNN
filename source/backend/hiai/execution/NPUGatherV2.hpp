//
//  NPUGatherV2.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUGatherV2_HPP
#define MNN_NPUGatherV2_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {
class NPUGatherV2 : public NPUCommonExecution {
public:
    NPUGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUGatherV2() = default;
private:
    hiai::op::Const mConst;
};

} // namespace MNN

#endif // MNN_NPUGatherV2_HPP
