//
//  NPUPadding.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUPadding_HPP
#define MNN_NPUPadding_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {
class NPUPadding : public NPUCommonExecution {
public:
    NPUPadding(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUPadding() = default;
private:
    hiai::op::Const mConst;
    std::vector<int32_t> mPadData;
};

} // namespace MNN

#endif // MNN_NPUPadding_HPP
