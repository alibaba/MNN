//
//  NPUBroadCastTo.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUBroadCastTo_HPP
#define NPUDEMO_NPUBroadCastTo_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUBroadCastTo : public NPUCommonExecution {
public:
    NPUBroadCastTo(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUBroadCastTo() = default;
private:
    hiai::op::Const mConst_s;
};

} // namespace MNN

#endif // NPUDEMO_NPUBroadCastTo_HPP
