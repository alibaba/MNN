//
//  NPUEltwiseInt8.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUEltwiseInt8_HPP
#define NPUDEMO_NPUEltwiseInt8_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUEltwiseInt8 : public NPUCommonExecution {
public:
    NPUEltwiseInt8(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUEltwiseInt8() = default;
   
private:
    hiai::op::Const mConst_scale0;
    hiai::op::Const mConst_scale1;
    hiai::op::Const mConstMin0;
    hiai::op::Const mConstMax0;
    hiai::op::Const mConstMin1;
    hiai::op::Const mConstMax1;
    hiai::op::Const mConstMin;
    hiai::op::Const mConstMax;
};
} // namespace MNN

#endif // NPUDEMO_NPUEltwiseInt8_HPP
