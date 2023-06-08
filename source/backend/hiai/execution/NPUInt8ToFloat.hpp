//
//  NPUInt8ToFloat.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUINT8TOFLOAT_HPP
#define MNN_NPUINT8TOFLOAT_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"
namespace MNN {

class NPUInt8ToFloat : public NPUCommonExecution {
public:
    NPUInt8ToFloat(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUInt8ToFloat() = default;
   
private:
    hiai::op::Const mConst_fliter;
    hiai::op::Const mConstMax;
    hiai::op::Const mConstMin;
};

} // namespace MNN

#endif // MNN_NPUINT8TOFLOAT_HPP
