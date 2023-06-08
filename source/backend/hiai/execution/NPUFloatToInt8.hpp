//
//  NPUFloatToInt8.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUFLOATTOINT8_HPP
#define MNN_NPUFLOATTOINT8_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUFloatToInt8 : public NPUCommonExecution {
public:
    NPUFloatToInt8(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUFloatToInt8() = default;
    
private:
    hiai::op::Const mConst_fliter;
    hiai::op::Const mConstMax;
    hiai::op::Const mConstMin;
};

} // namespace MNN

#endif // MNN_NPUFLOATTOINT8_HPP
