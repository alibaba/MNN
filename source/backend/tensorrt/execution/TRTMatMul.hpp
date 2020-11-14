//
//  TRTMatMul.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTMatMul_HPP
#define MNN_TRTMatMul_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTMatMul : public TRTCommonExecution {
public:
    TRTMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTMatMul() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
};

} // namespace MNN

#endif // MNN_TRTMatMul_HPP
