//
//  TRTBatchMatMul.hpp
//  MNN
//
//  Created by MNN on 2021/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTBatchMatMul_HPP
#define MNN_TRTBatchMatMul_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTBatchMatMul : public TRTCommonExecution {
public:
    TRTBatchMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTBatchMatMul() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
};

} // namespace MNN

#endif // MNN_TRTBatchMatMul_HPP
