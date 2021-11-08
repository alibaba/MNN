//
//  TRTLayerNorm.hpp
//  MNN
//
//  Created by MNN on 2021/02/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTLayerNorm_HPP
#define MNN_TRTLayerNorm_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTLayerNorm : public TRTCommonExecution {
public:
    TRTLayerNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTLayerNorm() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
private:
    int mAxis;
};

} // namespace MNN

#endif // MNN_TRTLayerNorm_HPP
