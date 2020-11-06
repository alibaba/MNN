//
//  TRTSelect.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTSelect_HPP
#define MNN_TRTSelect_HPP

#ifdef MNN_USE_TRT7

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTSelect : public TRTCommonExecution {
public:
    TRTSelect(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTSelect() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

} // namespace MNN
#endif
#endif // MNN_TRTSelect_HPP
