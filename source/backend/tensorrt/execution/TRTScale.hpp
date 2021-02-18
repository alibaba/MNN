//
//  TRTScale.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTScale_HPP
#define MNN_TRTScale_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTScale : public TRTCommonExecution {
public:
    TRTScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTScale() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

} // namespace MNN

#endif // MNN_TRTScale_HPP
