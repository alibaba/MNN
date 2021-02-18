//
//  TRTGather.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTGather_HPP
#define MNN_TRTGather_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTGather : public TRTCommonExecution {
public:
    TRTGather(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTGather() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

} // namespace MNN

#endif // MNN_TRTGather_HPP
