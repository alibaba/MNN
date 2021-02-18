//
//  TRTInterp.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTInterp_HPP
#define MNN_TRTInterp_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTInterp : public TRTCommonExecution {
public:
    TRTInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTInterp() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
};

} // namespace MNN

#endif // MNN_TRTInterp_HPP
