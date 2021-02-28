//
//  TRTOneHot.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTOneHot_HPP
#define MNN_TRTOneHot_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTOneHot : public TRTCommonExecution {
public:
    TRTOneHot(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTOneHot() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
private:
    int mAxis;
};

} // namespace MNN

#endif // MNN_TRTOneHot_HPP
