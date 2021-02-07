//
//  TRTGatherV2.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTGatherV2_HPP
#define MNN_TRTGatherV2_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTGatherV2 : public TRTCommonExecution {
public:
    TRTGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTGatherV2() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;
private:
    int mAxis;
};

} // namespace MNN

#endif // MNN_TRTGatherV2_HPP
