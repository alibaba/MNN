//
//  TRTReduce.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTReduce_HPP
#define MNN_TRTReduce_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTReduce : public TRTCommonExecution {
public:
    TRTReduce(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTReduce() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
    int inputDim;
};

} // namespace MNN

#endif // MNN_TRTReduce_HPP
