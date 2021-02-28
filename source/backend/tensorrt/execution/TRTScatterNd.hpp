//
//  TRTScatterNd.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTScatterNd_HPP
#define MNN_TRTScatterNd_HPP

#include "TRTBackend.hpp"
#include "TRTCommonExecution.hpp"

namespace MNN {

class TRTScatterNd : public TRTCommonExecution {
public:
    TRTScatterNd(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~TRTScatterNd() = default;
    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) override;

private:
    IConstantLayer *const_layer;
};


} // namespace MNN

#endif // MNN_TRTScatterNd_HPP
