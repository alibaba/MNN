//
//  BF16Pooling.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPool.hpp"
#include "VecHalf.hpp"
#include "BF16Backend.hpp"

namespace MNN {
using Vec4Half = MNN::Math::VecHalf<4>;

class BF16PoolingCreator : public BF16Backend::BF16Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPool<int16_t, Vec4Half>(backend, op->main_as_Pool());
    }
};

REGISTER_BF16_OP_CREATOR(OpType_Pooling, BF16PoolingCreator);

} // namespace MNN
