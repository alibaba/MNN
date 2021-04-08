//
//  CPUPool.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPool.hpp"
#include "math/Vec.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;
using Vec16 = MNN::Math::Vec<int8_t, 16>;

namespace MNN {


class CPUPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs[0]->getType() == halide_type_of<int8_t>()) {
            return new CPUPool<int8_t, Vec16>(backend, op->main_as_Pool());
        }
        return new CPUPool<float, Vec4>(backend, op->main_as_Pool());
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolCreator, OpType_Pooling);

} // namespace MNN
