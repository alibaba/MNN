//
//  ShapeGather.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
class GatherComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto embedding = inputs[0];
        auto indices   = inputs[1];

        auto output                                       = outputs[0];
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(embedding)->dimensionFormat;
        output->buffer().dimensions = indices->buffer().dimensions + embedding->buffer().dimensions - 1;
        for (int i = 0; i < indices->buffer().dimensions; i++) {
            output->buffer().dim[i].extent = indices->buffer().dim[i].extent;
        }

        for (int i = 0; i < embedding->buffer().dimensions - 1; i++) {
            output->buffer().dim[i+indices->buffer().dimensions].extent = embedding->buffer().dim[i + 1].extent;
        }

        output->buffer().type = embedding->buffer().type;

        return true;
    }
};

REGISTER_SHAPE(GatherComputer, OpType_Gather);

} // namespace MNN
