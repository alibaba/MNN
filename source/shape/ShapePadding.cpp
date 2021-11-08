//
//  ShapePadding.cpp
//  MNN
//
//  Created by MNN on 2019/6/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class PaddingComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if ((2 != inputs.size() && 3 != inputs.size()) || 1 != outputs.size()) {
            MNN_ERROR("Padding inputs or outputs number error: %d -> %d\n", (int)inputs.size(), (int)outputs.size());
            return false;
        }
        auto input            = inputs[0];
        auto paddings         = inputs[1];
        auto output           = outputs[0];
        output->buffer().type = input->buffer().type;
        TensorUtils::copyShape(input, output, true);

        auto size = paddings->elementSize();
        if (size < output->dimensions() * 2) {
            MNN_ERROR("Padding blob size not match output's dimension\n");
            return false;
        }
        auto paddingPtr = paddings->host<int32_t>();
        auto dimensions = input->dimensions();
        for (int i = 0; i < dimensions; ++i) {
            output->setLength(i, input->length(i) + paddingPtr[2 * i] + paddingPtr[2 * i + 1]);
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(PaddingComputer, OpType_Padding, {1});
} // namespace MNN
