//
//  ShapeDropout.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"

namespace MNN {

class DropoutSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto output = outputs[0];
        auto mask = outputs[1];
        auto input  = inputs[0];
        TensorUtils::copyShape(input, output, true);
        TensorUtils::copyShape(input, mask, true);

        output->buffer().type          = input->buffer().type;
        mask->buffer().type          = input->buffer().type;

        return true;
    }
};
REGISTER_SHAPE(DropoutSizeComputer, OpType_Dropout);

} // namespace MNN
