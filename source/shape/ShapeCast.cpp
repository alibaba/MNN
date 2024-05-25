//
//  ShapeCast.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"

namespace MNN {

class CastSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto output = outputs[0];
        auto input  = inputs[0];
        TensorUtils::copyShape(input, output, true);
        if (OpType_FloatToInt8 == op->type()) {
            output->buffer().type = halide_type_of<int8_t>();
            return true;
        }
        if (OpType_Int8ToFloat == op->type()) {
            output->buffer().type = halide_type_of<float>();
            return true;
        }

        const auto opParam = op->main_as_CastParam();
        outputs[0]->setType(opParam->dstT());

        return true;
    }
};
REGISTER_SHAPE(CastSizeComputer, OpType_Cast);
REGISTER_SHAPE(CastSizeComputer, OpType_FloatToInt8);
REGISTER_SHAPE(CastSizeComputer, OpType_Int8ToFloat);
} // namespace MNN
