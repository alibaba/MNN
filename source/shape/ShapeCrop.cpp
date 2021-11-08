//
//  ShapeCrop.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class CropSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(4 == inputs[0]->buffer().dimensions && 4 == inputs[1]->buffer().dimensions);
        MNN_ASSERT(inputs[0]->buffer().dimensions == inputs[1]->buffer().dimensions);

        auto& ibInput0 = inputs[0]->buffer();
        auto& ibInput1 = inputs[1]->buffer();
        auto& ob       = outputs[0]->buffer();

        ob.dimensions = ibInput1.dimensions;
        ::memcpy(ob.dim, ibInput1.dim, ibInput1.dimensions * sizeof(halide_dimension_t));

        auto cropParam = op->main_as_Crop();
        for (int i = 0; i < ibInput1.dimensions; ++i) {
            if (i < cropParam->axis()) {
                ob.dim[i].extent = ibInput0.dim[i].extent;
            }
        }
        ob.type = ibInput0.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(CropSizeComputer, OpType_Crop);

} // namespace MNN
