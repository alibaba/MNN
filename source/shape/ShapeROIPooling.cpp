//
//  ShapeROIPooling.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

// Size Computer
class ROIPoolingComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(2 == inputs.size() || 3 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        if (inputs.size() == 2) {
            // copy dims
            auto &input  = inputs[0]->buffer();
            auto &output = outputs[0]->buffer();
            memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
            output.type = halide_type_of<float>();

            // width & height
            auto roi             = op->main_as_RoiParameters();
            output.dim[3].extent = roi->pooledWidth();
            output.dim[2].extent = roi->pooledHeight();
            output.dim[0].extent = inputs[1]->batch();
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        }

        // backward mode, third input is backward diff, output is the grad of inputs[0]
        if (inputs.size() == 3) {
            TensorUtils::copyShape(inputs[0], outputs[0], true);
            outputs[0]->buffer().type = inputs[0]->getType();
        }

        return true;
    }
};

REGISTER_SHAPE(ROIPoolingComputer, OpType_ROIPooling);
} // namespace MNN
