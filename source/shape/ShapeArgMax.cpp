//
//  ShapeArgMax.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Size Computer
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class ArgMaxComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto &input       = inputs[0]->buffer();
        auto &output      = outputs[0]->buffer();
        output.dimensions = input.dimensions;
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);

        auto argMax = op->main_as_ArgMax();

        const auto inputDimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        // Argmax can accept intput dimension-format:NC4HW4/NHWC NOW!
        MNN_ASSERT(inputDimensionFormat == MNN_DATA_FORMAT_NHWC || inputDimensionFormat == MNN_DATA_FORMAT_NC4HW4);

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = inputDimensionFormat;

        if (inputDimensionFormat == MNN_DATA_FORMAT_NHWC) {
            int axis = argMax->axis();
            if(axis < 0){
                axis = input.dimensions + axis;
            }
            // reduce axis dimension
            output.dimensions = input.dimensions - 1;
            for (int i = 0, j = 0; i < input.dimensions; ++i) {
                if (i == axis) {
                    continue;
                }
                output.dim[j].extent = input.dim[i].extent;
                j++;
            }
            output.dim[input.dimensions - 1].extent = 0;
            // set output data type to be INT(according to tensorflow implementation)
            output.type = halide_type_of<int>();
        } else {
            // Legacy code
            // key extent
            int keyExtent = argMax->topK();
            if (argMax->outMaxVal()) {
                keyExtent *= 2;
            }

            if (input.dim[3].extent > 1) {
                output.dim[3].extent = keyExtent;
            } else if (input.dim[2].extent > 1) { // iw = ow = 1
                output.dim[2].extent = keyExtent;
            } else { // iw = ow = 1, ih = oh = 1;
                output.dim[1].extent = keyExtent;
            }
        }

        return true;
    }
};

REGISTER_SHAPE(ArgMaxComputer, OpType_ArgMax);

} // namespace MNN
