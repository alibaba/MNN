//
//  ShapeExpandDims.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class ExpandDimsComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        const int inputSize = (int)inputs.size();
        MNN_ASSERT(2 == inputSize || 1 == inputSize);
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];

        // default -1
        int dim = -1;
        if (inputSize == 2) {
            // read dim from the second input
            auto dims = inputs[1];
            dim       = dims->host<int32_t>()[0];
        } else {
            // get dim from expand_dims parameter(axis)
            auto param = op->main_as_ExpandDims();
            dim        = param->axis();
        }

        if (dim < 0) {
            dim = input->dimensions() + 1 + dim;
        }
        output->buffer().type       = input->buffer().type;
        int outputShapeDims = 0;

        for (int i = 0; i < input->buffer().dimensions; i++) {
            if (i == dim) {
                output->buffer().dim[outputShapeDims++].extent = 1;
            }
            output->buffer().dim[outputShapeDims++].extent = input->buffer().dim[i].extent;
        }
        if (dim == input->buffer().dimensions) {
            output->buffer().dim[outputShapeDims++].extent = 1;
        }
        output->buffer().dimensions = outputShapeDims;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(ExpandDimsComputer, OpType_ExpandDims, {1});

} // namespace MNN
