//
//  ShapeExpandDims.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class ExpandDimsComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];
        auto dims   = inputs[1];
        int dim     = dims->host<int32_t>()[0];
        if (dim == -1) {
            dim = input->dimensions() + 1 + dim;
        }

        std::vector<int> outputShape;
        for (int i = 0; i < input->buffer().dimensions; i++) {
            if (i == dim) {
                outputShape.push_back(1);
            }
            outputShape.push_back(input->buffer().dim[i].extent);
        }
        if (dim == input->buffer().dimensions) {
            outputShape.push_back(1);
        }
        output->buffer().dimensions = (int)outputShape.size();
        output->buffer().type       = input->buffer().type;
        int previousStride          = 1;
        for (int i = output->buffer().dimensions - 1; i >= 0; i--) {
            output->buffer().dim[i].stride = previousStride;
            output->buffer().dim[i].extent = outputShape[i];
            output->buffer().dim[i].flags  = 0;
            previousStride *= output->buffer().dim[i].extent;
        }

        return true;
    }
};

REGISTER_SHAPE(ExpandDimsComputer, OpType_ExpandDims);

} // namespace MNN
