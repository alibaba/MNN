//
//  ShapeGridSample.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class GridSampleSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        // https://pytorch.org/docs/1.7.1/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample
        // inputs[0] is input, inputs[1] is grid
        MNN_ASSERT(2 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto &ibInput0 = inputs[0]->buffer();
        auto &ob = outputs[0]->buffer();
        ob.type = ibInput0.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(
                inputs[0])->dimensionFormat;
        if (inputs.size() > 2) {
            // For Grad, just copy the shape
            ob.dimensions = inputs[2]->length(0);
            auto shapePtr = inputs[2]->host<int>();
            for (int i=0; i<ob.dimensions; ++i) {
                ob.dim[i].extent = shapePtr[i];
            }
            return true;
        }

        int input_dim = inputs[0]->buffer().dimensions;
        int grid_dim = inputs[1]->buffer().dimensions;
        MNN_ASSERT((4 == input_dim && 4 == grid_dim) || (5 == input_dim && 5 == grid_dim));
        if (inputs[0]->buffer().dim[0].extent != inputs[1]->buffer().dim[0].extent) {
            return false;
        }
        MNN_ASSERT(grid_dim - 2 == inputs[1]->buffer().dim[grid_dim - 1].extent);

        auto &ibInput1 = inputs[1]->buffer();

        ob.dimensions = ibInput1.dimensions;
        ob.dim[0].extent = ibInput0.dim[0].extent;
        ob.dim[1].extent = ibInput0.dim[1].extent;
        ob.dim[2].extent = ibInput1.dim[1].extent;
        ob.dim[3].extent = ibInput1.dim[2].extent;
        if (grid_dim == 5) {
            ob.dim[4].extent = ibInput1.dim[3].extent;
        }

        return true;
    }

    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        auto gridSampleParam = op->main_as_GridSample();
        if (gridSampleParam->mode() == MNN::SampleMode_BILINEAR) {
            return 4 * SizeComputer::onComputeFlops(op, inputs, outputs);
        }

        return SizeComputer::onComputeFlops(op, inputs, outputs);
    }
};

REGISTER_SHAPE_INPUTS(GridSampleSizeComputer, OpType_GridSample, {2});

} // namespace MNN
