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
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(4 == inputs[0]->buffer().dimensions && 4 == inputs[1]->buffer().dimensions);
        MNN_ASSERT(inputs[0]->buffer().dim[0].extent == inputs[1]->buffer().dim[0].extent);
        MNN_ASSERT(2 == inputs[1]->buffer().dim[3].extent);

        auto &ibInput0 = inputs[0]->buffer();
        auto &ibInput1 = inputs[1]->buffer();
        auto &ob = outputs[0]->buffer();

        ob.dimensions = ibInput1.dimensions;
        ob.dim[0].extent = ibInput0.dim[0].extent;
        ob.dim[1].extent = ibInput0.dim[1].extent;
        ob.dim[2].extent = ibInput1.dim[1].extent;
        ob.dim[3].extent = ibInput1.dim[2].extent;

        ob.type = ibInput0.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(
                inputs[0])->dimensionFormat;
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

REGISTER_SHAPE(GridSampleSizeComputer, OpType_GridSample);

} // namespace MNN
