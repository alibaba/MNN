//
//  ShapeGatherV2.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class GatherV2Computer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto params  = inputs[0];
        auto indices = inputs[1];
        if (indices->getType().code != halide_type_int) {
            return false;
        }
        int axis = 0;
        if (inputs.size() == 3) {
            auto axis_tensor = inputs[2];
            axis = axis_tensor->host<int32_t>()[0];
        }
        if (op->main_type() == OpParameter_Axis) {
            axis = op->main_as_Axis()->axis();
        }
        if( axis <= -params->buffer().dimensions || axis >= params->buffer().dimensions) {
            return false;
        }

        if (axis < 0) {
            axis = params->buffer().dimensions + axis;
        }

        const int gather_dim_size = params->buffer().dim[axis].extent;
        MNN_ASSERT(gather_dim_size <= std::numeric_limits<int32_t>::max());

        const int numDimensions = params->buffer().dimensions + indices->buffer().dimensions - 1;
        MNN_ASSERT(axis <= numDimensions);

        std::vector<int> result_shape;

        for (int i = 0; i < axis; i++) {
            result_shape.push_back(params->buffer().dim[i].extent);
        }

        for (int i = 0; i < indices->buffer().dimensions; i++) {
            result_shape.push_back(indices->buffer().dim[i].extent);
        }

        for (int i = axis + 1; i < params->buffer().dimensions; i++) {
            result_shape.push_back(params->buffer().dim[i].extent);
        }

        outputs[0]->buffer().dimensions = (int)result_shape.size();
        outputs[0]->buffer().type       = params->buffer().type;
        for (int i = 0; i < result_shape.size(); i++) {
            outputs[0]->buffer().dim[i].extent = result_shape.at(i);
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(GatherV2Computer, OpType_GatherV2, (std::vector<int>{2}));
REGISTER_SHAPE(GatherV2Computer, OpType_Gather);
} // namespace MNN
