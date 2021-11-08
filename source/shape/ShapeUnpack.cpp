//
//  ShapeUnpack.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class UnpackComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        if (nullptr == op || inputs.empty() || outputs.empty()) {
            // Avoid crash for special model
            return false;
        }
        auto unpack    = op->main_as_Axis();
        int axis = unpack->axis();
        if (axis < 0) {
            axis += inputs[0]->dimensions();
        }

        auto &input = inputs[0]->buffer();

        const int inputDimensions = input.dimensions;
        MNN_ASSERT(1 <= inputDimensions);
        int32_t outDims[MNN_MAX_TENSOR_DIM];
        if (outputs.size() > input.dim[axis].extent) {
            return false;
        }

        for (int i = 0; i < axis; i++) {
            outDims[i] = input.dim[i].extent;
        }
        for (int i = axis + 1; i < inputDimensions; i++) {
            outDims[i - 1] = input.dim[i].extent;
        }
        const int outputDimensions = inputDimensions - 1;
        for (int i = 0; i < outputs.size(); i++) {
            auto &output      = outputs[i]->buffer();
            output.dimensions = outputDimensions;
            output.type       = input.type;
            for (int j = 0; j < outputDimensions; j++) {
                output.dim[j].extent = outDims[j];
            }
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        }
        return true;
    }
};

REGISTER_SHAPE(UnpackComputer, OpType_Unpack);
} // namespace MNN
