//
//  ShapeBroadcastTo.cpp
//  MNN
//
//  Created by MNN on 2019/12/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class ShapeBroadcastTo : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);
        auto input  = inputs[0];
        auto shape  = inputs[1];
        auto output = outputs[0];
        int inputDims = input->dimensions();
        int shapeDims = shape->elementSize();
        output->buffer().dimensions = inputDims > shapeDims ? inputDims : shapeDims;
        const int dimension = output->dimensions();
        const int* shapeData        = shape->host<int>();
        if (op->main() && op->main_as_Axis()->axis()) {
            for (int i = 0; i < dimension; i++) {
                output->setLength(i, shapeData[i]);
            }
        } else {
            for (int i = 1; i <= dimension; ++i) {
                int inputDim = 1, shapeDim = 1;
                if (i <= inputDims) {
                    inputDim = input->length(inputDims - i);
                }
                if (i <= shapeDims) {
                    shapeDim = shapeData[shapeDims - i];
                }
                if (shapeDim <= 1) {
                    // shapeDim is {-1,0,1}, keep inputDim
                    output->setLength(dimension - i, inputDim);
                } else {
                    // broadcast inputDim to shapeDim, need shapDim % inputDim == 0
                    // inputDim == 0, need shapeDim <= 0 keep dim
                    MNN_ASSERT(inputDim != 0);
                    MNN_ASSERT(shapeDim % inputDim == 0);
                    output->setLength(dimension - i, shapeDim);
                }
            }
        }
        output->buffer().type                             = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_INPUTS(ShapeBroadcastTo, OpType_BroadcastTo, {1});

} // namespace MNN
