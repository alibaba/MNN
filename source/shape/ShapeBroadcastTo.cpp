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
            int offset;
            int alignShape[MNN_MAX_TENSOR_DIM];
            if (inputDims > shapeDims) {
                for (int i = 0; i < input->dimensions(); ++i) {
                    output->setLength(i, input->length(i));
                }
                offset = inputDims - shapeDims;
                for (int i=0; i<shapeDims; ++i) {
                    alignShape[i] = shapeData[i];
                }
            } else {
                for (int i = 0; i < shapeDims; ++i) {
                    output->setLength(i, shapeData[i]);
                }
                for (int i=0; i<input->dimensions(); ++i) {
                    alignShape[i] = input->length(i);
                }
                offset = shapeDims - inputDims;
            }
            for (int i = offset; i < output->dimensions(); ++i) {
                int dim1 = alignShape[i - offset];
                int dim2 = output->length(i);
                if (dim1 != dim2 && (dim1 != 1 && dim2 != 1)) {
                    MNN_ERROR("Broad cast error, dim1 = %d, dim2 = %d\n", dim1, dim2);
                    return false;
                }
                if (dim1 == dim2) {
                    continue;
                }
                if (dim1 != 1) {
                    output->setLength(i, dim1);
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
