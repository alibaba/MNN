//
//  ShapeReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
class FlattenComputer : public SizeComputer {
public:
    // Ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto flatten = op->main_as_Flatten();
        if (nullptr == flatten || inputs.empty() || outputs.empty()) {
            return false;
        }
        auto axis = flatten->axis();
        auto endAxis = flatten->endAxis();
        auto dim = inputs[0]->dimensions();
        if (axis < 0) {
            axis += dim;
        }
        if (endAxis < 0) {
            endAxis += dim;
        }
        int inside = 1;
        int middle = 1;
        int outside = 1;
        if (endAxis == 0) {
            for (int i=0; i<axis; ++i) {
                outside *= inputs[0]->length(i);
            }
            for (int i=axis; i<dim; ++i) {
                inside *= inputs[0]->length(i);
            }
            outputs[0]->buffer().dimensions = 2;
            outputs[0]->setLength(0, outside);
            outputs[0]->setLength(1, inside);
        } else {
            // [ 0 - axis, 1, endAxis - lastDim]
            outputs[0]->buffer().dimensions = dim - endAxis + axis;
            for (int i = 0; i < axis; ++i) {
                outputs[0]->setLength(i, inputs[0]->length(i));
            }
            for (int i = axis; i <= endAxis; ++i) {
                outside *= inputs[0]->length(i);
            }
            outputs[0]->setLength(axis, outside);
            if (dim > endAxis + 1) {
                for (int i = endAxis + 1; i < dim; ++i) {
                    outputs[0]->setLength(i, inputs[0]->length(i));
                }
            }
        }
        outputs[0]->buffer().type = inputs[0]->getType();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};
class ReshapeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input                = inputs[0];
        auto output               = outputs[0];
        outputs[0]->buffer().type = inputs[0]->buffer().type;
        int dimSize               = 0;
        int shapes[MNN_MAX_TENSOR_DIM];
        auto inputFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        bool fromTf = false;
        auto mainType = op->main_type();
        if (1 == inputs.size()) {
            // Const shape
            if (OpParameter_Reshape == mainType) {
                auto shape = op->main_as_Reshape()->dims();
                dimSize    = shape->size();
                for (int i = 0; i < dimSize; ++i) {
                    shapes[i] = shape->data()[i];
                }
            } else {
                // For old model compability
                auto shape = op->main_as_QuantizedReshape()->dims();
                dimSize    = shape->size();
                for (int i = 0; i < dimSize; ++i) {
                    shapes[i] = shape->data()[i];
                }
            }
        } else {
            // shape which is getted at the runtime
            auto inputShape = inputs[1];
            // For the model convert from tensorflow, the format is NHWC, otherwise NCHW
            fromTf          = TensorUtils::getDescribe(inputShape)->dimensionFormat == MNN_DATA_FORMAT_NHWC;
            dimSize         = inputShape->elementSize();
            auto dim = inputShape->host<int32_t>();
            auto dimType = MNN_DATA_FORMAT_NHWC;
            if (OpParameter_Reshape == mainType) {
                dimType = op->main_as_Reshape()->dimType();
            }
            if ((inputFormat == MNN_DATA_FORMAT_NC4HW4) && dimType == MNN_DATA_FORMAT_NHWC) {
                //NCHW / NC4HW4
                //NHWC -> NCHW
                shapes[0] = dim[0];
                shapes[1] = dim[3];
                shapes[2] = dim[1];
                shapes[3] = dim[2];
            } else {
                for (int i = 0; i < dimSize; ++i) {
                    shapes[i] = dim[i];
                }
            }
        }
        output->buffer().dimensions = dimSize;

        int totalSizeInput  = 1;
        for (int i = 0; i < input->buffer().dimensions; ++i) {
            auto l = input->length(i);
            totalSizeInput *= l;
        }

        int determinAxis = -1;
        for (int i = 0; i < dimSize; ++i) {
            int reshapeDim = shapes[i];
            if (reshapeDim == -1) {
                determinAxis                   = i;
                output->buffer().dim[i].extent = 1;
                continue;
            }
            // Keep input dimension if reshape dimension is 0 and the element
            // count of the input does not equal to 0.
            // TODO: Reshape 0 is not allowed if the input element count is not
            // 0 for TensorFlow.
            if (reshapeDim == 0 && (!fromTf)) {
                output->buffer().dim[i].extent = input->buffer().dim[i].extent;
            } else {
                output->buffer().dim[i].extent = reshapeDim;
            }
        }
        int totalSizeOutput = 1;
        for (int i = 0; i < dimSize; ++i) {
            totalSizeOutput *= output->buffer().dim[i].extent;
        }
        if (determinAxis >= 0) {
            output->buffer().dim[determinAxis].extent = totalSizeOutput ? totalSizeInput / totalSizeOutput : 0;
            totalSizeOutput *= output->buffer().dim[determinAxis].extent;
        }
        if (totalSizeInput != totalSizeOutput) {
            MNN_PRINT("Reshape error: %d -> %d\n", totalSizeInput, totalSizeOutput);
            return false;
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(ReshapeComputer, OpType_Reshape, {1});
REGISTER_SHAPE_INPUTS(ReshapeComputer, OpType_QuantizedReshape, {1});

REGISTER_SHAPE(FlattenComputer, OpType_Flatten);

} // namespace MNN
