//
//  ShapeReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"

namespace MNN {
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
        std::vector<int> shapes;
        if (1 == inputs.size()) {
            // Const shape
            auto shape = op->main_as_Reshape()->dims();
            dimSize    = shape->size();
            shapes.resize(dimSize);
            for (int i = 0; i < dimSize; ++i) {
                shapes[i] = shape->data()[i];
            }
        } else {
            // shape which is getted at the runtime
            auto inputShape = inputs[1];
            dimSize         = inputShape->length(0);
            shapes.resize(dimSize);
            auto dim = inputShape->host<int32_t>();
            std::shared_ptr<Tensor> inputShapeCopy;
            if (dim == nullptr) {
                inputShapeCopy.reset(Tensor::createHostTensorFromDevice(inputShape, true));
                dim = inputShapeCopy.get()->host<int32_t>();
            }
            if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                //NCHW / NC4HW4
                //NHWC -> NCHW
                shapes = {dim[0], dim[3], dim[1], dim[2]};
            } else {
                for (int i = 0; i < dimSize; ++i) {
                    shapes[i] = dim[i];
                }
            }
        }

        output->buffer().dimensions = dimSize;

        int determinAxis = -1;
        for (int i = 0; i < dimSize; ++i) {
            int reshapeDim = shapes[i];
            if (reshapeDim == -1) {
                determinAxis                   = i;
                output->buffer().dim[i].extent = 1;
                continue;
            }
            if (reshapeDim == 0) {
                output->buffer().dim[i].extent = input->buffer().dim[i].extent;
                continue;
            }
            output->buffer().dim[i].extent = reshapeDim;
        }
        int totalSizeInput  = 1;
        int totalSizeOutput = 1;
        for (int i = 0; i < input->buffer().dimensions; ++i) {
            totalSizeInput *= input->buffer().dim[i].extent;
        }
        for (int i = 0; i < dimSize; ++i) {
            totalSizeOutput *= output->buffer().dim[i].extent;
        }
        if (determinAxis >= 0) {
            output->buffer().dim[determinAxis].extent = totalSizeInput / totalSizeOutput;
        }
        if (output->buffer().dimensions >= 2) {
            output->buffer().dim[1].flags = input->buffer().dim[1].flags;
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(ReshapeComputer, OpType_Reshape);
} // namespace MNN
