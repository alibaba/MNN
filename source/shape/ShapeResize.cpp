//
//  ShapeResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
// Size Computer
class ResizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        // copy dims
        auto resize  = op->main_as_Resize();
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        TensorUtils::copyShape(inputs[0], outputs[0], true);

        // set dims
        output.dim[3].extent = input.dim[3].extent * resize->xScale();
        output.dim[2].extent = input.dim[2].extent * resize->yScale();
        output.type = inputs[0]->getType();

        return true;
    }
    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f * 4;
    }
};

class ImageProcessComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(1 == inputs.size() || inputs.size() == 3);
        MNN_ASSERT(1 == outputs.size());
        if (inputs.size() == 3) {
            auto &output = outputs[0]->buffer();
            output.dimensions = 1;
            output.dim[0].extent = 1;
            return true;
        }

        // copy dims
        auto &input  = inputs[0]->buffer();
        auto &output = outputs[0]->buffer();
        TensorUtils::copyShape(inputs[0], outputs[0], true);

        // set dims
        auto process  = op->main_as_ImageProcessParam();
        int c = process->shape()->Get(1);
        int h = process->shape()->Get(2);
        int w = process->shape()->Get(3);
        if (MNN_DATA_FORMAT_NHWC == TensorUtils::getDescribe(inputs[0])->dimensionFormat) {
            output.dim[1].extent = h;
            output.dim[2].extent = w;
            output.dim[3].extent = c;
        } else {
            output.dim[1].extent = c;
            output.dim[2].extent = h;
            output.dim[3].extent = w;
        }
        // set dtype
        outputs[0]->setType(process->outputType());
        return true;
    }
    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f * 4;
    }
};

REGISTER_SHAPE(ResizeComputer, OpType_Resize);
REGISTER_SHAPE(ImageProcessComputer, OpType_ImageProcess);
} // namespace MNN
