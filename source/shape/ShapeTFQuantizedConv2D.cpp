//
//  ShapeTFQuantizedConv2D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class TFQuantizedConv2DComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_TfQuantizedConv2D()->common();

        MNN_ASSERT(layer->dilateX() == 1);
        MNN_ASSERT(layer->dilateY() == 1);
        MNN_ASSERT(layer->strideX() == layer->strideY());

        int kernel_width  = layer->dilateX() * (layer->kernelX() - 1) + 1;
        int kernel_height = layer->dilateY() * (layer->kernelY() - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];

        if (layer->padMode() == PadMode_SAME) {                                     // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());  // NHWC for tensorflow
            output_height = ceil((float)input->height() / (float)layer->strideY()); // the default layout is NCHW
        } else if (layer->padMode() == PadMode_VALID) {                             // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            MNN_ASSERT(false); // unsupported type
        }

        // output：NCHW
        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[1].extent = layer->outputCount();
        outputBuffer.dim[2].extent = output_height;
        outputBuffer.dim[3].extent = output_width;

        outputs[0]->buffer().type = halide_type_of<uint8_t>();

        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_TfQuantizedConv2D()->common();
        auto kw    = layer->kernelX();
        auto kh    = layer->kernelY();
        int group  = 1;
        if (op->type() == OpType_QuantizedDepthwiseConv2D) {
            group = inputs[0]->channel();
        }
        auto ic    = inputs[0]->channel();
        auto oc    = outputs[0]->channel();
        auto oSize = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();

        return (float)oSize * kw * kh * (ic * oc / group) / FLOPS_M;
    }
};

REGISTER_SHAPE(TFQuantizedConv2DComputer, OpType_TfQuantizedConv2D);
REGISTER_SHAPE(TFQuantizedConv2DComputer, OpType_QuantizedDepthwiseConv2D);
} // namespace MNN
