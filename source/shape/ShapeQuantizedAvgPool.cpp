//
//  ShapeQuantizedAvgPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "Macro.h"
#include "SizeComputer.hpp"

namespace MNN {
class QuantizedAvgPoolComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_QuantizedAvgPool();

        MNN_ASSERT(layer->strideX() == layer->strideY());

        int kernel_width  = layer->kernelX();
        int kernel_height = layer->kernelY();

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];

        if (layer->padType() == PoolPadType_SAME) {                                   // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());  // NHWC for tensorflow
            output_height = ceil((float)input->height() / (float)layer->strideY()); // the default layout is NCHW
        } else if (layer->padType() == PoolPadType_VALID) {                           // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            MNN_ASSERT(false); // unsupported type
        }
        
        // output：NHWC MNN: nchw
        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        
        outputBuffer.dim[2].extent = output_height;
        outputBuffer.dim[3].extent = output_width;
        outputBuffer.dim[1].extent = input->buffer().dim[1].extent;
        
        if (3 == inputs.size()) {
            auto output_min          = outputs[1]->buffer();
            output_min.dimensions    = 0;
            output_min.dim[0].extent = output_min.dim[1].extent = output_min.dim[2].extent = output_min.dim[3].extent =
            1;
            
            auto output_max          = outputs[2]->buffer();
            output_max.dimensions    = 0;
            output_max.dim[0].extent = output_max.dim[1].extent = output_max.dim[2].extent = output_max.dim[3].extent =
            1;
            outputs[0]->setType(DataType_DT_INT32);
        } else {
            outputs[0]->setType(DataType_DT_UINT8);
        }
        
        return true;
    }
};

REGISTER_SHAPE(QuantizedAvgPoolComputer, OpType_QuantizedAvgPool);
} // namespace MNN
