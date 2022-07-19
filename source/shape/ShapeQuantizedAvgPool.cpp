//
//  ShapeQuantizedAvgPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <math.h>
#include "shape/SizeComputer.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include "core/Macro.h"

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
        outputs[0]->setType(DataType_DT_UINT8);
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[2].extent = output_height;
        outputBuffer.dim[3].extent = output_width;
        outputBuffer.dim[1].extent = input->buffer().dim[1].extent;
        if (format == MNN_DATA_FORMAT_NHWC) {
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
            outputBuffer.dim[3].extent = input->channel();
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;

        return true;
    }
};

} // namespace MNN
#endif
namespace MNN {
REGISTER_SHAPE_OLD(QuantizedAvgPoolComputer, OpType_QuantizedAvgPool);
};
