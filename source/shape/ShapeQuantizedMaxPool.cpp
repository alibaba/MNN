//
//  ShapeQuantizedMaxPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "shape/SizeComputer.hpp"
#ifdef MNN_SUPPORT_DEPRECATED_OP
#include <math.h>
#include "core/Macro.h"

namespace MNN {
class QuantizedMaxPoolComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_QuantizedMaxPool();

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

        if (output_width <= 0 || output_height <= 0) {
            return false;
        }

        // max pool use nhwc
        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;

        outputBuffer.dim[1].extent = output_height;
        outputBuffer.dim[2].extent = output_width;
        outputBuffer.dim[3].extent = input->buffer().dim[3].extent;
        outputs[0]->setType(DataType_DT_UINT8);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

} // namespace MNN
#endif

namespace MNN {
REGISTER_SHAPE_OLD(QuantizedMaxPoolComputer, OpType_QuantizedMaxPool);
};
