//
//  ShapePool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {
class PoolSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];

        ::memcpy(output->buffer().dim, input->buffer().dim, input->buffer().dimensions * sizeof(halide_dimension_t));
        output->buffer().dimensions = input->dimensions();

        auto layer = op->main_as_Pool();
        int outw   = 1;
        int outh   = 1;
        if (!layer->isGlobal()) {
            // when given explicit pad value in tensorflow mode pool, size compute will fast failed to help find problem
            if ((layer->padType() == PoolPadType_VALID || layer->padType() == PoolPadType_SAME) && (layer->padX() != 0 || layer->padY() != 0)) {
                MNN_PRINT("tensorflow mode pool should not have explict pad value\n");
                return false;
            }
            int w = input->width();
            int h = input->height();
            if (nullptr != layer->pads()) {
                w = w + layer->pads()->data()[1] + layer->pads()->data()[3];
                h = h + layer->pads()->data()[0] + layer->pads()->data()[2];
            } else {
                w += layer->padX() * 2;
                h += layer->padY() * 2;
            }
            int kernelWidth  = std::min(layer->kernelX(), input->width());
            int kernelHeight = std::min(layer->kernelY(), input->height());

            if (layer->padType() == PoolPadType_SAME) { // Tensorflow padding mode SAME
                outw = ceil((float)w / (float)layer->strideX());
                outh = ceil((float)h / (float)layer->strideY());
            } else if (layer->padType() == PoolPadType_VALID) { // Tensorflow padding mode VALID
                outw = ceil((float)(w - kernelWidth + 1) / (float)layer->strideX());
                outh = ceil((float)(h - kernelHeight + 1) / (float)layer->strideY());
            } else {
                if (layer->ceilModel()) {
                    outw = UP_DIV(w - kernelWidth, layer->strideX()) + 1;
                    outh = UP_DIV(h - kernelHeight, layer->strideY()) + 1;
                } else {
                    outw = floor((w - kernelWidth) / layer->strideX() + 1);
                    outh = floor((h - kernelHeight) / layer->strideY() + 1);
                }
            }
        }
        if (outw <= 0 || outh <= 0) {
            return false;
        }
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (format != MNN_DATA_FORMAT_NC4HW4) {
            return false;
        }
        output->buffer().dim[3].extent = outw;
        output->buffer().dim[2].extent = outh;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        output->buffer().type          = input->buffer().type;

        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto size  = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        auto layer = op->main_as_Pool();
        return size * layer->kernelX() * layer->kernelY();
    }
};

REGISTER_SHAPE(PoolSizeComputer, OpType_Pooling);
REGISTER_SHAPE(PoolSizeComputer, OpType_PoolInt8);
} // namespace MNN
