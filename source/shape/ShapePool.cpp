//
//  ShapePool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "Macro.h"
#include "SizeComputer.hpp"

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

        auto layer = op->main_as_Pool();
        int outw   = 1;
        int outh   = 1;
        if (!layer->isGlobal()) {
            int w = input->width();
            int h = input->height();
            if (layer->padX() > 0)
                w += layer->padX() * 2;
            if (layer->padY() > 0)
                h += layer->padY() * 2;

            if (layer->padType() == PoolPadType_SAME) { // Tensorflow padding mode SAME
                outw = ceil((float)w / (float)layer->strideX());
                outh = ceil((float)h / (float)layer->strideY());
            } else if (layer->padType() == PoolPadType_VALID) { // Tensorflow padding mode VALID
                outw = ceil((float)(w - layer->kernelX() + 1) / (float)layer->strideX());
                outh = ceil((float)(h - layer->kernelY() + 1) / (float)layer->strideY());
            } else {
                if (layer->ceilModel()) {
                    outw = UP_DIV(w - layer->kernelX(), layer->strideX()) + 1;
                    outh = UP_DIV(h - layer->kernelY(), layer->strideY()) + 1;
                } else {
                    outw = floor((w - layer->kernelX()) / layer->strideX() + 1);
                    outh = floor((h - layer->kernelY()) / layer->strideY() + 1);
                }
            }
        }
        if (outw <= 0 || outh <= 0) {
            return false;
        }
        output->buffer().dim[3].extent = outw;
        output->buffer().dim[2].extent = outh;

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
} // namespace MNN
