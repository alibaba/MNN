//
//  ShapePool3D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {
class Pool3DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];
        
        for (unsigned int i = 0; i < input->buffer().dimensions; ++i) {
            MNN_ASSERT(input->buffer().dim[i].extent > 0);
        }
        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().dim[0] = input->buffer().dim[0];
        output->buffer().dim[1] = input->buffer().dim[1];

        auto layer = op->main_as_Pool3D();
        for (unsigned int i = 0; i < input->dimensions() - 2; ++i) {
            int inputLength = input->buffer().dim[i + 2].extent, outputLength = 0;
            const int kernel = (*layer->kernels())[i], stride = (*layer->strides())[i];
            
            if (layer->padType() == PoolPadType_CAFFE) {
                int pad = (*layer->pads())[i];
                outputLength = (inputLength + 2 * pad - kernel) / stride + 1;
            } else if (layer->padType() == PoolPadType_SAME) {
                outputLength = UP_DIV(inputLength, stride);
            } else if (layer->padType() == PoolPadType_VALID) {
                outputLength = (inputLength - kernel) / stride + 1;
            } else {
                MNN_ERROR("PoolPadType %d not support\n", layer->padType());
            }
            if (outputLength <= 0) {
                return false;
            }
            output->buffer().dim[i + 2].extent = outputLength;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        output->buffer().type          = input->buffer().type;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto size  = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        auto layer = op->main_as_Pool3D();
        float flopsPerElement = 1;
        for (auto kernel: *layer->kernels()) {
            flopsPerElement *= kernel;
        }
        return size * flopsPerElement;
    }
};

REGISTER_SHAPE(Pool3DSizeComputer, OpType_Pooling3D);
} // namespace MNN
