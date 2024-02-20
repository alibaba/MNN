//
//  ShapePool3D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class Pool3DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto input  = inputs[0];
        auto output = outputs[0];
        
        auto layer = op->main_as_Pool3D();
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        // only check channel dim when global pool
        int maxCheckDim = (layer->isGlobal() ? 1 :input->buffer().dimensions - 1);
        for (unsigned int i = 1; i <= maxCheckDim; ++i) {
            if (input->buffer().dim[i].extent <= 0) {
                return false;
            }
        }
        output->buffer().dimensions = input->buffer().dimensions;
        ::memcpy(output->buffer().dim, input->buffer().dim, input->buffer().dimensions * sizeof(halide_dimension_t));

        if (layer->isGlobal()) {
            if (format == MNN_DATA_FORMAT_NHWC) {
                // N [1...] C
                for (int d = 1; d < output->dimensions() - 1; d++) {
                    output->buffer().dim[d].extent = 1;
                }
            } else {
                // N C [1...]
                for (int d = 2; d < output->dimensions(); d++) {
                    output->buffer().dim[d].extent = 1;
                }
            }
        } else {
            int offset = format == MNN_DATA_FORMAT_NHWC ? 1 : 2;
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
                output->buffer().dim[i + offset].extent = outputLength;
            }
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
        if (layer->kernels() == nullptr) {
            return size * flopsPerElement;
        }
        for (auto kernel: *layer->kernels()) {
            flopsPerElement *= kernel;
        }
        return size * flopsPerElement;
    }
};

REGISTER_SHAPE(Pool3DSizeComputer, OpType_Pooling3D);
} // namespace MNN
