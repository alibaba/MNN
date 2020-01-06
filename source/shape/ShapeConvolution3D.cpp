//
//  ShapeConvolution3D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class Convolution3DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        auto layer        = op->main_as_Convolution3D()->common();
        for (auto stride: *layer->strides()) {
            MNN_ASSERT(stride == 1);
        }
        for (auto dilate: *layer->dilates()) {
            MNN_ASSERT(dilate == 1);
        }
        
        auto input = inputs[0];
        if (input->buffer().dimensions != 5) {
            return false;
        }
        
        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[1].extent = layer->outputCount();
        
        for (int i = 0; i < 3; ++i) {
            const int inputLength = input->length(i + 2), stride = (*layer->strides())[i];
            if (inputLength <= 0) {
                return false;
            }
            int outputLength;
            if (layer->padMode() == PadMode_SAME) {
                outputLength = UP_DIV(inputLength, stride);
            } else {
                const int pad = (*layer->pads())[i], kernel = (*layer->kernels())[i], dialate = (*layer->dilates())[i];
                const int dialatedKernel = (kernel - 1) * dialate + 1;
                outputLength = (inputLength + 2 * pad - dialatedKernel) / stride + 1;
            }
            outputBuffer.dim[i + 2].extent = outputLength;
        }
        
        outputBuffer.type = input->getType();

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution3D()->common();
        int oSize = outputs[0]->length(1);
        float flopsPerElement = inputs[0]->length(1);
        for (int i = 0; i < 3; ++i) {
            flopsPerElement *= (*layer->kernels())[i];
            oSize *= outputs[0]->length(i + 2);
        }
        float flops = oSize * flopsPerElement / FLOPS_M;

        return flops;
    }
};

REGISTER_SHAPE(Convolution3DSizeComputer, OpType_Convolution3D);
} // namespace MNN
