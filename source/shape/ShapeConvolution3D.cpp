//
//  ShapeConvolution3D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "SizeComputer.hpp"
#include "TensorUtils.hpp"
namespace MNN {
class Convolution3DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        auto input = inputs[0];
        if (input->buffer().dimensions != 5) {
            return false;
        }
        int input_depth  = input->length(2);
        int input_height = input->length(3);
        int input_width  = input->length(4);
        if (input_depth <= 0 || input_height <= 0 || input_width <= 0) {
            return false;
        }
        
        auto layer        = op->main_as_Convolution3D()->common();
        for (auto stride: *layer->strides()) {
            MNN_ASSERT(stride == 1);
        }
        for (auto dilate: *layer->dilates()) {
            MNN_ASSERT(dilate == 1);
        }
        
        int kernel_depth  = (*layer->kernels())[0];
        int kernel_height = (*layer->kernels())[1];
        int kernel_width  = (*layer->kernels())[2];
        
        int pad_depth  = (*layer->pads())[0];
        int pad_height = (*layer->pads())[1];
        int pad_width  = (*layer->pads())[2];

        int output_depth  = input_depth + 2 * pad_depth - kernel_depth + 1;
        int output_height = input_height + 2 * pad_height - kernel_height + 1;
        int output_width  = input_width + 2 * pad_width - kernel_width + 1;

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[1].extent = layer->outputCount();
        outputBuffer.dim[2].extent = output_depth;
        outputBuffer.dim[3].extent = output_height;
        outputBuffer.dim[4].extent = output_width;
        
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
