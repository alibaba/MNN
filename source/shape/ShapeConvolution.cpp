//
//  ShapeConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class ConvolutionSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 1);
        MNN_ASSERT(1 == outputs.size());
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (format != MNN_DATA_FORMAT_NC4HW4) {
            return false;
        }
        auto layer        = op->main_as_Convolution2D()->common();
        int kernel_width  = layer->dilateX() * (layer->kernelX() - 1) + 1;
        int kernel_height = layer->dilateY() * (layer->kernelY() - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->buffer().dimensions < 4) {
            return false;
        }
        if (input->width() <= 0 || input->height() <= 0) {
            return false;
        }
        if (layer->inputCount() > 0 && input->channel() != layer->inputCount() && OpType_Convolution == op->type()) {
            MNN_ERROR("Error for compute convolution shape, need channel = %d, input channel = %d\n", layer->inputCount(), input->channel());
            return false;
        }

        if (layer->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());
            output_height = ceil((float)input->height() / (float)layer->strideY());
        } else if (layer->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            // Pad_Caffe means User setted padding
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                int input_width  = input->width() + layer->pads()->data()[1] + layer->pads()->data()[3];
                int input_height = input->height() + layer->pads()->data()[0] + layer->pads()->data()[2];
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;

        outputBuffer.dim[1].extent = layer->outputCount();
        outputBuffer.dim[2].extent = output_height;
        outputBuffer.dim[3].extent = output_width;
        outputBuffer.type = input->getType();
        //MNN_PRINT("%d, %d, %d, %d\n", outputs[0]->length(0), outputs[0]->length(1), outputs[0]->length(2), outputs[0]->length(3));

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution2D()->common();
        auto kw    = layer->kernelX();
        auto kh    = layer->kernelY();
        auto group = layer->group();
        auto ic    = inputs[0]->channel();
        auto oc    = outputs[0]->channel();
        auto oSize = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();

        auto flops = (float)oSize * kw * kh * (ic * oc / group) / FLOPS_M;
        return flops;
    }
};

class Dilation2DSizeComputer : public ConvolutionSizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() && 1 == outputs.size());
        return ConvolutionSizeComputer::onComputeSize(op, inputs, outputs);
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto output = outputs[0];
        auto layer = op->main_as_Convolution2D()->common();
        auto oSize = output->batch() * output->height() * output->width() * output->channel();
        auto flops = (float)oSize * layer->kernelY() * layer->kernelX() / FLOPS_M;
        return flops;
    }
};
class Conv2DBackpropFilterSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto common = op->main_as_Convolution2D()->common();
        auto kernel = outputs[0];
        kernel->buffer().dimensions = 4;
        kernel->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(kernel)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        kernel->setLength(0, inputs[1]->channel());
        kernel->setLength(1, inputs[0]->channel() / common->group());
        kernel->setLength(2, common->kernelY());
        kernel->setLength(3, common->kernelX());
        return true;
    }
};

REGISTER_SHAPE(ConvolutionSizeComputer, OpType_Convolution);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_ConvolutionDepthwise);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_ConvInt8);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_DepthwiseConvInt8);
REGISTER_SHAPE(Dilation2DSizeComputer, OpType_Dilation2D);
REGISTER_SHAPE(Conv2DBackpropFilterSizeComputer, OpType_Conv2DBackPropFilter);
} // namespace MNN
