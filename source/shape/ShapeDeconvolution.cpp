//
//  ShapeDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
namespace MNN {

class DeconvolutionSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution2D()->common();

        auto inputTensor = inputs[0];
        int outputHeight = 0, outputWidth = 0;
        if (layer->hasOutputShape()) {
            MNN_ASSERT(inputs.size() >= 2);
            auto outputShape = inputs.back();
            outputHeight = outputShape->host<int>()[1];
            outputWidth  = outputShape->host<int>()[2];
        }

        int input_width   = inputTensor->width();
        int input_height  = inputTensor->height();
        int sH            = layer->strideY();
        int sW            = layer->strideX();
        int kH            = layer->kernelY();
        int kW            = layer->kernelX();
        int pH            = layer->padY();
        int pW            = layer->padX();
        int dH            = layer->dilateY();
        int dW            = layer->dilateX();
        int output_width;
        int output_height;
        auto format = TensorUtils::getDescribe(inputTensor)->dimensionFormat;

        if (outputHeight > 0 && outputWidth > 0) {
            output_width = outputWidth;
            output_height = outputHeight;
        } else if (layer->padMode() == PadMode_SAME) { // Tensorflow support
            output_width  = input_width * sW;
            output_height = input_height * sH;
        } else {
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                output_width  = (input_width - 1) * sW + dW * (kW - 1) + 1 - layer->pads()->data()[1] - layer->pads()->data()[3];
                output_height = (input_height - 1) * sH + dH * (kH - 1) + 1 - layer->pads()->data()[0] - layer->pads()->data()[2];
            } else {
                output_width  = (input_width - 1) * sW + dW * (kW - 1) + 1 - pW * 2;
                output_height = (input_height - 1) * sH + dH * (kH - 1) + 1 - pH * 2;
            }
            if(nullptr != layer->outPads()) {
                output_width  += layer->outPads()->data()[1];
                output_height += layer->outPads()->data()[0];
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.type = inputTensor->getType();
        outputBuffer.dimensions    = inputTensor->buffer().dimensions;
        outputBuffer.dim[0].extent = inputTensor->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = op->main_as_Convolution2D()->common()->outputCount();
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        } else {
            outputBuffer.dim[1].extent = op->main_as_Convolution2D()->common()->outputCount();
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;

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
        auto oSize = inputs[0]->width() * inputs[0]->height() * inputs[0]->batch();

        return (float)oSize * kw * kh * (ic * oc / group) / FLOPS_M;
    }
};

REGISTER_SHAPE(DeconvolutionSizeComputer, OpType_Deconvolution);
REGISTER_SHAPE(DeconvolutionSizeComputer, OpType_DeconvolutionDepthwise);
} // namespace MNN
