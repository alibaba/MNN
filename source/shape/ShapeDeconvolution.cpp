//
//  ShapeDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"
#include "TensorUtils.hpp"
namespace MNN {

class DeconvolutionSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution2D()->common();

        auto inputTensor = inputs[0];

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
        int output_width  = (input_width - 1) * sW + dW * (kW - 1) + 1 - pW * 2;
        int output_height = (input_height - 1) * sH + dH * (kH - 1) + 1 - pH * 2;

        if (layer->padMode() == PadMode_SAME) { // Tensorflow support
            output_width  = input_width * sW;
            output_height = input_height * sH;
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = inputTensor->buffer().dimensions;
        outputBuffer.dim[0].extent = inputTensor->buffer().dim[0].extent;

        outputBuffer.dim[1].extent = op->main_as_Convolution2D()->common()->outputCount();
        outputBuffer.dim[2].extent = output_height;
        outputBuffer.dim[3].extent = output_width;
        outputBuffer.dim[1].flags = Tensor::REORDER_4;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

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
