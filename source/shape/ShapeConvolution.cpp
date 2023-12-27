//
//  ShapeConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class ConvolutionSizeComputer : public SizeComputer {
public:
    static const Convolution2DCommon* loadCommon(const Op* op) {
        const Convolution2DCommon* layer = nullptr;
        if (op->main_type() == OpParameter_Convolution2D) {
            layer = op->main_as_Convolution2D()->common();
        } else {
            MNN_ASSERT(op->main_type() == OpParameter_TfQuantizedConv2D);
            layer = op->main_as_TfQuantizedConv2D()->common();
        }
        return layer;
    }
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 1);
        MNN_ASSERT(1 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        int kX = layer->kernelX();
        int kY = layer->kernelY();
        auto outputCount = layer->outputCount();
        if (inputs.size() > 1 && outputCount == 0) {
            // From TF's multi input convolution
            outputCount = inputs[1]->length(0);
            kX = inputs[1]->length(3);
            kY = inputs[1]->length(2);
        }
        int kernel_width  = layer->dilateX() * (kX - 1) + 1;
        int kernel_height = layer->dilateY() * (kY - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->dimensions() <= 1) {
            // Convolution is not valid for dimension <= 1
            return false;
        }

        auto inputCount = layer->inputCount();
        bool depthwiseMatch =
            inputCount == layer->outputCount() &&
            inputCount == layer->group() &&
            inputCount == input->channel();
        int commonChannelMatch =
            inputCount == inputs[0]->channel() ||            // real relationship in express
            (inputCount * layer->group() == input->channel()); // standard definition of group convolution
        bool valid = inputCount == 0 || depthwiseMatch || commonChannelMatch;

        // For Tensorflow Group Convolution, the inputCount is the size of filter's input count
        if (inputs.size() == 1 && !valid && OpType_Convolution == op->type()) {
            input->printShape();
            MNN_ERROR(
                "Error for compute convolution shape, inputCount:%d, outputCount:%d, KH:%d, KW:%d, group:%d\ninputChannel: %d, batch:%d, width:%d, height:%d. "
                "Input data channel may be mismatch with filter channel count\n",
                layer->inputCount(), outputCount, kY, kX, layer->group(),
                input->channel(), input->batch(), input->width(), input->height());
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
                output_width     = input_width < kernel_width ? 0 : (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = input_height < kernel_height ? 0 : (input_height - kernel_height) / layer->strideY() + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->symmetricQuan() && op->main_as_Convolution2D()->symmetricQuan()->outputDataType() != DataType_DT_INT8) {
            auto type = op->main_as_Convolution2D()->symmetricQuan()->outputDataType();
            outputs[0]->setType(type);
        }
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = outputCount;
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        } else {
            outputBuffer.dim[1].extent = outputCount;
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        // MNN_PRINT("outputs: %d, %d, %d, %d\n", outputs[0]->length(0), outputs[0]->length(1), outputs[0]->length(2), outputs[0]->length(3));
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        const Convolution2DCommon* layer = loadCommon(op);
        auto kw    = layer->kernelX();
        auto kh    = layer->kernelY();
        auto group = layer->group();
        auto ic    = inputs[0]->channel();
        auto oc    = outputs[0]->channel();
        auto oSize = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
        if (op->type() == OpType_QuantizedDepthwiseConv2D) {
            group = ic;
        }
        if (layer->inputCount() != ic && layer->inputCount() > 0) {
            group = ic / layer->inputCount();
        }
        auto flops = (float)oSize * kw * kh * (ic * oc / (group == 0 ? 1 : group)) / FLOPS_M;
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
class Im2ColSizeComputer : public ConvolutionSizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() && 1 == outputs.size());
        // get kh, kw
        const Convolution2DCommon* layer = loadCommon(op);
        auto kh    = layer->kernelY();
        auto kw    = layer->kernelX();
        // get oh, ow
        ConvolutionSizeComputer::onComputeSize(op, inputs, outputs);
        auto output = outputs[0];
        int oh = output->height();
        int ow = output->width();
        // [n, ic, ih, iw] -> [ic*kh*kw, n*oh*ow]
        auto input = inputs[0];
        int n = input->batch();
        int ic = input->channel();
        int ih = input->height();
        int iw = input->width();
        output->buffer().dimensions = 2;
        output->setLength(0, ic * kh * kw);
        output->setLength(1, n * oh * ow);
        return true;
    }
};

class Col2ImSizeComputer : public ConvolutionSizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size() && 1 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        auto kh    = layer->kernelY();
        auto kw    = layer->kernelX();
        auto input = inputs[0];
        auto output = outputs[0];
        auto outputShape = inputs[1];
        auto oDim = outputShape->host<int32_t>();
        int oh = 1, ow = 1;
        if (outputShape->elementSize() == 2) {
            oh = oDim[0];
            ow = oDim[1];
        } else {
            MNN_ASSERT(false);
        }
        auto iDim = input->shape();
        int batch = 1;
        int colSize = iDim[0];
        if (iDim.size() == 3) {
            batch = iDim[0];
            colSize = iDim[1];
        } else if (iDim.size() == 2) {
            colSize = iDim[0];
        } else {
            MNN_ASSERT(false);
        }
        output->buffer().dimensions = 4;
        output->setLength(0, batch);
        output->setLength(1, colSize / (kh * kw));
        output->setLength(2, oh);
        output->setLength(3, ow);
        return true;
    }
};

REGISTER_SHAPE(ConvolutionSizeComputer, OpType_Convolution);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_ConvolutionDepthwise);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_TfQuantizedConv2D);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_QuantizedDepthwiseConv2D);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_ConvInt8);
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_DepthwiseConvInt8);
REGISTER_SHAPE(Dilation2DSizeComputer, OpType_Dilation2D);
REGISTER_SHAPE(Conv2DBackpropFilterSizeComputer, OpType_Conv2DBackPropFilter);
REGISTER_SHAPE(Im2ColSizeComputer, OpType_Im2Col);
REGISTER_SHAPE_INPUTS(Col2ImSizeComputer, OpType_Col2Im, {1});
} // namespace MNN
