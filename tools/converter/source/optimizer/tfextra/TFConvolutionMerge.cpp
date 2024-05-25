//
//  TFConvolutionMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "MNN_generated.h"
#include "TFExtraManager.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace Express {
static bool _writeCommonAttr(Convolution2DCommonT* common, const Extra* extra, const std::string& name) {
    if (nullptr == extra || nullptr == extra->attr()) {
        return false;
    }
    auto attrSize = extra->attr()->size();
    for (int v = 0; v < attrSize; ++v) {
        auto attr      = extra->attr()->GetAs<Attribute>(v);
        const auto key = attr->key()->str();
        auto list      = attr->list();
        // "rates" for tf.nn.atrous_conv2d
        // "dilations" for tf.nn.conv2d or tf.nn.dilation2d or tf.nn.conv2d_transpose
        // "rate" has been here when I change the code, so I reserve it though I don't know where use it
        if (key == "rate" || key == "rates" || key == "dilations") {
            common->dilateX = list->i()->data()[2];
            common->dilateY = list->i()->data()[1];
        } else if (key == "strides") {
            common->strideX = list->i()->data()[2];
            common->strideY = list->i()->data()[1];
        } else if (key == "padding") {
            common->padMode  = MNN::PadMode_SAME;
            auto paddingType = attr->s()->str();
            if (paddingType == "VALID") {
                common->padMode = MNN::PadMode_VALID;
            } else if (paddingType == "Symmetric") {
                common->padMode = MNN::PadMode_CAFFE;
                common->padX    = 1;
                common->padY    = 1;
            }
        }
    }
    return true;
}
class ConvolutionTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op               = expr->get();
        auto inputs           = expr->inputs();
        auto weight           = inputs[1];
        auto weightInfo       = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common = convolution2D->common.get();
        common->relu        = false;
        common->group       = 1;
        common->padX        = 0;
        common->padY        = 0;
        common->outputCount = 0;

        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }
        if (!weightInfo || !weightTensorData) {
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->name       = expr->name();
            newOp->type       = OpType_Convolution;
            newOp->main.type  = OpParameter_Convolution2D;
            newOp->main.value = convolution2D.release();
            // Turn weight to NCHW
            inputs[1]         = _Transpose(inputs[1], {3, 2, 0, 1});
            auto newExpr      = Expr::create(newOp.get(), inputs, 1);
            return newExpr;
        }
        int kh           = weightInfo->dim[0];
        int kw           = weightInfo->dim[1];
        int num_input    = weightInfo->dim[2];
        int weight_input = weightInfo->dim[2];
        common->kernelX     = kw;
        common->kernelY     = kh;
        auto src = inputs[0];
        auto srcInfo = src->getInfo();
        if (nullptr != srcInfo && srcInfo->dim.size() > 0) {
            if (NHWC == srcInfo->order) {
                num_input = srcInfo->dim[(int)srcInfo->dim.size() - 1];
            } else {
                num_input = srcInfo->dim[1];
            }
        }
        int num_output   = weightInfo->dim[3];
        common->outputCount = num_output;
        common->inputCount  = num_input;
        if (0 != weight_input) {
            common->group   = num_input / weight_input;
        }
        if (common->group < 1) {
            common->group = 1;
        }
        weight           = _Transpose(weight, {3, 2, 0, 1});
        weightInfo       = weight->getInfo();
        weightTensorData = weight->readMap<float>();

        convolution2D->bias.resize(num_output);
        std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);
        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = expr->name();
        newOp->type       = OpType_Convolution;
        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();
        auto newExpr      = Expr::create(newOp.get(), {inputs[0]}, 1);
        return newExpr;
    }
};

class ConvolutionDepthwiseTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op               = expr->get();
        auto inputs           = expr->inputs();
        auto input            = inputs[0];
        auto weight           = inputs[1];
        auto weightInfo       = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (!weightInfo || !weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }

        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        int kh           = weightInfo->dim[0];
        int kw           = weightInfo->dim[1];
        int num_input    = weightInfo->dim[2];
        int multiplier   = weightInfo->dim[3];
        int num_output   = num_input * multiplier;
        weight           = _Transpose(weight, {3, 2, 0, 1});
        if (multiplier <= 1) {
            weightInfo       = weight->getInfo();
            weightTensorData = weight->readMap<float>();
            int once_weight  = weightInfo->size / multiplier;
            convolution2D->weight.resize(once_weight);
            ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
            convolution2D->bias.resize(num_output);
            std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);
        }
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common = convolution2D->common.get();

        common->relu        = false;
        common->group       = num_input;
        common->outputCount = num_input;
        common->inputCount  = num_input;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->padX        = 0;
        common->padY        = 0;

        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = expr->name();
        newOp->type       = OpType_ConvolutionDepthwise;
        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();
        if (multiplier <= 1) {
            return (Expr::create(newOp.get(), {inputs[0]}, 1));
        }
        std::vector<int> split(multiplier, 1);
        auto weights = _Split(weight, split);
        std::vector<VARP> convs(multiplier);
        for (int i = 0; i < multiplier; i++) {
            convs[i] = (Variable::create(Expr::create(newOp.get(), {inputs[0], weights[i]})));
        }
        // NHWC => NMHWC (Raster: NCHW => NMCHW)
        auto x = _Concat(convs, 1);
        // NMHWC => NMAC (Raster: NMCHW => NMCA)
        auto shape = _Split(_Shape(convs[0]), {1, 1, 1, 1}, 0);
        auto batch_n  = shape[0];
        auto kernel_h = shape[1];
        auto kernel_w = shape[2];
        auto input_c  = shape[3];
        auto multip   = _Const(&multiplier, {1}, NHWC, halide_type_of<int>());
        x = _Reshape(x, _Concat({batch_n, multip, _Multiply(kernel_h, kernel_w), input_c}, 0));
        // NMAC => NACM (Raster: NMCA => NCMA)
        x = _Transpose(x, {0, 2, 3, 1});
        auto outputShape = _Concat({batch_n, kernel_h, kernel_w, _Multiply(input_c, multip)}, 0);
        // NACM => NHWC (NCMA => NCHW)
        std::unique_ptr<OpT> reshape(new OpT);
        reshape->type                      = OpType_Reshape;
        reshape->name                      = expr->name() + "_Reshape";
        reshape->main.type                 = OpParameter_Reshape;
        reshape->main.value                = new ReshapeT;
        reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NHWC;
        return (Expr::create(reshape.get(), {x, outputShape}));
    }
};

class DeconvolutionTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op        = expr->get();
        bool depthwise = false;
        {
            std::unique_ptr<ExtraT> extraT(op->main_as_Extra()->UnPack());
            if (extraT->type == "DepthwiseConv2dNativeBackpropInput") {
                depthwise = true;
            }
        }
        auto inputs           = expr->inputs();
        auto weight           = inputs[1];
        auto weightInfo       = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        int kh           = weightInfo->dim[0];
        int kw           = weightInfo->dim[1];
        int num_input    = weightInfo->dim[2];
        int num_output   = weightInfo->dim[3];
        weight           = _Transpose(weight, {3, 2, 0, 1});
        weightInfo       = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        convolution2D->bias.resize(num_input);
        std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common = convolution2D->common.get();

        common->relu        = false;
        common->group       = 1;
        common->outputCount = num_input;
        common->inputCount  = num_output;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->padX        = 0;
        common->padY        = 0;
        bool success        = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name = expr->name();
        newOp->type = OpType_Deconvolution;
        if (depthwise) {
            newOp->type = OpType_DeconvolutionDepthwise;
        }
        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();
        if (inputs.size() == 2) {
            return Expr::create(newOp.get(), {inputs[0]}, 1);
        }
        MNN_ASSERT(inputs.size() == 3);
        auto newExpr = Expr::create(newOp.get(), {inputs[2]}, 1);
        /* check shape consistent between tf's output_shape attribute and MNN inferred output shape
         * When stride > 1, one output-shape can be reached from (stride - 1) input-shapes
         */
        auto output          = Variable::create(newExpr);
        auto outputInfo      = output->getInfo();
        auto realOutputShape = inputs[0]->readMap<int>();
        if (nullptr != outputInfo && nullptr != realOutputShape) {
            int inferHeight = outputInfo->dim[2], inferWidth = outputInfo->dim[3]; // MNN format NCHW
            if (outputInfo->order == NHWC) {
                inferWidth  = outputInfo->dim[2];
                inferHeight = outputInfo->dim[1];
            }
            int realHeight = realOutputShape[1], realWidth = realOutputShape[2]; // tf format NHWC
            if (realHeight != inferHeight || realWidth != inferWidth) {
                MNN_ERROR("==== output_shape is not consistent with inferred output shape in MNN. ====\n");
                MNN_ERROR("====(height,width): (%d,%d) vs (%d,%d)\n ====", realHeight, realWidth, inferHeight,
                          inferWidth);
                return nullptr;
            }
        }
        return newExpr;
    }
};

class Dilation2DTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op               = expr->get();
        auto inputs           = expr->inputs();
        auto weight           = inputs[1];
        auto weightInfo       = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        int kh           = weightInfo->dim[0];
        int kw           = weightInfo->dim[1];
        int depth        = weightInfo->dim[2];
        weight           = _Transpose(weight, {2, 0, 1});
        weightInfo       = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common         = convolution2D->common.get();
        common->outputCount = depth;
        common->kernelX     = kw;
        common->kernelY     = kh;

        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = expr->name();
        newOp->type       = OpType_Dilation2D;
        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();

        return Expr::create(newOp.get(), {inputs[0]}, 1);
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("Conv2D", std::shared_ptr<TFExtraManager::Transform>(new ConvolutionTransform));
    TFExtraManager::get()->insert("Conv2DBackpropInput",
                                  std::shared_ptr<TFExtraManager::Transform>(new DeconvolutionTransform));
    TFExtraManager::get()->insert("DepthwiseConv2dNative",
                                  std::shared_ptr<TFExtraManager::Transform>(new ConvolutionDepthwiseTransform));
    TFExtraManager::get()->insert("DepthwiseConv2dNativeBackpropInput",
                                  std::shared_ptr<TFExtraManager::Transform>(new DeconvolutionTransform));
    TFExtraManager::get()->insert("Dilation2D", std::shared_ptr<TFExtraManager::Transform>(new Dilation2DTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
