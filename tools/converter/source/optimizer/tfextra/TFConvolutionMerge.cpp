//
//  TFConvolutionMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "TFExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static bool _writeCommonAttr(Convolution2DCommonT* common, const Extra* extra, const std::string& name) {
    if (nullptr == extra || nullptr == extra->attr()) {
        return false;
    }
    auto attrSize = extra->attr()->size();
    for (int v=0; v<attrSize; ++v) {
        auto attr = extra->attr()->GetAs<Attribute>(v);
        const auto key = attr->key()->str();
        auto list = attr->list();
        if (key == "rate") {
            common->dilateX = list->i()->data()[2];
            common->dilateY = list->i()->data()[1];
        } else if (key == "strides") {
            common->strideX = list->i()->data()[2];
            common->strideY = list->i()->data()[1];
        } else if (key == "padding") {
            common->padMode = MNN::PadMode_SAME;
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
        auto op = expr->get();
        auto inputs = expr->inputs();
        auto weight = inputs[1];
        auto weightInfo = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }
        
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        int kh         = weightInfo->dim[0];
        int kw         = weightInfo->dim[1];
        int num_input  = weightInfo->dim[2];
        int num_output = weightInfo->dim[3];
        weight = _Transpose(weight, {3, 2, 0, 1});
        weightInfo = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        convolution2D->bias.resize(num_output);
        std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);

        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common          = convolution2D->common.get();

        common->relu        = false;
        common->group       = 1;
        common->outputCount = num_output;
        common->inputCount  = num_input;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->padX = 0;
        common->padY = 0;

        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_Convolution;
        newOp->main.type = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();

        auto newExpr = Expr::create(newOp.get(), {inputs[0]}, 1);
        return newExpr;
    }
};

class ConvolutionDepthwiseTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto inputs = expr->inputs();
        auto weight = inputs[1];
        auto weightInfo = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }
        
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);

        int kh         = weightInfo->dim[0];
        int kw         = weightInfo->dim[1];
        int num_input  = weightInfo->dim[2];
        int num_output = num_input;
        weight = _Transpose(weight, {3, 2, 0, 1});
        weightInfo = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        convolution2D->bias.resize(num_output);
        std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common          = convolution2D->common.get();

        common->relu        = false;
        common->group       = 1;
        common->outputCount = num_output;
        common->inputCount  = num_input;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->padX = 0;
        common->padY = 0;
        
        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_ConvolutionDepthwise;
        newOp->main.type = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();

        auto newExpr = Expr::create(newOp.get(), {inputs[0]}, 1);
        return newExpr;
    }
};

class DeconvolutionTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto inputs = expr->inputs();
        auto weight = inputs[1];
        auto weightInfo = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s convolution weight is not const\n", expr->name().c_str());
            return nullptr;
        }
        std::unique_ptr<Convolution2DT> convolution2D(new MNN::Convolution2DT);
        int kh         = weightInfo->dim[0];
        int kw         = weightInfo->dim[1];
        int num_input  = weightInfo->dim[2];
        int num_output = weightInfo->dim[3];;
        weight = _Transpose(weight, {3, 2, 0, 1});
        weightInfo = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        convolution2D->weight.resize(weightInfo->size);
        ::memcpy(convolution2D->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        convolution2D->bias.resize(num_input);
        std::fill(convolution2D->bias.begin(), convolution2D->bias.end(), 0.0f);
        convolution2D->common.reset(new MNN::Convolution2DCommonT);
        auto common          = convolution2D->common.get();

        common->relu        = false;
        common->group       = 1;
        common->outputCount = num_input;
        common->inputCount  = num_output;
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->padX = 0;
        common->padY = 0;
        
        bool success = _writeCommonAttr(common, op->main_as_Extra(), op->name()->str());
        if (!success) {
            return nullptr;
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_Deconvolution;
        newOp->main.type = OpParameter_Convolution2D;
        newOp->main.value = convolution2D.release();
        if (inputs.size() == 2) {
            return Expr::create(newOp.get(), {inputs[0]}, 1);
        }
        MNN_ASSERT(inputs.size() == 3);
        auto newExpr = Expr::create(newOp.get(), {inputs[2]}, 1);
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("Conv2D", std::shared_ptr<TFExtraManager::Transform>(new ConvolutionTransform));
    TFExtraManager::get()->insert("Conv2DBackpropInput", std::shared_ptr<TFExtraManager::Transform>(new DeconvolutionTransform));
    TFExtraManager::get()->insert("DepthwiseConv2dNative", std::shared_ptr<TFExtraManager::Transform>(new ConvolutionDepthwiseTransform));
    return true;
}();
}
} // namespace MNN

