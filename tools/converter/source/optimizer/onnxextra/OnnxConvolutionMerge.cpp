//
//  OnnxConvolutionMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace Express {

static int convSpatialDim(EXPRP expr) {
    auto attrs = expr->get()->main_as_Extra()->attr();
    for (int i = 0; i < attrs->size(); ++i) {
        auto attr = attrs->GetAs<Attribute>(i);
        if (attr->key()->str() == "kernel_shape") {
            return attr->list()->i()->size();
        }
    }
    return -1;
}

static EXPRP _transformConv3D(EXPRP expr) {
    auto inputs         = expr->inputs();
    const int inputSize = inputs.size();
    if (inputSize != 3 && inputSize != 2) {
        MNN_ERROR("Convolution3D Input ERROR!\n");
        return nullptr;
    }
    auto weight = inputs[1];

    auto weightInfo = weight->getInfo();
    if (nullptr == weightInfo) {
        MNN_ERROR("Convolution3D should know weight shape infromation!\n");
        return nullptr;
    }
    auto& weightShape = weightInfo->dim;

    auto extraParam = expr->get()->main_as_Extra();
    std::string originalOpType(extraParam->type()->c_str());
    bool isDeconv = originalOpType == "ConvTranspose";
    int co    = weightShape[0];
    int ci    = weightShape[1];
    int depth = weightShape[2];
    int kh    = weightShape[3];
    int kw    = weightShape[4];
    if (isDeconv) {
        co = weightShape[1];
        ci = weightShape[0];
    }
    std::unique_ptr<Convolution3DT> conv3d(new MNN::Convolution3DT);
    const float* weightDataPtr = weight->readMap<float>();
    conv3d->weight.resize(weightInfo->size);
    ::memcpy(conv3d->weight.data(), weightDataPtr, weightInfo->size * sizeof(float));
    conv3d->bias.resize(co);
    std::fill(conv3d->bias.begin(), conv3d->bias.end(), 0.0f);
    if (inputSize == 3) {
        auto biasDataPtr = inputs[2]->readMap<float>();
        ::memcpy(conv3d->bias.data(), biasDataPtr, co * sizeof(float));
    }

    conv3d->common.reset(new MNN::Convolution3DCommonT);
    auto common = conv3d->common.get();
    const int attrSize = extraParam->attr()->size();
    std::vector<int> outputPadding;
    for (int i = 0; i < attrSize; ++i) {
        auto attr       = extraParam->attr()->GetAs<Attribute>(i);
        const auto& key = attr->key()->str();
        if (key == "dilations") {
            auto values     = attr->list()->i()->data();
            common->dilates = std::vector<int>({values[0], values[1], values[2]});
        } else if (key == "group") {
            common->group = attr->i();
        } else if (key == "strides") {
            auto values     = attr->list()->i()->data();
            common->strides = std::vector<int>({values[0], values[1], values[2]});
        } else if (key == "pads") {
            auto values     = attr->list()->i()->data();
            common->padMode = MNN::PadMode_CAFFE;
            common->pads    = std::vector<int>({values[0], values[1], values[2]});
        } else if (key == "output_padding") {
            // only valid in ConvTranspose
            auto dataList  = attr->list();
            const int size = dataList->i()->size();
            for (int k = 0; k < size; ++k) {
                outputPadding.push_back(dataList->i()->data()[k]);
            }
        }
        // TODO: Support outputshape
    }
    common->outPads = outputPadding;

    common->relu = common->relu6 = false;
    if (isDeconv) {
        common->outputCount = co * common->group; // deconv set inputCount to be ci, dw to be group
        common->inputCount = ci;
    } else {
        common->outputCount = co;
        common->inputCount  = ci * common->group; // conv set inputCount to be ci, dw to be group
    }
    common->kernels              = std::vector<int>({depth, kh, kw});

    std::unique_ptr<OpT> newOp(new OpT);
    newOp->name       = expr->name();
    newOp->type       = isDeconv ? OpType_ConvTranspose3D : OpType_Convolution3D;
    newOp->main.type  = OpParameter_Convolution3D;
    newOp->main.value = conv3d.release();

    auto newExpr = Expr::create(newOp.get(), {inputs[0]}, 1);
    return newExpr;
}

class OnnxConvolutionTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        if (convSpatialDim(expr) == 3) {
            return _transformConv3D(expr);
        }
        auto inputs         = expr->inputs();
        const int inputSize = inputs.size();
        if (inputSize != 3 && inputSize != 2) {
            MNN_ERROR("Convolution Input ERROR!\n");
            return nullptr;
        }
        auto weight = inputs[1];

        auto weightInfo = weight->getInfo();
        if (nullptr == weightInfo) {
            MNN_ERROR("Convolution should know weight shape infromation!\n");
            return nullptr;
        }
        auto& weightShape = weightInfo->dim;

        auto op         = expr->get();
        auto extraParam = op->main_as_Extra();
        std::string originalOpType(extraParam->type()->c_str());
        bool isDeconv = originalOpType == "ConvTranspose";

        int co = weightShape[0];
        int ci = weightShape[1];
        int kh = weightShape[2];
        int kw = 1;
        if (weightShape.size() >= 4) {
            kw = weightShape[3];
        }

        if (isDeconv) {
            co = weightShape[1];
            ci = weightShape[0];
        }

        int group           = 1;
        int dilation_h      = 1;
        int dilation_w      = 1;
        int stride_h        = 1;
        int stride_w        = 1;
        PadMode modePadding = PadMode_CAFFE;
        std::vector<int> outputPadding;
        std::vector<int> inputPadding;
        std::vector<int> outputShape;

        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "dilations") {
                auto dataList = attr->list();
                dilation_h    = dataList->i()->data()[0];
                if (dataList->i()->size() >= 2) {
                    dilation_w      = dataList->i()->data()[1];
                }
            } else if (key == "group") {
                group = attr->i();
            } else if (key == "strides") {
                auto dataList = attr->list();
                stride_h      = dataList->i()->data()[0];
                if (dataList->i()->size() >= 2) {
                    stride_w      = dataList->i()->data()[1];
                }
            } else if (key == "auto_pad") {
                if (attr->s()->str() == "NOTSET") {
                    modePadding = PadMode_CAFFE;
                } else if (attr->s()->str() == "SAME_UPPER" || attr->s()->str() == "SAME_LOWER") {
                    modePadding = PadMode_SAME;
                } else if (attr->s()->str() == "VALID") {
                    modePadding = PadMode_VALID;
                } else {
                    MNN_ERROR("Conv auto_pad not support %s\n", attr->s()->c_str());
                    return nullptr;
                }
            } else if (key == "pads") {
                auto dataList = attr->list();
                inputPadding.resize(dataList->i()->size());
                for (int v = 0; v < inputPadding.size(); v++) {
                    inputPadding[v] = dataList->i()->data()[v];
                }
                // Support Convolution 1D
                if (inputPadding.size() == 2) {
                    inputPadding = {inputPadding[0], 0, inputPadding[1], 0};
                }
            } else if (key == "output_padding") {
                // only valid in ConvTranspose
                auto dataList  = attr->list();
                const int size = dataList->i()->size();
                for (int k = 0; k < size; ++k) {
                    outputPadding.push_back(dataList->i()->data()[k]);
                }
                if (outputPadding.size() == 1) {
                    outputPadding = {outputPadding[0], 0};
                }
            } else if (key == "output_shape") {
                auto dataList = attr->list();
                outputShape.resize(dataList->i()->size());
                for (int v = 0; v < outputShape.size(); v++) {
                    outputShape[v] = dataList->i()->data()[v];
                }
            }
        }

        std::unique_ptr<Convolution2DT> convParam(new MNN::Convolution2DT);
        convParam->common.reset(new MNN::Convolution2DCommonT);
        auto common = convParam->common.get();

        // For old mnn compability
        if (inputPadding.size() >= 4) {
            common->padY = inputPadding[0];
            common->padX = inputPadding[1];
        }
        common->padMode = modePadding;

        // set param
        common->relu  = false;
        common->group = group;
        if (isDeconv) {
            common->outputCount = co * group; // deconv set inputCount to be ci, dw to be group
            common->inputCount = ci;
        } else {
            common->outputCount = co;
            common->inputCount  = ci * group; // conv set inputCount to be ci, dw to be group
        }
        common->kernelX = kw;
        common->kernelY = kh;
        common->dilateX = dilation_w;
        common->dilateY = dilation_h;
        common->strideX = stride_w;
        common->strideY = stride_h;
        common->pads    = inputPadding;
        common->outPads = outputPadding;
        if (!outputShape.empty()) {
            common->hasOutputShape = true;
            common->padMode = PadMode_SAME;
        }
        auto config = Global<modelConfig>::Get();
        // read weight data
        const float* weightDataPtr = nullptr;
        int limitNumber = 4;
        if (config->optimizePrefer == 1) {
            // Smallest
            limitNumber = 1;
        } else if (config->optimizePrefer == 2) {
            // Fastest
            limitNumber = 100;
        }
        if (weight->linkNumber() <= limitNumber) {
            weightDataPtr = weight->readMap<float>();
        }
        // weight is Constant node
        if (weightDataPtr) {
            if (weight->linkNumber() > 1) {
                static bool gPrint = false;
                if (!gPrint) {
                    MNN_PRINT("The Convolution use shared weight, may increase the model size\n");
                    gPrint = true;
                }
            }

            // MNN_PRINT("MNNCountNNZBlock:%p\n", MNNCountNNZBlock);
            const size_t weightSize = co * ci * kh * kw;
            convParam->weight.resize(weightSize);
            ::memcpy(convParam->weight.data(), weightDataPtr, weightSize * sizeof(float));
            convParam->bias.resize(common->outputCount);
            if (inputSize == 3) {
                // read bias data
                auto bias          = inputs[2];
                const int biasNums = bias->getInfo()->size;
                if (biasNums != common->outputCount) {
                    // TODO broacast
                    MNN_ERROR("[TODO] Conv's bias support broadcast!\n");
                    return nullptr;
                }
                auto biasDataPtr = bias->readMap<float>();
                if (!biasDataPtr) {
                    MNN_ERROR("Conv's bias input should be Constant!\n");
                    return nullptr;
                }
                ::memcpy(convParam->bias.data(), biasDataPtr, common->outputCount * sizeof(float));
            } else {
                ::memset(convParam->bias.data(), 0, common->outputCount * sizeof(float));
            }
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name = expr->name();

        if (isDeconv) {
            newOp->type = OpType_Deconvolution;
            if (group > 1 && group == ci * co) {
                newOp->type = OpType_DeconvolutionDepthwise;
            }
        } else {
            newOp->type = OpType_Convolution;
            if (group > 1 && group == ci * co) {
                newOp->type = OpType_ConvolutionDepthwise;
            }
        }

        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convParam.release();
        auto x            = inputs[0];
        bool needSqueeze  = false;
        if (nullptr != x->getInfo()) {
            if (x->getInfo()->dim.size() == 3) {
                x           = _Unsqueeze(x, {3});
                needSqueeze = true;
            }
        }
        EXPRP convolutinExpr;
        if (!outputShape.empty()) {
            // [1, outputHeight, outputWidth, 1]
            outputShape.insert(outputShape.begin(), 1);
            outputShape.push_back(1);
            auto output_shape = _Const(outputShape.data(), {4}, NHWC, halide_type_of<int>());
            if (weightDataPtr) {
                // merge weight(bias) node to Conv parameter
                convolutinExpr = Expr::create(newOp.get(), {x, output_shape});
            } else {
                // construct bias input, because mnn runtime constrain that conv should have 3 inputs when weight is not
                // Constant
                if (inputs.size() > 2) {
                    convolutinExpr = Expr::create(newOp.get(), {x, inputs[1], inputs[2], output_shape});
                } else {
                    convolutinExpr = Expr::create(newOp.get(), {x, inputs[1], output_shape});
                }
            }
        } else if (weightDataPtr) {
            // merge weight(bias) node to Conv parameter
            convolutinExpr = Expr::create(newOp.get(), {x});
        } else {
            // construct bias input, because mnn runtime constrain that conv should have 3 inputs when weight is not
            // Constant
            if (inputs.size() > 2) {
                convolutinExpr = Expr::create(newOp.get(), {x, inputs[1], inputs[2]});
            } else {
                convolutinExpr = Expr::create(newOp.get(), {x, inputs[1]});
            }
        }
        convolutinExpr->setName(expr->name());
        auto res = Variable::create(convolutinExpr);
        if (needSqueeze) {
            res = _Squeeze(res, {3});
        }
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Conv", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxConvolutionTransform));

    OnnxExtraManager::get()->insert("ConvTranspose",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxConvolutionTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
