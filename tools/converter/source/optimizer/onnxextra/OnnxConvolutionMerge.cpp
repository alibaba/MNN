//
//  OnnxConvolutionMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
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

    int co    = weightShape[0];
    int ci    = weightShape[1];
    int depth = weightShape[2];
    int kh    = weightShape[3];
    int kw    = weightShape[4];
    
    std::unique_ptr<Convolution3DT> conv3d(new MNN::Convolution3DT);
    
    auto weightDataPtr = weight->readMap<float>();
    conv3d->weight.resize(weightInfo->size);
    ::memcpy(conv3d->weight.data(), weightDataPtr, weightInfo->size * sizeof(float));
    conv3d->bias.resize(co);
    std::fill(conv3d->bias.begin(), conv3d->bias.end(), 0.0f);
    if (inputSize == 3) {
        auto biasDataPtr = inputs[2]->readMap<float>();
        ::memcpy(conv3d->bias.data(), biasDataPtr, co * sizeof(float));
    }
    
    conv3d->common.reset(new MNN::Convolution3DCommonT);
    auto common          = conv3d->common.get();
    
    common->relu = common->relu6 = false;
    common->outputCount = co;
    common->inputCount = ci;
    common->kernels = std::vector<int>({depth, kh, kw});

    const int attrSize = extraParam->attr()->size();
    for (int i = 0; i < attrSize; ++i) {
        auto attr       = extraParam->attr()->GetAs<Attribute>(i);
        const auto& key = attr->key()->str();
        if (key == "dilations") {
            auto values = attr->list()->i()->data();
            if (values[0] != 1 || values[1] != 1 || values[2] != 1) {
                MNN_ERROR("conv3d not support dilation bigger than 1\n");
                return nullptr;
            }
            common->dilates = std::vector<int>({values[0], values[1], values[2]});
        } else if (key == "group") {
            if (attr->i() != 1) {
                MNN_ERROR("group conv3d not support\n");
                return nullptr;
            }
        } else if (key == "strides") {
            auto values = attr->list()->i()->data();
            if (values[0] != 1 || values[1] != 1 || values[2] != 1) {
                MNN_ERROR("conv3d not support strides bigger than 1\n");
                return nullptr;
            }
            common->strides = std::vector<int>({values[0], values[1], values[2]});
        } else if (key == "pads") {
            auto values = attr->list()->i()->data();
            common->padMode = MNN::PadMode_CAFFE;
            common->pads = std::vector<int>({values[0], values[1], values[2]});
        }
    }

    std::unique_ptr<OpT> newOp(new OpT);
    newOp->name = expr->name();
    newOp->type = OpType_Convolution3D;
    newOp->main.type = OpParameter_Convolution3D;
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
        int kw = weightShape[3];

        if (isDeconv) {
            co = weightShape[1];
            ci = weightShape[0];
        }

        int group      = 1;
        int dilation_h = 1;
        int dilation_w = 1;
        int stride_h   = 1;
        int stride_w   = 1;
        int padX       = 0;
        int padY       = 0;
        std::vector<int> outputPadding;

        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "dilations") {
                auto dataList = attr->list();
                dilation_h    = dataList->i()->data()[0];
                dilation_w    = dataList->i()->data()[1];
            } else if (key == "group") {
                group = attr->i();
            } else if (key == "strides") {
                auto dataList = attr->list();
                stride_h      = dataList->i()->data()[0];
                stride_w      = dataList->i()->data()[1];
            } else if (key == "auto_pad") {
                if (attr->s()->str() != "NOTEST") {
                    MNN_ERROR("Conv auto_pad now only support NOTSET\n");
                    return nullptr;
                }
            } else if (key == "pads") {
                auto dataList = attr->list();
                padX          = dataList->i()->data()[1];
                padY          = dataList->i()->data()[0];
                int padX_end  = dataList->i()->data()[3];
                int padY_end  = dataList->i()->data()[2];
                if (padX != padX_end || padY != padY_end) {
                    MNN_ERROR("Asymmetrical pads in convolution is not supported\n");
                    return nullptr;
                }
            }else if (key == "output_padding"){
                // only valid in ConvTranspose
                auto dataList = attr->list();
                const int size = dataList->i()->size();
                for(int i = 0; i < size; ++i){
                    outputPadding.push_back(dataList->i()->data()[i]);
                }
            }
        }

        std::unique_ptr<Convolution2DT> convParam(new MNN::Convolution2DT);

        // read weight data
        auto weightDataPtr = weight->readMap<float>();
        // weight is Constant node
        if (weightDataPtr) {
            const int weightSize = co * ci * kh * kw;
            convParam->weight.resize(weightSize);
            ::memcpy(convParam->weight.data(), weightDataPtr, weightSize * sizeof(float));

            convParam->bias.resize(co);
            if (inputSize == 3) {
                // read bias data
                auto bias          = inputs[2];
                const int biasNums = bias->getInfo()->size;
                if (biasNums != co) {
                    // TODO broacast
                    MNN_ERROR("[TODO] Conv's bias support broadcast!\n");
                    return nullptr;
                }
                auto biasDataPtr = bias->readMap<float>();
                if (!biasDataPtr) {
                    MNN_ERROR("Conv's bias input should be Constant!\n");
                    return nullptr;
                }
                ::memcpy(convParam->bias.data(), biasDataPtr, co * sizeof(float));
            } else {
                ::memset(convParam->bias.data(), 0, co);
            }
        }

        convParam->common.reset(new MNN::Convolution2DCommonT);
        auto common = convParam->common.get();

        // set param
        common->relu        = false;
        common->group       = group;
        common->outputCount = co;
        common->inputCount  = ci * group; // conv set inputCount to be ci, dw to be group
        common->kernelX     = kw;
        common->kernelY     = kh;
        common->dilateX     = dilation_w;
        common->dilateY     = dilation_h;
        common->strideX     = stride_w;
        common->strideY     = stride_h;
        common->padX        = padX;
        common->padY        = padY;
        common->padMode     = MNN::PadMode_CAFFE;

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name = expr->name();

        if (isDeconv) {
            newOp->type = OpType_Deconvolution;
        } else {
            newOp->type = OpType_Convolution;
        }

        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convParam.release();

        if (weightDataPtr) {
            // merge weight(bias) node to Conv parameter
            
            auto realOutputExpr = Expr::create(newOp.get(), {inputs[0]});
            if(isDeconv && outputPadding.size() == 2){
                // if output_padding is not empty, add Padding after deconv
                std::vector<int> realOutputPadding(4 * 2);
                realOutputPadding[2 * 2 + 1] = outputPadding[0];
                realOutputPadding[3 * 2 + 1] = outputPadding[1];
                auto padValue = _Const(realOutputPadding.data(), {8}, NCHW, halide_type_of<int>());
                auto padInput = Variable::create(realOutputExpr);
                auto padOutput = _Pad(padInput, padValue);
                realOutputExpr = padOutput->expr().first;
            }
            return realOutputExpr;
        } else {
            // construct bias input, because mnn runtime constrain that conv should have 3 inputs when weight is not
            // Constant
            auto biasDummy = _Const(0.0, {co});
            return Expr::create(newOp.get(), {inputs[0], inputs[1], biasDummy});
        }
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
