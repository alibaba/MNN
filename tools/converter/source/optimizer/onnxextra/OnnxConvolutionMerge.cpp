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

class OnnxConvolutionTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
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
        common->inputCount  = group == 1 ? ci : group; // conv set inputCount to be ci, dw to be group
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

        if (isDeconv) {
            newOp->type = OpType_Deconvolution;
        } else {
            newOp->type = OpType_Convolution;
        }

        newOp->main.type  = OpParameter_Convolution2D;
        newOp->main.value = convParam.release();

        if (weightDataPtr) {
            // merge weight(bias) node to Conv parameter
            return Expr::create(newOp.get(), {inputs[0]});
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
