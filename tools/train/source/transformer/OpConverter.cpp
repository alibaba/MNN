//
//  OpConverter.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <map>

using namespace MNN;
using namespace MNN::Express;
static std::map<MNN::OpType, OpConverter*>& getConverter() {
    static std::map<MNN::OpType, OpConverter*> gConverterMap;
    return gConverterMap;
}

OpConverter* OpConverter::get(MNN::OpType type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpConverter::insert(MNN::OpType type, OpConverter* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

EXPRP OpConverter::convert(EXPRP source, std::map<std::string, MNN::Express::VARP>& helpInfo) {
    auto opOrigin = source->get();
    if (nullptr == opOrigin) {
        return source;
    }
    std::unique_ptr<MNN::OpT> op(opOrigin->UnPack());
    if (op->type == OpType_BatchNorm) {
        printf("transform batchnorm: %s\n", source->name().c_str());

        auto params = op->main.AsBatchNorm();
        auto channels = params->channels;

        auto input = source->inputs()[0];
        if (input->getInfo()->dim.size() != 4) {
            printf("only support BatchNorm with 4-D input\n");
            return nullptr;
        }
        auto preExpr = input->expr().first;
        bool cond = (preExpr->get() != nullptr) && (preExpr->get()->type() == OpType_Convolution) && (preExpr->inputs().size() == 3);
        auto oriInputOrder = input->getInfo()->order;
        if (oriInputOrder == NC4HW4) {
            input = _Convert(input, NCHW);
            if (cond) input->setName(source->name() + "_MNN_BN_after_conv_first_op");
            else input->setName(source->name() + "_MNN_single_BN_first_op");
        }
        auto inputOrder = input->getInfo()->order;

        std::vector<int> reduceDims = {0, 2, 3};
        std::vector<int> statShape = {1, channels, 1, 1};
        if (inputOrder == NHWC) {
            reduceDims = {0, 1, 2};
            statShape = {1, 1, 1, channels};
        }

        auto rMean = _Const((void*)params->meanData.data(), statShape, inputOrder);
        rMean->setName(source->name() + "_BN_RunningMean_Weight");
        auto rVar = _Const((void*)params->varData.data(), statShape, inputOrder);
        rVar->setName(source->name() + "_BN_RunningVariance_Weight");
        auto w = _Const((void*)params->slopeData.data(), statShape, inputOrder);
        w->setName(source->name() + "_BN_Gamma_Weight");
        auto b = _Const((void*)params->biasData.data(), statShape, inputOrder);
        b->setName(source->name() + "_BN_Beta_Bias");
        auto eps = _Scalar<float>(params->epsilon);
        eps->setName(source->name() + "_BN_Eps_Weight");

        auto meanX = _ReduceMean(input, reduceDims, true);
        meanX->setName(source->name() + "_BN_xmean");
        auto varX = _ReduceMean(_Square(input - meanX), reduceDims, true);
        varX->setName(source->name() + "_BN_xvariance");

        auto isTraining = helpInfo["is_training_float"];
        auto one = helpInfo["one_float"];
        auto momentum = helpInfo["bn_momentum"] * isTraining + (one - isTraining) * one;
        
        auto mMean = momentum * rMean + (one - momentum) * meanX;
        mMean->setName(source->name() + "_BN_momentum_mean");
        helpInfo[rMean->name()] = mMean;
        auto mVar = momentum * rVar + (one - momentum) * varX;
        mVar->setName(source->name() + "_BN_momentum_variance");
        helpInfo[rVar->name()] = mVar;

        auto meanFinal = isTraining * meanX + (one - isTraining) * mMean;
        meanFinal->setName(source->name() + "_BN_mean_final");
        auto varFinal = isTraining * varX + (one - isTraining) * mVar;
        varFinal->setName(source->name() + "_BN_variance_final");
        auto stdFinal = _Sqrt(varFinal + eps);

        auto subMean = input - meanFinal;
        if (oriInputOrder != NC4HW4) {
            if (cond) subMean->setName(source->name() + "_MNN_BN_after_conv_first_op");
            else subMean->setName(source->name() + "_MNN_single_BN_first_op");
        }
        auto normed = subMean / stdFinal;
        auto res = normed * w + b;

        if (oriInputOrder == NC4HW4) {
            res = _Convert(res, oriInputOrder);
        }
        res->setName(source->name());
        return res->expr().first;
    }

    if (op->type != OpType_Convolution && op->type != OpType_ConvolutionDepthwise) {
        return source;
    }
    auto conv2D       = op->main.AsConvolution2D();
    auto conv2DCommon = conv2D->common.get();
    auto inputs       = source->inputs();
    if (inputs.size() == 3) {
        return source;
    }
    VARP weightValue, biasValue;
    {
        std::unique_ptr<OpT> weight(new OpT);
        weight->type      = OpType_Const;
        weight->main.type = OpParameter_Blob;
        auto srcCount     = (int)conv2D->weight.size() * conv2DCommon->group / conv2DCommon->outputCount /
                        conv2DCommon->kernelX / conv2DCommon->kernelY;
        weight->main.value          = new BlobT;
        weight->main.AsBlob()->dims = {conv2DCommon->outputCount, srcCount / conv2DCommon->group, conv2DCommon->kernelY,
                                       conv2DCommon->kernelX};
        weight->main.AsBlob()->dataType   = DataType_DT_FLOAT;
        weight->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NCHW;
        weight->main.AsBlob()->float32s   = std::move(op->main.AsConvolution2D()->weight);
        EXPRP weightExpr                  = Expr::create(std::move(weight), {}, 1);
        weightValue                       = Variable::create(weightExpr, 0);
        conv2DCommon->inputCount          = srcCount;
    }
    biasValue = _Const((const void*)conv2D->bias.data(), {(int)conv2D->bias.size()}, NCHW);
    weightValue->setName(source->name() + "_Weight");
    biasValue->setName(source->name() + "_Bias");
    // Origin Convolution
    std::unique_ptr<OpT> newConvOp(new OpT);
    {
        newConvOp->type       = op->type;
        newConvOp->main.type  = OpParameter_Convolution2D;
        newConvOp->main.value = new Convolution2DT;
        newConvOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT(*conv2DCommon));
    }

    newConvOp->main.AsConvolution2D()->common->relu6 = false;
    newConvOp->main.AsConvolution2D()->common->relu  = false;

    auto relu  = conv2DCommon->relu;
    auto relu6 = conv2DCommon->relu6;

    EXPRP newConv       = Expr::create(std::move(newConvOp), {inputs[0], weightValue, biasValue});
    VARP resultVariable = Variable::create(newConv, 0);
    resultVariable->setName(source->name());
    if (relu) {
        resultVariable = _Relu(resultVariable);
        resultVariable->setName(source->name() + "_Relu");
    } else if (relu6) {
        resultVariable = _Relu6(resultVariable);
        resultVariable->setName(source->name() + "_Relu6");
    }
    return resultVariable->expr().first;
}
