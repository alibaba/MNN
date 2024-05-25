//
//  OpConverter.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OpConverter.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <map>
namespace MNN {

static std::map<int, OpConverter*>& getConverter() {
    static std::map<int, OpConverter*> gConverterMap;
    return gConverterMap;
}

OpConverter* OpConverter::get(int type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpConverter::insert(int type, OpConverter* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

MNN::Express::EXPRP OpConverter::convert(MNN::Express::EXPRP source, TrainInfo& trainInfo) {
    auto opOrigin = source->get();
    if (nullptr == opOrigin) {
        return source;
    }
    std::unique_ptr<MNN::OpT> op(opOrigin->UnPack());
    auto& helpInfo = trainInfo.bnVariables;
    if (op->type == MNN::OpType_BatchNorm) {
        printf("transform batchnorm: %s\n", source->name().c_str());
        
        auto params = op->main.AsBatchNorm();
        auto channels = params->channels;
        
        auto input = source->inputs()[0];
        if (input->getInfo()->dim.size() != 4) {
            printf("only support BatchNorm with 4-D input\n");
            return nullptr;
        }
        auto preExpr = input->expr().first;
        bool cond = (preExpr->get() != nullptr) && (preExpr->get()->type() == MNN::OpType_Convolution) && (preExpr->inputs().size() == 3);
        auto oriInputOrder = input->getInfo()->order;
        if (oriInputOrder == MNN::Express::NC4HW4) {
            input = MNN::Express::_Convert(input, MNN::Express::NCHW);
            if (cond) input->setName(source->name() + "_MNN_BN_after_conv_first_op");
            else input->setName(source->name() + "_MNN_single_BN_first_op");
        }
        auto inputOrder = input->getInfo()->order;
        
        std::vector<int> reduceDims = {0, 2, 3};
        std::vector<int> statShape = {1, channels, 1, 1};
        if (inputOrder == MNN::Express::NHWC) {
            reduceDims = {0, 1, 2};
            statShape = {1, 1, 1, channels};
        }
        
        auto rMean = MNN::Express::_Const((void*)params->meanData.data(), statShape, inputOrder);
        rMean->setName(source->name() + "_BN_RunningMean_Weight");
        auto rVar = MNN::Express::_Const((void*)params->varData.data(), statShape, inputOrder);
        rVar->setName(source->name() + "_BN_RunningVariance_Weight");
        auto w = MNN::Express::_Const((void*)params->slopeData.data(), statShape, inputOrder);
        w->setName(source->name() + "_BN_Gamma_Weight");
        auto b = MNN::Express::_Const((void*)params->biasData.data(), statShape, inputOrder);
        b->setName(source->name() + "_BN_Beta_Bias");
        auto eps = MNN::Express::_Scalar<float>(params->epsilon);
        eps->setName(source->name() + "_BN_Eps_Weight");
        
        auto meanX = MNN::Express::_ReduceMean(input, reduceDims, true);
        meanX->setName(source->name() + "_BN_xmean");
        auto varX = MNN::Express::_ReduceMean(_Square(input - meanX), reduceDims, true);
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
        if (oriInputOrder != MNN::Express::NC4HW4) {
            if (cond) subMean->setName(source->name() + "_MNN_BN_after_conv_first_op");
            else subMean->setName(source->name() + "_MNN_single_BN_first_op");
        }
        auto normed = subMean / stdFinal;
        auto res = normed * w + b;
        
        if (oriInputOrder == MNN::Express::NC4HW4) {
            res = MNN::Express::_Convert(res, oriInputOrder);
        }
        res->setName(source->name());
        return res->expr().first;
    }
    
    if (op->type != MNN::OpType_Convolution && op->type != MNN::OpType_ConvolutionDepthwise) {
        return source;
    }
    auto conv2D       = op->main.AsConvolution2D();
    auto conv2DCommon = conv2D->common.get();
    auto inputs       = source->inputs();
    if (inputs.size() == 3) {
        return source;
    }
    MNN::Express::VARP weightValue, biasValue;
    {
        std::unique_ptr<MNN::OpT> weight(new MNN::OpT);
        weight->type      = MNN::OpType_Const;
        weight->main.type = MNN::OpParameter_Blob;
        auto srcCount     = (int)conv2D->weight.size() * conv2DCommon->group / conv2DCommon->outputCount /
        conv2DCommon->kernelX / conv2DCommon->kernelY;
        weight->main.value          = new MNN::BlobT;
        weight->main.AsBlob()->dims = {conv2DCommon->outputCount, srcCount / conv2DCommon->group, conv2DCommon->kernelY,
            conv2DCommon->kernelX};
        weight->main.AsBlob()->dataType   = MNN::DataType_DT_FLOAT;
        weight->main.AsBlob()->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;
        weight->main.AsBlob()->float32s   = std::move(op->main.AsConvolution2D()->weight);
        MNN::Express::EXPRP weightExpr                  = MNN::Express::Expr::create(std::move(weight), {}, 1);
        weightValue                       = MNN::Express::Variable::create(weightExpr, 0);
        conv2DCommon->inputCount          = srcCount;
    }
    biasValue = MNN::Express::_Const((const void*)conv2D->bias.data(), {(int)conv2D->bias.size()}, MNN::Express::NCHW);
    weightValue->setName(source->name() + "_Weight");
    biasValue->setName(source->name() + "_Bias");
    trainInfo.convolutionVariables.insert(std::make_pair(source->name(), std::make_pair(weightValue->name(), biasValue->name())));
    // Origin Convolution
    std::unique_ptr<MNN::OpT> newConvOp(new MNN::OpT);
    {
        newConvOp->type       = op->type;
        newConvOp->main.type  = MNN::OpParameter_Convolution2D;
        newConvOp->main.value = new MNN::Convolution2DT;
        newConvOp->main.AsConvolution2D()->common.reset(new MNN::Convolution2DCommonT(*conv2DCommon));
    }

    newConvOp->main.AsConvolution2D()->common->relu6 = false;
    newConvOp->main.AsConvolution2D()->common->relu  = false;
    
    auto relu  = conv2DCommon->relu;
    auto relu6 = conv2DCommon->relu6;
    
    MNN::Express::EXPRP newConv       = MNN::Express::Expr::create(std::move(newConvOp), {inputs[0], weightValue, biasValue});
    MNN::Express::VARP resultVariable = MNN::Express::Variable::create(newConv, 0);
    resultVariable->setName(source->name());
    if (relu) {
        resultVariable = MNN::Express::_Relu(resultVariable);
        resultVariable->setName(source->name() + "_Relu");
    } else if (relu6) {
        resultVariable = MNN::Express::_Relu6(resultVariable);
        resultVariable->setName(source->name() + "_Relu6");
    }
    return resultVariable->expr().first;
}
};
