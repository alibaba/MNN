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

EXPRP OpConverter::convert(EXPRP source) {
    auto opOrigin = source->get();
    if (nullptr == opOrigin) {
        return source;
    }
    std::unique_ptr<MNN::OpT> op(opOrigin->UnPack());
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
    if (conv2DCommon->padX != 0 && conv2DCommon->padMode == PadMode_CAFFE) {
        newConvOp->main.AsConvolution2D()->common->padMode = PadMode_SAME;
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
