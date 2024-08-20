//
//  ConvQuantizeDequantizeLinearFuseToConvInt8.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"
#include "MNN_compression.pb.h"
#include <fstream>

namespace MNN {
namespace Express {
static VARP _ReshapeF(VARP x, VARP shape, MNN::MNN_DATA_FORMAT format) {
    MNN_ASSERT(nullptr != x);
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = format;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}

static VARP _ConvertF(VARP input, MNN::MNN_DATA_FORMAT format) {
    std::unique_ptr<OpT> convert(new OpT);
    convert->type                               = OpType_ConvertTensor;
    convert->main.type                          = OpParameter_TensorConvertInfo;
    convert->main.value                         = new TensorConvertInfoT;
    convert->main.AsTensorConvertInfo()->source = MNN_DATA_FORMAT_NC4HW4;
    convert->main.AsTensorConvertInfo()->dest   = format;
    return (Variable::create(Expr::create(convert.get(), {input})));
}

static bool matchConvInt8ToOther(EXPRP expr, int i) { // convint8->quant->cast->dequant->other
    // check op type not convint8.
    if (nullptr == expr->get()) {
        return false;
    }
    if (expr->get()->type() == OpType_ConvInt8 || expr->get()->type() == OpType_Cast || expr->get()->type() == OpType_Int8ToFloat || expr->get()->type() == OpType_FloatToInt8 || expr->get()->type() == OpType_Const || expr->get()->type() == OpType_DepthwiseConvInt8 || expr->get()->type() == OpType_MatMul) {
        return false;
    }
    // check dequantize linear
    VARP dequant_var   = expr->inputs().at(i);
    EXPRP dequant_expr = dequant_var->expr().first;
    if (!dequant_expr->get() || dequant_expr->get()->type() != OpType_Int8ToFloat) {
        return false;
    }
    if (dequant_expr->inputs().size() != 5) {
        return false;
    }
    // check cast
    VARP cast_var = dequant_expr->inputs().at(0);
    EXPRP cast_expr = cast_var->expr().first;
    if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
        return false;
    }
    // check quantize linear
    VARP quan_var = cast_expr->inputs().at(0);
    EXPRP quan_expr = quan_var->expr().first;
    if (!quan_expr->get() || quan_expr->get()->type() != OpType_FloatToInt8) {
        return false;
    }
    if (quan_expr->inputs().size() != 5) {
        return false;
    }
    // check convInt8
    VARP conv_var = quan_expr->inputs().at(0);
    EXPRP conv_expr = conv_var->expr().first;
    
    if (!conv_expr->get() || (conv_expr->get()->type() != OpType_ConvInt8 && conv_expr->get()->type() != OpType_DepthwiseConvInt8 && conv_expr->get()->type() != OpType_ReLU && conv_expr->get()->type() != OpType_ReLU6)) {
        return false;
    }
    if (conv_expr->get()->type() == OpType_ReLU || conv_expr->get()->type() == OpType_ReLU6) {
        conv_var = conv_expr->inputs().at(0);
        conv_expr = conv_var->expr().first;
        if (!conv_expr->get() || (conv_expr->get()->type() != OpType_ConvInt8 && conv_expr->get()->type() != OpType_DepthwiseConvInt8)) {
            return false;
        }
    }
    return true;
}
static VARP transformConvInt8ToOther(EXPRP expr, int i) { // convint8->quant->cast->dequant->other => convInt8(float output)->other
    auto dequant_var  = expr->inputs()[i];
    auto dequant_expr  = dequant_var->expr().first;
    auto cast_var = dequant_expr->inputs().at(0);
    auto cast_expr = cast_var->expr().first;
    auto quan_var = cast_expr->inputs().at(0);
    auto quan_expr = quan_var->expr().first;
    auto conv_var = quan_expr->inputs().at(0);
    auto conv_expr = conv_var->expr().first;
    auto convInt8Input = conv_expr->inputs().at(0);
    bool hasRelu = false, hasRelu6 = false;
    if (conv_expr->get()->type() == OpType_ReLU || conv_expr->get()->type() == OpType_ReLU6) {
        hasRelu = conv_expr->get()->type() == OpType_ReLU ? true : false;
        hasRelu6 = conv_expr->get()->type() == OpType_ReLU6 ? true : false;
        conv_expr = convInt8Input->expr().first;
        convInt8Input = conv_expr->inputs().at(0);
    }

    // change old convInt8 to return a float value, which is input to expr;
    std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
    std::unique_ptr<OpT> oldConvOp(conv_expr->get()->UnPack());
    auto oldConvParams  = oldConvOp->main.AsConvolution2D();
    
    float output_zero  = oldConvParams->symmetricQuan->outputZeroPoint;
    float output_scale = oldConvParams->quanParameter->scaleOut;
    float input_scale  = oldConvParams->quanParameter->scaleIn;
    float input_zero   = oldConvParams->symmetricQuan->zeroPoint;

    newConvInt8->common.reset(new MNN::Convolution2DCommonT);
    newConvInt8->common = std::move(oldConvParams->common);
    newConvInt8->common->relu = hasRelu;
    newConvInt8->common->relu6 = hasRelu6;
    newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
    newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
    //newConvInt8->symmetricQuan->outputDataType = MNN::DataType_DT_FLOAT;
    newConvInt8->quanParameter.reset(new IDSTQuanT);
    newConvInt8->bias = std::move(oldConvParams->bias);
    newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
    
    std::unique_ptr<OpT> conv_op(new OpT);
    conv_op->name = conv_expr->name();
    conv_op->type = OpType_ConvInt8;
    conv_op->main.type  = OpParameter_Convolution2D;
    conv_op->main.value = newConvInt8.release();

    convInt8Input->writeScaleMap(input_scale, input_zero);
    auto newconv_expr = Expr::create(conv_op.get(), {convInt8Input});
    newconv_expr->setName(conv_expr->name());
    auto newconv_var = Variable::create(newconv_expr);
    newconv_var->setName(conv_expr->outputName(0));
    newconv_var->writeScaleMap(output_scale, output_zero);
    if (conv_expr->inputs().size() == 5) { // Process matmul output
        auto config = Global<modelConfig>::Get();
        auto format = MNN::MNN_DATA_FORMAT_NCHW;
        if (config->model == modelConfig::TFLITE || config->model == modelConfig::TENSORFLOW) {
            format = MNN_DATA_FORMAT_NHWC;
        }
        // expr->inputs = {input, concat, needSqueezeA, needSqueezeB, transposeA}
        auto concat_var = conv_expr->inputs().at(1);
        bool needSqueezeA = conv_expr->inputs().at(2)->readMap<float>()[0] > 0.f;
        bool needSqueezeB = conv_expr->inputs().at(3)->readMap<float>()[0] > 0.f;

        auto output = _ConvertF(newconv_var, format);
        output->writeScaleMap(output_scale, output_zero);
        VARP reshapeVar = _ReshapeF(output, concat_var, format);
        reshapeVar->writeScaleMap(output_scale, output_zero);
        if (needSqueezeA) {
            reshapeVar = _Squeeze(reshapeVar, {0});
            reshapeVar->writeScaleMap(output_scale, output_zero);
        }
        if (needSqueezeB) {
            reshapeVar = _Squeeze(reshapeVar, {1});
            reshapeVar->writeScaleMap(output_scale, output_zero);
        }
        reshapeVar->setName(expr->outputName(0) + "__matmul_cvt_convInt8_reshape");
        Expr::replace(conv_expr, reshapeVar->expr().first);
        return reshapeVar;
    }
    Expr::replace(conv_expr, newconv_expr);
    return newconv_var;
    
}

static bool matchOtherToOther (EXPRP expr, int i) { // ohter->quant->cast->dequant->other
    // check op type not convint8.
    if (nullptr == expr->get()) {
        return false;
    }
    if (expr->get()->type() == OpType_ConvInt8 || expr->get()->type() == OpType_Cast || expr->get()->type() == OpType_Int8ToFloat || expr->get()->type() == OpType_FloatToInt8 || expr->get()->type() == OpType_Const || expr->get()->type() == OpType_DepthwiseConvInt8 || expr->get()->type() == OpType_MatMul) {
        return false;
    }
    // check dequantize linear
    VARP dequant_var   = expr->inputs().at(i);
    EXPRP dequant_expr = dequant_var->expr().first;
    if (!dequant_expr->get() || dequant_expr->get()->type() != OpType_Int8ToFloat) {
        return false;
    }
    if (dequant_expr->inputs().size() != 5) {
        return false;
    }
    // check cast
    VARP cast_var = dequant_expr->inputs().at(0);
    EXPRP cast_expr = cast_var->expr().first;
    if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
        return false;
    }
    // check quantize linear
    VARP quan_var = cast_expr->inputs().at(0);
    EXPRP quan_expr = quan_var->expr().first;
    if (!quan_expr->get() || quan_expr->get()->type() != OpType_FloatToInt8) {
        return false;
    }
    if (quan_expr->inputs().size() != 5) {
        return false;
    }
    // check other
    VARP other_var = quan_expr->inputs().at(0);
    EXPRP other_expr = other_var->expr().first;
    
    if (!other_expr->get()) {
        return false;
    }
    if (other_expr->get()->type() == OpType_ConvInt8 || other_expr->get()->type() == OpType_Cast || other_expr->get()->type() == OpType_Int8ToFloat || other_expr->get()->type() == OpType_FloatToInt8 || other_expr->get()->type() == OpType_Const || other_expr->get()->type() == OpType_DepthwiseConvInt8) {
        return false;
    }
    return true;
}
static VARP transformOtherToOther (EXPRP expr, int i) { // ohter->quant->cast->dequant->other => other->other
    auto dequant_var  = expr->inputs()[i];
    auto dequant_expr  = dequant_var->expr().first;
    auto cast_var = dequant_expr->inputs().at(0);
    auto cast_expr = cast_var->expr().first;
    auto quan_var = cast_expr->inputs().at(0);
    auto quan_expr = quan_var->expr().first;
    auto input_var = quan_expr->inputs().at(0);

    float scale = quan_expr->inputs().at(2)->readMap<float>()[0];
    float zero = quan_expr->inputs().at(3)->readMap<float>()[0];
    input_var->writeScaleMap(scale, zero);
    return input_var;
}
static VARP buildInputForMatmulInt8 (VARP input, VARP transposeA, VARP SqueezeA, int num_input) {
    auto transposeAType = transposeA->expr().first;
    auto transposeAInfo = transposeA->getInfo();
    if (!transposeAInfo) {
        return input;
    }
    if (transposeAInfo) {
        if (!transposeAInfo->dim.empty()) {
            return input;
        }
    }
    VARP newInput = std::move(input);
    auto format = MNN::MNN_DATA_FORMAT_NCHW;
    auto inputL = _Unsqueeze(_Scalar<int>(num_input), {0});
    inputL.fix(VARP::CONSTANT);
    VARP inputE;
    float needSqueezeA = SqueezeA->readMap<float>()[0];
    if (needSqueezeA != 0) {
        newInput = _Unsqueeze(newInput, {0});
    }
    auto rank = _Rank(newInput);
    auto inputShape = _Shape(newInput, NCHW);
    if (transposeA->readMap<float>()[0]) {
        inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
        newInput = _ReshapeF(newInput, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, inputE, _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
    } else {
        newInput = _ReshapeF(newInput, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
    }
    return newInput;
}

static EXPRP buildNewConvExpr(EXPRP oldConvExpr, VARP convInput, std::vector<bool> updateInfo = {}) {
    std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
    std::unique_ptr<OpT> oldConvOp(oldConvExpr->get()->UnPack());
    auto oldConvParams  = oldConvOp->main.AsConvolution2D();
    newConvInt8->common.reset(new MNN::Convolution2DCommonT);
    newConvInt8->common = std::move(oldConvParams->common);
    newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
    newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
    newConvInt8->quanParameter.reset(new IDSTQuanT);
    newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
    newConvInt8->bias = std::move(oldConvParams->bias);

    if (updateInfo.size() > 0) {
        newConvInt8->common->relu = updateInfo[0] ? true : false;
    }
    if (updateInfo.size() > 1) {
        newConvInt8->common->relu6 = updateInfo[1] ? true : false;
    }
    if (updateInfo.size() > 2) {
        newConvInt8->symmetricQuan->outputDataType = updateInfo[2] ? DataType_DT_FLOAT : DataType_DT_INT8;
    }
    float input_scale = newConvInt8->quanParameter->scaleIn;
    float input_zero = newConvInt8->symmetricQuan->zeroPoint;
    convInput->writeScaleMap(input_scale, input_zero);

    std::unique_ptr<OpT> conv_op(new OpT);
    conv_op->name = oldConvExpr->name();
    conv_op->type = oldConvOp->type;
    conv_op->main.type  = OpParameter_Convolution2D;
    conv_op->main.value = newConvInt8.release();

    auto new_conv_expr = Expr::create(conv_op.get(), {convInput});
    return new_conv_expr;
}

static auto gRegister = []() { // convInt8->(relu)->quant->cast->dequant->convInt8
    auto matchConvInt8ToConvInt8 = [](EXPRP expr) {
        // check convInt8
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_ConvInt8 && expr->get()->type() != OpType_DepthwiseConvInt8) {
            return false;
        }
        // check dequantize linear
        VARP dequant_var   = expr->inputs().at(0);
        EXPRP dequant_expr = dequant_var->expr().first;
        if (!dequant_expr->get() || dequant_expr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        if (dequant_expr->inputs().size() != 5) {
            return false;
        }
        // check cast
        VARP cast_var = dequant_expr->inputs().at(0);
        EXPRP cast_expr = cast_var->expr().first;
        if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
            return false;
        }
        // check quantize linear
        VARP quan_var = cast_expr->inputs().at(0);
        EXPRP quan_expr = quan_var->expr().first;
        if (!quan_expr->get() || quan_expr->get()->type() != OpType_FloatToInt8) {
            return false;
        }
        if (quan_expr->inputs().size() != 5) {
            return false;
        }
        // check convInt8
        VARP conv_var = quan_expr->inputs().at(0);
        EXPRP conv_expr = conv_var->expr().first;
        if (!conv_expr->get()) {
            return false;
        }
        if (conv_expr->get()->type() != OpType_PReLU && conv_expr->get()->type() != OpType_ReLU && conv_expr->get()->type() != OpType_ReLU6 && conv_expr->get()->type() != OpType_ConvInt8 && conv_expr->get()->type() != OpType_DepthwiseConvInt8) {
            return false;
        }
        if (conv_expr->get()->type() == OpType_PReLU || conv_expr->get()->type() == OpType_ReLU || conv_expr->get()->type() == OpType_ReLU6) {
            VARP conv_var_0 = conv_expr->inputs().at(0);
            EXPRP conv_expr_0 = conv_var_0->expr().first;
            if (!conv_expr_0->get()) {
                return false;
            }
            if (conv_expr_0->get()->type() != OpType_ConvInt8 && conv_expr_0->get()->type() != OpType_DepthwiseConvInt8) {
                return false;
            }
        }
        return true;
    };
    auto transformConvInt8ToConvInt8 = [](EXPRP expr) {
        auto dequant_var  = expr->inputs()[0];
        auto dequant_expr  = dequant_var->expr().first;
        auto cast_var = dequant_expr->inputs().at(0);
        auto cast_expr = cast_var->expr().first;
        auto quan_var = cast_expr->inputs().at(0);
        auto quan_expr = quan_var->expr().first;
        auto convInt8Input = quan_expr->inputs().at(0);
        /* conv params*/
        std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
        std::unique_ptr<OpT> oldConvOp(expr->get()->UnPack());
        auto oldConvParams  = oldConvOp->main.AsConvolution2D();
        float input_scale = oldConvParams->quanParameter->scaleIn;
        float input_zero  = oldConvParams->symmetricQuan->zeroPoint;
        /* check */
        auto conv_var = quan_expr->inputs().at(0);
        conv_var->writeScaleMap(input_scale, input_zero);
        EXPRP conv_expr = conv_var->expr().first;
        VARP first_conv_input_var = conv_expr->inputs().at(0);
        if (conv_expr->get()->type() == OpType_PReLU || conv_expr->get()->type() == OpType_ReLU || conv_expr->get()->type() == OpType_ReLU6) {
            auto relu_expr = conv_expr;
            bool relu_ = relu_expr->get()->type() == OpType_ReLU ? true: false;
            bool relu6_ = relu_expr->get()->type() == OpType_ReLU6 ? true: false;
            VARP conv_var_0 = relu_expr->inputs().at(0);
            conv_expr = conv_var_0->expr().first;
            first_conv_input_var = conv_expr->inputs().at(0);
            auto newFirstConvExpr = buildNewConvExpr(conv_expr, first_conv_input_var, {relu_, relu6_}); // write scale for first_conv_input_var
            Expr::replace(conv_expr, newFirstConvExpr);
            convInt8Input = Variable::create(conv_expr);
            conv_var = convInt8Input;
            conv_var->writeScaleMap(input_scale, input_zero);
        } else {
            auto newFirstConvExpr = buildNewConvExpr(conv_expr, first_conv_input_var); // Just write scale for first_conv_input_var, do not update conv info.
            Expr::replace(conv_expr, newFirstConvExpr);
            convInt8Input = Variable::create(conv_expr);
            conv_var = convInt8Input;
            conv_var->writeScaleMap(input_scale, input_zero);
        }
        if (conv_expr->inputs().size() == 5) {
             // Process matmul output
            auto config = Global<modelConfig>::Get();
            auto format = MNN::MNN_DATA_FORMAT_NCHW;
            if (config->model == modelConfig::TFLITE || config->model == modelConfig::TENSORFLOW) {
                format = MNN_DATA_FORMAT_NHWC;
            }
            // expr->inputs = {input, concat, needSqueezeA, needSqueezeB, transposeA}
            auto concat_var = conv_expr->inputs().at(1);
            bool needSqueezeA = conv_expr->inputs().at(2)->readMap<float>()[0] > 0.f;
            bool needSqueezeB = conv_expr->inputs().at(3)->readMap<float>()[0] > 0.f;

            auto output = _ConvertF(conv_var, format);
            output->writeScaleMap(input_scale, input_zero);
            
            VARP reshapeVar = _ReshapeF(output, concat_var, format);
            reshapeVar->writeScaleMap(input_scale, input_zero);
            if (needSqueezeA) {
                reshapeVar = _Squeeze(reshapeVar, {0});
            }
            if (needSqueezeB) {
                reshapeVar = _Squeeze(reshapeVar, {1});
            }
            reshapeVar->setName(conv_expr->outputName(0) + "__matmul_cvt_convInt8_reshape");
            Expr::replace(conv_expr, reshapeVar->expr().first);
            convInt8Input = reshapeVar;
            convInt8Input->writeScaleMap(input_scale, input_zero);
        }
        
        if (expr->inputs().size() == 5) {
            auto matmulop = expr->get();
            auto count_input = matmulop->main_as_Convolution2D()->common()->inputCount();
            convInt8Input = buildInputForMatmulInt8(convInt8Input, expr->inputs().at(4), expr->inputs().at(2), count_input);
            convInt8Input->writeScaleMap(input_scale, input_zero);
        }
        
        
        newConvInt8->common.reset(new MNN::Convolution2DCommonT);
        newConvInt8->common = std::move(oldConvParams->common);
        newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
        newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
        newConvInt8->quanParameter.reset(new IDSTQuanT);
        newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
        newConvInt8->bias = std::move(oldConvParams->bias);
        float scaleout = newConvInt8->quanParameter->scaleOut;
        float zeroout  = newConvInt8->symmetricQuan->outputZeroPoint;
        
        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = oldConvOp->type;
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = newConvInt8.release();
        

        auto new_conv_expr = Expr::create(conv_op.get(), {convInt8Input});
        if (expr->inputs().size() == 5) {
            new_conv_expr = Expr::create(conv_op.get(), {convInt8Input, expr->inputs()[1], expr->inputs()[2], expr->inputs()[3], expr->inputs()[4]});
        }
        new_conv_expr->setName(expr->name());
        auto new_conv_var = Variable::create(new_conv_expr);
        new_conv_var->writeScaleMap(scaleout, zeroout);
        Expr::replace(expr, new_conv_expr);
        return true;
        
    };
    
    auto matchOtherToConvInt8 = [](EXPRP expr) { // otherOp->quant->cast->dequant->convint8
        // check op type is convint8.
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_ConvInt8 && expr->get()->type() != OpType_DepthwiseConvInt8) {
            return false;
        }
        // check dequantize linear
        VARP dequant_var   = expr->inputs().at(0);
        EXPRP dequant_expr = dequant_var->expr().first;
        if (!dequant_expr->get() || dequant_expr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        if (dequant_expr->inputs().size() != 5) {
            return false;
        }
        // check cast
        VARP cast_var = dequant_expr->inputs().at(0);
        EXPRP cast_expr = cast_var->expr().first;
        if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
            return false;
        }
        // check quantize linear
        VARP quan_var = cast_expr->inputs().at(0);
        EXPRP quan_expr = quan_var->expr().first;
        if (!quan_expr->get() || (quan_expr->get()->type() != OpType_FloatToInt8 && quan_expr->get()->type() != OpType_ConvertTensor)) {
            return false;
        }
        if (quan_expr->get()->type() == OpType_FloatToInt8 && quan_expr->inputs().size() != 5) {
            return false;
        }
        // check other
        VARP other_var = quan_expr->inputs().at(0);
        EXPRP other_expr = other_var->expr().first;
        if (!other_expr->get()) {
            return true;
        }
        
        if (other_expr->get()->type() == OpType_ConvInt8 || other_expr->get()->type() == OpType_Cast || other_expr->get()->type() == OpType_Int8ToFloat || other_expr->get()->type() == OpType_FloatToInt8 || other_expr->get()->type() == OpType_Const || other_expr->get()->type() == OpType_DepthwiseConvInt8) {
            return false;
        }
        return true;
    };
    auto transformOtherToConvInt8 = [](EXPRP expr) {
        auto dequant_var  = expr->inputs()[0];
        auto dequant_expr  = dequant_var->expr().first;
        auto cast_var = dequant_expr->inputs().at(0);
        auto cast_expr = cast_var->expr().first;
        auto quan_var = cast_expr->inputs().at(0);
        auto quan_expr = quan_var->expr().first;
        auto convInt8Input = quan_expr->inputs().at(0);
        auto other_var = convInt8Input;
        if (expr->inputs().size() == 5) {
            // [input,concat,squeezeA,squeezeB,transposeA]
            auto matmulop = expr->get();
            auto count_input = matmulop->main_as_Convolution2D()->common()->inputCount();
            convInt8Input = buildInputForMatmulInt8(convInt8Input, expr->inputs().at(4), expr->inputs().at(2), count_input);
            convInt8Input->setName(expr->name() + "__matmul_converted_input");
        }
        
        std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
        std::unique_ptr<OpT> oldConvOp(expr->get()->UnPack());
        auto oldConvParams  = oldConvOp->main.AsConvolution2D();
        float input_scale   = oldConvParams->quanParameter->scaleIn;
        float output_scale  = oldConvParams->quanParameter->scaleOut;
        float input_zero    = static_cast<float>(oldConvParams->symmetricQuan->zeroPoint);
        float output_zero   = static_cast<float>(oldConvParams->symmetricQuan->outputZeroPoint);
        
        newConvInt8->common.reset(new MNN::Convolution2DCommonT);
        newConvInt8->common = std::move(oldConvParams->common);
        newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
        newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
        newConvInt8->bias = std::move(oldConvParams->bias);
        newConvInt8->quanParameter.reset(new IDSTQuanT);
        newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
        
        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = oldConvOp->type;
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = newConvInt8.release();
        
        other_var->writeScaleMap(input_scale, input_zero);
        convInt8Input->writeScaleMap(input_scale, input_zero);
        auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
        if (expr->inputs().size() == 5) {
            conv_expr = Expr::create(conv_op.get(), {convInt8Input, expr->inputs()[1], expr->inputs()[2], expr->inputs()[3], expr->inputs()[4]});
        }
        auto conv_var = Variable::create(conv_expr);
        conv_var->writeScaleMap(output_scale, output_zero);
        conv_expr->setName(expr->name());
        Expr::replace(expr, conv_expr);
        return true;
    };

    // X to otherOp
    auto matchXToOther = [](EXPRP expr) { // X->quant->cast->dequant->other
        // check op type not convint8.
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() == OpType_Const || expr->get()->type() == OpType_TrainableParam) {
            return false;
        }

        int inputs_size = static_cast<int32_t>(expr->inputs().size());
        for (int i = 0; i < inputs_size; ++i) {
            if (!matchConvInt8ToOther(expr, i) && !matchOtherToOther(expr, i)) {
                return false;
            }
        }
        return true;
    };
    auto transformXToOther = [](EXPRP expr) { // X->quant->cast->dequant->output_other => X->output_other
        int input_size = static_cast<int32_t>(expr->inputs().size());
        std::vector<VARP> new_inputs(input_size);
        for (int i = 0; i < input_size; ++i) {
            if (matchConvInt8ToOther(expr, i)) {
                VARP input_i = transformConvInt8ToOther(expr, i);
                new_inputs[i] = input_i;
            } else {
                VARP input_i = transformOtherToOther(expr, i);
                new_inputs[i] = input_i;
            }
        }

        // generate a new oher op.
        std::unique_ptr<OpT> oldOtherOp(expr->get()->UnPack());
        auto newop_expr = Expr::create(oldOtherOp.get(), new_inputs);
        Expr::replace(expr, newop_expr);
        return true;
        
    };

    // endding op->X
    auto matchXToEnd= [](EXPRP expr) { // otherOp->quant->cast->dequant->convint8
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() == OpType_Const || expr->get()->type() == OpType_TrainableParam) {
            return false;
        }
        // check op type is Int8ToFloat.
        if (expr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        // check op is the last op.
        if (expr->outputs().size() != 0) {
            return false;
        }
        
        // check cast
        VARP cast_var   = expr->inputs().at(0);
        EXPRP cast_expr = cast_var->expr().first;
        if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
            return false;
        }
        // check FloatToInt8
        VARP quan_var = cast_expr->inputs().at(0);
        EXPRP quan_expr = quan_var->expr().first;
        if (!quan_expr->get() || quan_expr->get()->type() != OpType_FloatToInt8) {
            return false;
        }
        // check X
        VARP X_var = quan_expr->inputs().at(0);
        EXPRP X_expr = X_var->expr().first;
        if (!X_expr->get() || X_expr->get()->type() == OpType_FloatToInt8 || X_expr->get()->type() == OpType_Const || X_expr->get()->type() == OpType_Cast || X_expr->get()->type() == OpType_Int8ToFloat) {
            return false;
        }
        if (X_expr->get()->type() == OpType_ConvInt8) {
            return true;
        }
        if (X_expr->get()->type() == OpType_Reshape) {
            auto convert_var = X_expr->inputs().at(0);
            auto convert_expr = convert_var->expr().first;
            if (convert_expr->get() && convert_expr->get()->type() == OpType_ConvertTensor) {
                auto convint8_var = convert_expr->inputs().at(0);
                auto convint8_expr = convint8_var->expr().first;
                if (convint8_expr->get() && convint8_expr->get()->type() == OpType_ConvInt8) {
                    return true;
                }
            }
            if (convert_expr->get() && convert_expr->get()->type() == OpType_ConvInt8) {
                return true;
            }
        }
        return true;
    };
    auto transformXToEnd = [](EXPRP expr) {
        auto cast_var  = expr->inputs()[0];
        auto cast_expr  = cast_var->expr().first;
        auto quan_var = cast_expr->inputs().at(0);
        auto quan_expr = quan_var->expr().first;
        auto X_var = quan_expr->inputs().at(0);
        auto X_expr = X_var->expr().first;

        bool convInt8End = X_expr->get()->type() == OpType_ConvInt8;

        if (convInt8End) {
            auto convInt8Input = X_expr->inputs().at(0);
            std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
            std::unique_ptr<OpT> oldConvOp(X_expr->get()->UnPack());
            auto oldConvParams  = oldConvOp->main.AsConvolution2D();
            newConvInt8->common.reset(new MNN::Convolution2DCommonT);
            newConvInt8->common = std::move(oldConvParams->common);
            newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
            newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
            newConvInt8->quanParameter.reset(new IDSTQuanT);
            //newConvInt8->symmetricQuan->outputDataType = DataType_DT_FLOAT; // If convInt8 is the last op, float value is the torch-fx model's output.
            newConvInt8->bias = std::move(oldConvParams->bias);
            newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
            
            float output_scale = newConvInt8->quanParameter->scaleOut;
            float output_zero = newConvInt8->symmetricQuan->outputZeroPoint;
            
            std::unique_ptr<OpT> conv_op(new OpT);
            conv_op->name = X_expr->name();
            conv_op->type = oldConvOp->type;
            conv_op->main.type  = OpParameter_Convolution2D;
            conv_op->main.value = newConvInt8.release();
            
            auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
            auto conv_var = Variable::create(conv_expr);
            conv_var->writeScaleMap(output_scale, output_zero);
            if (X_expr->inputs().size() == 5) {
                // Process matmul output
               auto config = Global<modelConfig>::Get();
               auto format = MNN::MNN_DATA_FORMAT_NCHW;
               if (config->model == modelConfig::TFLITE || config->model == modelConfig::TENSORFLOW) {
                   format = MNN_DATA_FORMAT_NHWC;
               }
                
                conv_var->setName(X_expr->outputName(0));
//                newconv_var->setName(conv_expr->outputName(0));
               // expr->inputs = {input, concat, needSqueezeA, needSqueezeB, transposeA}
               auto concat_var = X_expr->inputs().at(1);
               bool needSqueezeA = X_expr->inputs().at(2)->readMap<float>()[0] > 0.f;
               bool needSqueezeB = X_expr->inputs().at(3)->readMap<float>()[0] > 0.f;

               auto output = _ConvertF(conv_var, format);
                output->writeScaleMap(output_scale, output_zero);
               VARP reshapeVar = _ReshapeF(output, concat_var, format);
                reshapeVar->writeScaleMap(output_scale, output_zero);
               if (needSqueezeA) {
                   reshapeVar = _Squeeze(reshapeVar, {0});
                   reshapeVar->writeScaleMap(output_scale, output_zero);
               }
               if (needSqueezeB) {
                   reshapeVar = _Squeeze(reshapeVar, {1});
                   reshapeVar->writeScaleMap(output_scale, output_zero);
               }
               reshapeVar->setName(expr->name());
               Expr::replace(expr, reshapeVar->expr().first);
                return true;
           }
            conv_expr->setName(expr->name());
            Expr::replace(expr, conv_expr);
            return true;
        }
        float output_scale = quan_expr->inputs().at(2)->readMap<float>()[0];
        float output_zero  = quan_expr->inputs().at(3)->readMap<float>()[0];
        // directly return the op output.
        std::unique_ptr<OpT> oldOtherOp(X_expr->get()->UnPack());
        auto newop_expr = Expr::create(oldOtherOp.get(), X_expr->inputs());
        newop_expr->setName(expr->name());
        auto newop_var = Variable::create(newop_expr);
        newop_var->writeScaleMap(output_scale, output_zero);
        Expr::replace(expr, newop_expr);
        return true;
    };

   TemplateMerge::getInstance("Merge").insertTemplate("ConvInt8ToConvInt8", matchConvInt8ToConvInt8, transformConvInt8ToConvInt8,
                                                      PASS_PRIORITY_MIDDLE);
   TemplateMerge::getInstance("Merge").insertTemplate("OtherOpToConvInt8", matchOtherToConvInt8, transformOtherToConvInt8,
                                                      PASS_PRIORITY_MIDDLE);
   TemplateMerge::getInstance("Merge").insertTemplate("XToOtherOp", matchXToOther, transformXToOther,
                                                      PASS_PRIORITY_MIDDLE);
    TemplateMerge::getInstance("Merge").insertTemplate("XToEndOp", matchXToEnd, transformXToEnd,
                                                       PASS_PRIORITY_MIDDLE);
    return true;
}();

}
} // namespace MNN
