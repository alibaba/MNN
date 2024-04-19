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
    
    if (!conv_expr->get() || (conv_expr->get()->type() != OpType_ConvInt8 && conv_expr->get()->type() != OpType_DepthwiseConvInt8)) {
        return false;
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

    // change old convInt8 to return a float value, which is input to expr;
    std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
    std::unique_ptr<OpT> oldConvOp(conv_expr->get()->UnPack());
    auto oldConvParams  = oldConvOp->main.AsConvolution2D();
    newConvInt8->common.reset(new MNN::Convolution2DCommonT);
    newConvInt8->common = std::move(oldConvParams->common);
    newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
    newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
    newConvInt8->symmetricQuan->outputDataType = MNN::DataType_DT_FLOAT;
    // newConvInt8->bias = std::move(oldConvParams->bias);
    // newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
    
    //Update newConvInt8 scale
    float outputScale = quan_expr->inputs().at(2)->readMap<float>()[0];
    int oc = static_cast<int32_t>(newConvInt8->symmetricQuan->scale.size());
    float* ptr = newConvInt8->symmetricQuan->scale.data();
    for (int i = 0; i < oc; ++i) {
        ptr[i] = ptr[i] * outputScale;
    }
    
    std::unique_ptr<OpT> conv_op(new OpT);
    conv_op->name = conv_expr->name();
    conv_op->type = oldConvOp->type;
    conv_op->main.type  = OpParameter_Convolution2D;
    conv_op->main.value = newConvInt8.release();

    auto newconv_expr = Expr::create(conv_op.get(), {convInt8Input});
    newconv_expr->setName(conv_expr->name());
    auto newconv_var = Variable::create(newconv_expr);
    newconv_var->setName(conv_expr->outputName(0));
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
    auto other_var = quan_expr->inputs().at(0);

    return other_var;
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
        if (expr->inputs().size() == 3) {
            auto matmulop = expr->get();
            auto count_input = matmulop->main_as_Convolution2D()->common()->inputCount();
            convInt8Input = buildInputForMatmulInt8(convInt8Input, expr->inputs().at(1), expr->inputs().at(2), count_input);
        }
        
        std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
        std::unique_ptr<OpT> oldConvOp(expr->get()->UnPack());
        auto oldConvParams  = oldConvOp->main.AsConvolution2D();
        newConvInt8->common.reset(new MNN::Convolution2DCommonT);
        newConvInt8->common = std::move(oldConvParams->common);
        newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
        newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
        // newConvInt8->bias = std::move(oldConvParams->bias);
        // newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
        
        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = oldConvOp->type;
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = newConvInt8.release();

        auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
        conv_expr->setName(expr->name());
//        auto conv_var = Variable::create(conv_expr);
//        conv_var->setName(expr->outputName(0));
        Expr::replace(expr, conv_expr);
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
        auto convInt8Input = quan_expr->inputs().at(1);
        if (expr->inputs().size() == 3) { // The convInt8 comes from matmul.
            auto matmulop = expr->get();
            auto count_input = matmulop->main_as_Convolution2D()->common()->inputCount();
            auto matmulInput = expr->inputs().at(0);
            convInt8Input = buildInputForMatmulInt8(convInt8Input, expr->inputs().at(1), expr->inputs().at(2), count_input);
        }
        
        std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
        std::unique_ptr<OpT> oldConvOp(expr->get()->UnPack());
        auto oldConvParams  = oldConvOp->main.AsConvolution2D();
        newConvInt8->common.reset(new MNN::Convolution2DCommonT);
        newConvInt8->common = std::move(oldConvParams->common);
        newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
        newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
        // newConvInt8->bias = std::move(oldConvParams->bias);
        // newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
        
        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = oldConvOp->type;
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = newConvInt8.release();

        auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
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
    auto transformXToOther = [](EXPRP expr) { // ohter->quant->cast->dequant->other => other->other
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
        bool hasReshape = X_expr->get()->type() == OpType_Reshape;
        if (X_expr->get()->type() == OpType_Reshape) {
            auto convert_var = X_expr->inputs().at(0);
            auto convert_expr = convert_var->expr().first;
            if (convert_expr->get() && convert_expr->get()->type() == OpType_ConvertTensor) {
                auto convint8_var = convert_expr->inputs().at(0);
                auto convint8_expr = convint8_var->expr().first;
                if (convint8_expr->get() && convint8_expr->get()->type() == OpType_ConvInt8) {
                    convInt8End = true;
                    X_expr = std::move(convint8_expr);
                }
            }
            if (convert_expr->get() && convert_expr->get()->type() == OpType_ConvInt8) {
                convInt8End = true;
                X_expr = std::move(convert_expr);
            }
        }

        if (convInt8End) {
            auto convInt8Input = X_expr->inputs().at(0);
            std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
            std::unique_ptr<OpT> oldConvOp(X_expr->get()->UnPack());
            auto oldConvParams  = oldConvOp->main.AsConvolution2D();
            newConvInt8->common.reset(new MNN::Convolution2DCommonT);
            newConvInt8->common = std::move(oldConvParams->common);
            newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
            newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
            newConvInt8->symmetricQuan->outputDataType = DataType_DT_FLOAT; // If convInt8 is the last op, float value is the torch-fx model's output.
            // newConvInt8->bias = std::move(oldConvParams->bias);
            // newConvInt8->quanParameter = std::move(oldConvParams->quanParameter);
            
            //Update convInt8 scale.
            float outputScale = quan_expr->inputs().at(2)->readMap<float>()[0];
            int oc = static_cast<int32_t>(newConvInt8->symmetricQuan->scale.size());
            float* ptr = newConvInt8->symmetricQuan->scale.data();
            for (int i = 0; i < oc; ++i) {
                ptr[i] = ptr[i] * outputScale;
            }
            
            std::unique_ptr<OpT> conv_op(new OpT);
            conv_op->name = X_expr->name();
            conv_op->type = oldConvOp->type;
            conv_op->main.type  = OpParameter_Convolution2D;
            conv_op->main.value = newConvInt8.release();
            
            auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
            conv_expr->setName(expr->name());
            
            if (hasReshape) {
                conv_expr->setName(X_expr->name());
                std::unique_ptr<OpT> reshapeOp(X_var->expr().first->get()->UnPack());
                auto new_reshape_expr = Expr::create(reshapeOp.get(), X_var->expr().first->inputs());
                new_reshape_expr->setName(expr->name());
                Expr::replace(expr, new_reshape_expr);
            }
            Expr::replace(X_expr, conv_expr);
            return true;
        }

        // directly return the op output.
        std::unique_ptr<OpT> oldOtherOp(X_expr->get()->UnPack());
        auto newop_expr = Expr::create(oldOtherOp.get(), X_expr->inputs());
        newop_expr->setName(expr->name());
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
