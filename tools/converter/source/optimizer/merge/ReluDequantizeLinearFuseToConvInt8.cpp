//
//  ReluDequantizeLinearFuseToConvInt8.cpp
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

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get() || expr->get()->type() != OpType_QuantizeLinear) {
            return false;
        }

        VARP relu6_var   = expr->inputs().at(0);
        EXPRP relu6_expr = relu6_var->expr().first;
        if (!relu6_expr->get() || relu6_expr->get()->type() != OpType_ReLU6) {
            return false;
        }

        VARP dequant_var   = relu6_expr->inputs().at(0);
        EXPRP dequant_expr = dequant_var->expr().first;
        if (!dequant_expr->get() || (dequant_expr->get()->type() != OpType_DequantizeLinear)) {
            return false;
        }
        
        VARP convert_var   = dequant_expr->inputs().at(0);
        EXPRP convert_expr = convert_var->expr().first;
        if (!convert_expr->get() || convert_expr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        VARP cast_var   = convert_expr->inputs().at(0);
        EXPRP cast_expr = cast_var->expr().first;
        if (!cast_expr->get() || cast_expr->get()->type() != OpType_Cast) {
            return false;
        }
        
        VARP convInt8_var   = cast_expr->inputs().at(0);
        EXPRP convInt8_expr = convInt8_var->expr().first;
        if (!convInt8_expr->get() || (convInt8_expr->get()->type() != OpType_ConvInt8 && convInt8_expr->get()->type() != OpType_DepthwiseConvInt8)) {
            return false;
        }
        return true;
    };

    auto transform = [](EXPRP expr) {
        auto gConverterConfig = Global<modelConfig>::Get();
        std::string compressFileName = gConverterConfig->compressionParamsFile;
        Compression::Pipeline proto;
        if (compressFileName != "") {
            std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            }
        }

        auto relu6_var = expr->inputs()[0];
        auto relu6_expr = relu6_var->expr().first;
        auto dequantize_var = relu6_expr->inputs()[0];
        auto dequantize_expr = dequantize_var->expr().first;
        auto convert_var = dequantize_expr->inputs()[0];
        auto convert_expr = convert_var->expr().first;
        auto cast_var =  convert_expr->inputs()[0];
        auto cast_expr = cast_var->expr().first;
//        auto quantize_var = cast_expr->inputs()[0];
//        auto quantize_expr = quantize_var->expr().first;
        auto convInt8_var = cast_expr->inputs()[0];
        auto convInt8_expr= convInt8_var->expr().first;
        auto convInt8Input = convInt8_expr->inputs()[0];
        
        std::unique_ptr<OpT> convInt8Op(convInt8_expr->get()->UnPack());

        std::unique_ptr<OpT> newconv_op(new OpT);
        newconv_op->name = expr->name();
        newconv_op = std::move(convInt8Op);
        
        auto convParams  = newconv_op->main.AsConvolution2D();
        auto& common     = convParams->common;
        common->relu6 = true;
        newconv_op->type = OpType_ConvInt8;
        bool is_depthwise = common->inputCount == common->outputCount && common->outputCount == common->group;
        if (is_depthwise) {
            newconv_op->type = OpType_DepthwiseConvInt8;
        }

        auto conv_expr = Expr::create(newconv_op.get(), {convInt8Input});
        conv_expr->setName(convInt8_expr->name());
        Expr::replace(expr, conv_expr);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("ReluDequantizeLinearFuseToConvInt8", match, transform,
                                                       PASS_PRIORITY_LOW);
    return true;
}();

}
} // namespace MNN
