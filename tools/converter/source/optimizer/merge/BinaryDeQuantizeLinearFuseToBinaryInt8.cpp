//
//  BinaryDeQuantizeLinearFuseToBinaryInt8.cpp
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
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_QuantizeLinear) {
            return false;
        }

        VARP binary_var   = expr->inputs().at(0);
        EXPRP binary_expr = binary_var->expr().first;
        if (!binary_expr->get() || binary_expr->get()->type() != OpType_BinaryOp) {
            return false;
        }

        VARP binary_input1_var   = binary_expr->inputs().at(0);
        EXPRP binary_input1_expr = binary_input1_var->expr().first;
        if (!binary_input1_expr->get() || (binary_input1_expr->get()->type() != OpType_DequantizeLinear)) {
            return false;
        }
        
        VARP binary_input2_var = binary_expr->inputs().at(1);
        EXPRP binary_input2_expr = binary_input2_var->expr().first;
        if (!binary_input2_expr->get() || (binary_input2_expr->get()->type() != OpType_DequantizeLinear)) {
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
        
        VARP binary_var = expr->inputs().at(0);
        EXPRP binary_expr = binary_var->expr().first;
        VARP input1_var = binary_expr->inputs().at(0); // Binary float inputs
        VARP input2_var = binary_expr->inputs().at(1);
        EXPRP input1_expr = input1_var->expr().first; // Dequantize layer
        EXPRP input2_expr = input2_var->expr().first;

        // BinaryInt8 input
        VARP input1_int8_var = input1_expr->inputs().at(0);
        VARP input2_int8_var = input2_expr->inputs().at(0);
        
        // Binary input quant parameters
        VARP input1_scale_var = input1_expr->inputs().at(1);
        VARP input1_zero_var = input1_expr->inputs().at(2);
        VARP input2_scale_var = input2_expr->inputs().at(1);
        VARP input2_zero_var = input2_expr->inputs().at(2);
        // Binary output quant parameters
        VARP output_scale_var = expr->inputs().at(1);
        VARP output_zeroz_var = expr->inputs().at(2);
        // Binary out var
        
        
        float scale1 = input1_scale_var->readMap<float>()[0];
        int8_t zero1 = input1_zero_var->readMap<int8_t>()[0];
        float scale2 = input2_scale_var->readMap<float>()[0];
        int8_t zero2 = input2_zero_var->readMap<int8_t>()[0];
        float scale_out = output_scale_var->readMap<float>()[0];
        int8_t zero_out = output_zeroz_var->readMap<int8_t>()[0];
        
        input1_int8_var->writeScaleMap(scale1, (float)zero1);
        input2_int8_var->writeScaleMap(scale2, (float)zero2);
        
        // BinaryOp expr
        std::unique_ptr<OpT> binaryOp(binary_expr->get()->UnPack());
        auto binaryParams = binaryOp->main.AsBinaryOp();

        std::unique_ptr<OpT> binary_op(new OpT);
        binary_op->name = expr->name();
        binary_op->type = binaryOp->type;
        binary_op->main.type = OpParameter_BinaryOp;
        
        auto binary = new MNN::BinaryOpT;
        binary->opType = binaryParams->opType;
        binary_op->main.value = binary;
        
        auto new_expr = Expr::create(binary_op.get(), {input1_int8_var, input2_int8_var});
        
        new_expr->setName(binary_expr->name());
        Expr::replace(expr, new_expr);
        // Add quant info to output node.
        expr->outputs()[0].lock()->inputs().at(0)->writeScaleMap(scale_out, zero_out);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("BinaryDeQuantizeLinearFuseToBinaryInt8", match, transform,
                                                       PASS_PRIORITY_MIDDLE);
    return true;
}();

}
} // namespace MNN
