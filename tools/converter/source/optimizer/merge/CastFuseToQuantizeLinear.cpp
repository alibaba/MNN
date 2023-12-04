//
//  CastFuseToQuantizeLinear.cpp
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
        return false;
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_Cast) {
            return false;
        }
        
        VARP quantize_var = expr->inputs().at(0);
        EXPRP quantize_expr = quantize_var->expr().first;
        if (!quantize_expr->get() || quantize_expr->get()->type() != OpType_QuantizeLinear) {
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
        VARP quantize_var = expr->inputs().at(0);
        EXPRP quantize_expr = quantize_var->expr().first;
        
        VARP quantize_data_var = quantize_expr->inputs().at(0);
        VARP quantize_scale_var = quantize_expr->inputs().at(1);
        VARP quantize_zero_var = nullptr;
        if (quantize_expr->inputs().size() > 2) {
            quantize_zero_var = quantize_expr->inputs().at(2);
        }

        auto newParam = new QuantizeLinearT;
        std::unique_ptr<OpT> quantizeOp(quantize_expr->get()->UnPack());
        auto convParams  = quantizeOp->main.AsQuantizeLinear();
        newParam->scaleAxis = convParams->scaleAxis;
        newParam->scaleSize = convParams->scaleSize;
    
        std::unique_ptr<OpT> quantize_op(new OpT);
        quantize_op->name = expr->name();
        quantize_op->type = OpType_QuantizeLinear;
        quantize_op->main.type = OpParameter_QuantizeLinear;
        quantize_op->main.value = newParam;

        auto new_expr = Expr::create(quantize_op.get(), {quantize_data_var, quantize_scale_var, quantize_zero_var});
        new_expr->setName(quantize_expr->name());
        Expr::replace(expr, new_expr);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("CastFuseToQuantizeLinear", match, transform,
                                                       PASS_PRIORITY_MIDDLE);
    return true;
}();

}
} // namespace MNN
