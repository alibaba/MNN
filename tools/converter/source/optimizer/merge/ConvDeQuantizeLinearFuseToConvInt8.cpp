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

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_QuantizeLinear) {
            return false;
        }

        VARP conv_var   = expr->inputs().at(0);
        EXPRP conv_expr = conv_var->expr().first;
        if (!conv_expr->get() || (conv_expr->get()->type() != OpType_ConvInt8 &&
                                  conv_expr->get()->type() != OpType_DepthwiseConvInt8)) {
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
        auto convInt8Varp  = expr->inputs()[0];
        auto convInt8Expr  = convInt8Varp->expr().first;
        auto convInt8Input = convInt8Expr->inputs()[0];
        
        std::unique_ptr<Convolution2DT> newConvInt8(new MNN::Convolution2DT);
        std::unique_ptr<OpT> oldConvOp(convInt8Expr->get()->UnPack());
        auto oldConvParams  = oldConvOp->main.AsConvolution2D();
        newConvInt8->common.reset(new MNN::Convolution2DCommonT);
        newConvInt8->common = std::move(oldConvParams->common);
        newConvInt8->symmetricQuan.reset(new QuantizedFloatParamT);
        newConvInt8->symmetricQuan = std::move(oldConvParams->symmetricQuan);
        
        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = oldConvOp->type;
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = newConvInt8.release();

        auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
        conv_expr->setName(convInt8Expr->name());
        auto conv_var = Variable::create(conv_expr);
        conv_var->setName(convInt8Expr->outputName(0));
        Expr::replace(expr, conv_expr);
        return true;
        
    };

    TemplateMerge::getInstance("Merge").insertTemplate("ConvQuantizeDequantizeLinearFuseToConvInt8", match, transform,
                                                       PASS_PRIORITY_MIDDLE);
    return true;
}();

}
} // namespace MNN
