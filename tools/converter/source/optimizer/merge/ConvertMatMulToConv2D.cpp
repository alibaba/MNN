//
//  ConvertMatMulToConv2D.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>

#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"
#include "cli.hpp"
#include "../../common/Global.hpp"

namespace MNN {
namespace Express {

class ConvertMatMulToConv2D {
public:
    ConvertMatMulToConv2D();
};

ConvertMatMulToConv2D::ConvertMatMulToConv2D() {
    auto match = [this](EXPRP expr) -> bool {
        if (!expr->get()) {
            return false;
        }
        if (!expr->get() || expr->get()->type() != OpType_MatMul) {
            return false;
        }
        if (expr->inputs().size() != 2) {
            return false;
        }
        auto input = expr->inputs().at(0);
        auto info  = input->getInfo();
        if (nullptr == info) {
            // FIXME: Find better way to do optimization
            auto name = expr->name();
            if (name.find("dense") == std::string::npos) {
                return false;
            }
        } else {
            if (info->dim.size() != 2) {
                return false;
            }
        }
        // TODO(): Transpose?
        VARP weight = expr->inputs().at(1);
        if (weight->expr().first->outputs().size() > 1) {
            return false;
        }
        return helpers::IsConstant(weight->expr().first);
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto* param     = expr->get()->main_as_MatMul();
        bool transposeA = param->transposeA();
        bool transposeB = param->transposeB();

        VARP input  = expr->inputs().at(0);
        VARP weight = expr->inputs().at(1);
        if (transposeA) {
            input = _Transpose(input, {1, 0});
        }
        if (!transposeB) {
            weight = _Transpose(weight, {1, 0});
        }

        auto* info = weight->getInfo();
        if (!info || info->dim.size() != 2) {
            return false;
        }
        int num_input  = info->dim[1];
        int num_output = info->dim[0];

        std::unique_ptr<MNN::Convolution2DT> dense(new MNN::Convolution2DT);
        dense->bias.resize(num_output);
        std::fill(dense->bias.begin(), dense->bias.end(), 0.0f);
        dense->weight.resize(info->size);
        memcpy(dense->weight.data(), weight->readMap<float>(), info->size * sizeof(float));
        dense->common.reset(new Convolution2DCommonT);
        dense->common->inputCount  = num_input;
        dense->common->outputCount = num_output;

        std::unique_ptr<OpT> dense_op(new OpT);
        dense_op->name       = expr->name();
        dense_op->type       = OpType_Convolution;
        dense_op->main.type  = OpParameter_Convolution2D;
        dense_op->main.value = dense.release();

        auto gConverterConfig = Global<modelConfig>::Get();
        if (gConverterConfig->model == modelConfig::TENSORFLOW || gConverterConfig->model == modelConfig::TFLITE) {
            input = _Reshape(input, {1, 1, -1, num_input}, NHWC);
        } else {
            input = _Reshape(input, {-1, num_input, 1, 1}, NCHW);
        }

        EXPRP dense_expr = Expr::create(dense_op.get(), {input}, 1);
        VARP output = Variable::create(dense_expr);

        VARP reshape;
        if (gConverterConfig->model == modelConfig::TENSORFLOW) {
            reshape = _Reshape(output, {-1, num_output}, NHWC);
        } else {
            reshape = _Reshape(output, {-1, num_output}, NCHW);
        }
        // reshape->setName(expr->outputName(0));
        Expr::replace(expr, reshape->expr().first);

        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("ConvertMatMulToConv2D", match, fold);
}

static ConvertMatMulToConv2D g_convert_matmul_to_dense;

} // namespace Express
} // namespace MNN
