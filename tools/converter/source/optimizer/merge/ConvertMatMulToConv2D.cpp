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

namespace MNN {
namespace Express {

class ConvertMatMulToConv2D {
public:
    ConvertMatMulToConv2D();
};
static bool checkInputInfo(const std::string& exprName, const Variable::Info* info, const modelConfig* config) {
    if (nullptr == info) {
        if (config->optimizeLevel < 1) {
            return false;
        }
        if (config->optimizeLevel == 1) {
            // Namely dense op can be treat
            if (exprName.find("dense") == std::string::npos) {
                return false;
            }
        }
    } else {
        if (info->dim.size() != 2) {
            return false;
        }
    }
    return true;
}

ConvertMatMulToConv2D::ConvertMatMulToConv2D() {
    // Fuse MatMul + Bias
    {
        auto fold = [this](EXPRP expr) -> bool {
            auto config = Global<modelConfig>::Get();
            auto version = config->targetVersion;
            if (version < 1.1f) {
                // For target version < 1.1 , don't support matmul + bias fuse
                return false;
            }
            if (!expr->get() || expr->get()->type() != OpType_BinaryOp) {
                return false;
            }
            if (expr->get()->main_as_BinaryOp()->opType() != BinaryOpOperation_ADD) {
                return false;
            }
            auto input = expr->inputs()[0];
            auto bias = expr->inputs()[1];
            auto matmulOp = input->expr().first->get();
            if (nullptr == matmulOp || matmulOp->type() != OpType_MatMul || input->linkNumber() > 1) {
                return false;
            }
            // Compute number_output
            auto transposeB = matmulOp->main_as_MatMul()->transposeB();
            auto weight = input->expr().first->inputs()[1];
            auto weightInfo = weight->getInfo();
            if (nullptr == weightInfo || weightInfo->dim.size() != 2) {
                return false;
            }
            int numberOutput = weightInfo->dim[1];
            if (transposeB) {
                numberOutput = weightInfo->dim[0];
            }
            auto biasInfo = bias->getInfo();
            if (nullptr == biasInfo) {
                return false;
            }
            if (biasInfo->size != numberOutput) {
                return false;
            }
            auto matmulInput = input->expr().first->inputs().at(0);
            auto info = matmulInput->getInfo();
            if (!checkInputInfo(expr->name(), info, config)) {
                return false;
            }
            auto newExpr = Expr::create(input->expr().first->extra(), {matmulInput, weight, bias});
            newExpr->setName(expr->name());
            Expr::replace(expr, newExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FuseMatMulBias", fold, PASS_PRIORITY_HIGH);
    }
    // ConvertMatMulToConv2D
    {
        auto match = [this](EXPRP expr) -> bool {
            if (!expr->get()) {
                return false;
            }
            if (!expr->get() || expr->get()->type() != OpType_MatMul) {
                return false;
            }
            if (expr->inputs().size() != 2 && expr->inputs().size() != 3) {
                return false;
            }
            auto input = expr->inputs().at(0);
            auto config = Global<modelConfig>::Get();
            auto info  = input->getInfo();
            if (!checkInputInfo(expr->name(), info, config)) {
                return false;
            }
            // TODO(): Transpose?
            VARP weight = expr->inputs().at(1);
            if (weight->expr().first->outputs().size() > 1) {
                return false;
            }
            if (weight->readMap<float>() == nullptr) {
                // Not const
                return false;
            }
            if (expr->inputs().size() == 3) {
                auto bias = expr->inputs()[2];
                if (bias->readMap<float>() == nullptr) {
                    // Bias Not const
                    return false;
                }
            }
            return true;
        };

        auto fold = [this](EXPRP expr) -> bool {
            auto* param     = expr->get()->main_as_MatMul();
            bool transposeA = param->transposeA();
            bool transposeB = param->transposeB();

            VARP input  = expr->inputs().at(0);
            VARP weight = expr->inputs().at(1);
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
            if (expr->inputs().size() == 3) {
                auto bias = expr->inputs()[2];
                auto biasPtr = bias->readMap<float>();
                ::memcpy(dense->bias.data(), biasPtr, num_output * sizeof(float));
            } else {
                std::fill(dense->bias.begin(), dense->bias.end(), 0.0f);
            }
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
                if (transposeA) {
                    input = _Transpose(input, {1, 0});
                }
                input = _Reshape(input, {1, 1, -1, num_input}, NHWC);
            } else {
                if (!transposeA) {
                    input = _Transpose(input, {1, 0});
                }
                input = _Reshape(input, {1, num_input, 1, -1}, NCHW);
            }

            EXPRP dense_expr = Expr::create(dense_op.get(), {input}, 1);
            VARP output = Variable::create(dense_expr);

            VARP reshape;
            if (gConverterConfig->model == modelConfig::TENSORFLOW || gConverterConfig->model == modelConfig::TFLITE) {
                reshape = _Reshape(output, {-1, num_output}, NHWC);
            } else {
                reshape = _Reshape(output, {num_output, -1}, NCHW);
                reshape = _Transpose(reshape, {1, 0});
            }
            // reshape->setName(expr->outputName(0));
            Expr::replace(expr, reshape->expr().first);

            return true /*modified*/;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("ConvertMatMulToConv2D", match, fold, PASS_PRIORITY_MIDDLE);
    }
}

static ConvertMatMulToConv2D g_convert_matmul_to_dense;

} // namespace Express
} // namespace MNN
