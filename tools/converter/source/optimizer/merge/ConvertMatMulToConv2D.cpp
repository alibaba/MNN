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
#include "Utils.hpp"
#include "cli.hpp"

namespace MNN {
namespace Express {

class ConvertMatMulToConv2D {
public:
    ConvertMatMulToConv2D();
};
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
            if (input->expr().first->get() == nullptr || input->expr().first->get()->type() == OpType_Const) {
                bias = expr->inputs()[0];
                input = expr->inputs()[1];
            }
            if (input->expr().first->get() == nullptr) {
                return false;
            }

            // conv -> reshape -> convert -> add
            if (input->expr().first->get()->type() == OpType_ConvertTensor) {
                input = input->expr().first->inputs()[0];
                if (input->expr().first->get()->type() == OpType_Reshape) {
                    input = input->expr().first->inputs()[0];
                }
            }

            if (input->expr().first->inputs().size() > 2) { // matmul has already had a bias.
                return false;
            }

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
            // TODO(): Transpose?
            VARP weight = expr->inputs().at(1);
            if (weight->readMap<float>() == nullptr) {
                // Not const
                // Release compute cache for save memory
                weight->expr().first->inside()->mCache = nullptr;
                return false;
            }
            auto config = Global<modelConfig>::Get();
            int limitNumber = 4;
            if (config->optimizePrefer == 1) {
                // Smallest
                limitNumber = 1;
            } else if (config->optimizePrefer == 2) {
                // Fastest
                limitNumber = 100;
            }
            if (weight->linkNumber() > limitNumber) {
                return false;
            }
            if (weight->linkNumber() > 1) {
                static bool gPrint = false;
                if (!gPrint) {
                    MNN_PRINT("Convert MatMul Convolution use shared const B inputs, may increase the model size\n");
                    gPrint = true;
                }
            }
            if (expr->inputs().size() == 3) {
                auto bias = expr->inputs()[2];
                if (bias->readMap<float>() == nullptr) {
                    // Bias Not const
                    // Release compute cache for save memory
                    bias->expr().first->inside()->mCache = nullptr;
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
            auto* info = weight->getInfo();
            if (!info || info->dim.size() > 2) {
                return false;
            }
            if (info->dim.size() == 0) {
                return false;
            }
            bool needSqueezeB = false;
            if (info->dim.size() == 1) {
                weight = _Unsqueeze(weight, {1});
                needSqueezeB = true;
            }
            if (!transposeB) {
                weight = _Transpose(weight, {1, 0});
            }
            // Recompute weight info
            info = weight->getInfo();
            bool needSqueezeA = false;
            if (input->getInfo() != nullptr) {
                if (input->getInfo()->dim.size() <= 1) {
                    input = _Unsqueeze(input, {0});
                    needSqueezeA = true;
                }
            }
            if (needSqueezeA && needSqueezeB) {
                MNN_ERROR("Invalid MatMul for one-dimension A and B\n");
                return false;
            }
            auto config = Global<modelConfig>::Get();
            auto format = MNN::MNN_DATA_FORMAT_NCHW;
            if (config->model == modelConfig::TFLITE || config->model == modelConfig::TENSORFLOW) {
                format = MNN_DATA_FORMAT_NHWC;
            }

            int num_input  = info->dim[1];
            int num_output = info->dim[0];

            std::unique_ptr<MNN::Convolution2DT> dense(new MNN::Convolution2DT);
            dense->bias.resize(num_output);
            if (expr->inputs().size() == 3) {
                auto bias = expr->inputs()[2];
                auto biasPtr = bias->readMap<float>();
                ::memcpy(dense->bias.data(), biasPtr, num_output * sizeof(float));
                // Release compute cache for save memory
                bias->expr().first->inside()->mCache = nullptr;
            } else if (param->bias() && param->bias()->size() == num_output) {
                ::memcpy(dense->bias.data(), param->bias()->data(), num_output * sizeof(float));
            } else {
                std::fill(dense->bias.begin(), dense->bias.end(), 0.0f);
            }
            dense->weight.resize(info->size);
            memcpy(dense->weight.data(), weight->readMap<float>(), info->size * sizeof(float));
            // Release compute cache for save memory
            weight->expr().first->inside()->mCache = nullptr;
            dense->common.reset(new Convolution2DCommonT);
            dense->common->inputCount  = num_input;
            dense->common->outputCount = num_output;

            std::unique_ptr<OpT> dense_op(new OpT);
            dense_op->type       = OpType_Convolution;
            dense_op->main.type  = OpParameter_Convolution2D;
            dense_op->main.value = dense.release();
            auto rank = _Rank(input);
            auto inputShape = _Shape(input, NCHW);
            auto inputL = _Unsqueeze(_Scalar<int>(num_input), {0});
            inputL.fix(VARP::CONSTANT);
            auto outputH = _Unsqueeze(_Scalar<int>(num_output), {0});
            outputH.fix(VARP::CONSTANT);
            VARP inputE;
            VARP inputRemain = _StridedSlice(inputShape, _Unsqueeze(_Scalar<int>(0), {0}), _Unsqueeze(rank - _Scalar<int>(2), {0}), _Unsqueeze(_Scalar<int>(1), {0}), 0, 0, 0, 0, 0);
            if (transposeA) {
                inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
                if (format == MNN_DATA_FORMAT_NHWC) {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputE, _Unsqueeze(_Scalar<int>(1), {0}), inputL}, 0), format);
                } else {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, inputE, _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                }
            } else {
                inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(2), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
                if (format == MNN_DATA_FORMAT_NHWC) {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}), inputL}, 0), format);
                } else {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                }
            }
            EXPRP dense_expr = Expr::create(dense_op.get(), {input}, 1);
            VARP output = Variable::create(dense_expr);
            output->setName(expr->outputName(0) + "__matmul_converted");
            //MNN_PRINT("%d\n", output->getInfo()->order);
            output = _ConvertF(output, format);
            VARP reshapeVar = _ReshapeF(output, _Concat({inputRemain, inputE, outputH}, 0), format);
            if (needSqueezeA) {
                reshapeVar = _Squeeze(reshapeVar, {0});
            }
            if (needSqueezeB) {
                reshapeVar = _Squeeze(reshapeVar, {1});
            }
            reshapeVar->setName(expr->outputName(0));
            Expr::replace(expr, reshapeVar->expr().first);

            return true /*modified*/;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("ConvertMatMulToConv2D", match, fold, PASS_PRIORITY_MIDDLE);
    }
}

static ConvertMatMulToConv2D g_convert_matmul_to_dense;

} // namespace Express
} // namespace MNN
