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
#include "../../common/CommonUtils.hpp"

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
                if (input->expr().first->get() && input->expr().first->get()->type() == OpType_Reshape) {
                    input = input->expr().first->inputs()[0];
                }
            }

            if (input->expr().first->inputs().size() > 2) { // matmul has already had a bias or matmul comes from _MatMul_Int8
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
            // input shape may be change, don't fuse
            if (bias->expr().first->inputType() == VARP::InputType::INPUT) {
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
            auto config = Global<modelConfig>::Get();
            if(!config->convertMatmulToConv) {
                return false;
            }
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
            weight->expr().first->inside()->mCache = nullptr;
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
                bias->expr().first->inside()->mCache = nullptr;
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
            if (info->type.bits != 8 && info->type.bits != 32) {
                MNN_ERROR("Do not support weight bits=%d\n", (int)info->type.bits);
                return false;
            }
            bool convertToConvInt8 = info->type.bits == 8;
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
            const_cast<MNN::Express::Variable::Info*>(info)->syncSize();
            bool needSqueezeA = false;
            bool inputShapeUnknow = false;
            if (input->getInfo() != nullptr) {
                if (input->getInfo()->dim.size() <= 1) {
                    input = _Unsqueeze(input, {0});
                    needSqueezeA = true;
                }
            } else {
                inputShapeUnknow = true;
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
            
            const float* weightDataPtr = nullptr;
            const float* biasPtr = nullptr;
            weightDataPtr = weight->readMap<float>();
            if (convertToConvInt8) { // DynamicQuantizeLinear
                dense->symmetricQuan.reset(new QuantizedFloatParamT);
                dense->symmetricQuan->nbits = 8;
                std::vector<float> scale_1(num_output, 1.0);
                if (expr->inputs().size() == 3 && expr->inputs()[2]->getInfo()) {
                    MNN_ASSERT(expr->inputs()[2]->getInfo()->dim[0] == num_output);
                    if (!helpers::IsConstant(expr->inputs()[2]->expr().first) || !expr->inputs()[2]->readMap<float>()) {
                        MNN_ERROR("matmul convert to conv2d fail: In dynamic quant for Matmul, weight scale must be constant.");
                        return false;
                    }
                    ::memcpy(scale_1.data(), expr->inputs()[2]->readMap<float>(), num_output * sizeof(float));
                }
                dense->symmetricQuan->clampMin = -1;
                dense->symmetricQuan->clampMax = -1;
                dense->symmetricQuan->zeroPoint = 0;
                dense->symmetricQuan->outputZeroPoint = 0;
                dense->symmetricQuan->scale = std::move(scale_1);
                dense->symmetricQuan->outputDataType = DataType_DT_FLOAT;
            }
            
            if (weightDataPtr) {
                // Weight is a const node.
                if (false == convertToConvInt8) {
                    dense->bias.resize(num_output);
                    if (expr->inputs().size() == 3) { // bias is a const node.
                        auto bias = expr->inputs()[2];
                        biasPtr = bias->readMap<float>();
                        ::memcpy(dense->bias.data(), biasPtr, num_output * sizeof(float));
                        // Release compute cache for save memory
                        bias->expr().first->inside()->mCache = nullptr;
                    } else if (param->bias() && param->bias()->size() == num_output) {
                        ::memcpy(dense->bias.data(), param->bias()->data(), num_output * sizeof(float));
                    } else {
                        std::fill(dense->bias.begin(), dense->bias.end(), 0.0f);
                    }
                    if (config->externalFile && info->size >= config->externalTreshold) {
                        dense->external.emplace_back(config->externalOffset);
                        int64_t size = info->size * sizeof(float);
                        config->externalFile->write(reinterpret_cast<const char*>(weightDataPtr), size);
                        config->externalOffset += size;
                        dense->external.emplace_back(size);
                        size = dense->bias.size() * sizeof(float);
                        config->externalFile->write(reinterpret_cast<const char*>(dense->bias.data()), size);
                        config->externalOffset += size;
                        dense->external.emplace_back(size);
                        dense->bias.clear();
                        std::vector<float> empty;
                        dense->bias.swap(empty);
                    } else {
                        dense->weight.resize(info->size);
                        memcpy(dense->weight.data(), weightDataPtr, info->size * sizeof(float));
                    }
                } else {
                    dense->symmetricQuan->weight.resize(info->size);
                    memcpy(dense->symmetricQuan->weight.data(), weightDataPtr, info->size * sizeof(int8_t));
                    dense->symmetricQuan->bias.resize(num_output, 0);
                }
                // Release compute cache for save memory
                weight->expr().first->inside()->mCache = nullptr;
            }
            
            dense->common.reset(new Convolution2DCommonT);
            dense->common->inputCount  = num_input;
            dense->common->outputCount = num_output;

            std::unique_ptr<OpT> dense_op(new OpT);
            if (convertToConvInt8) {
                dense_op->type = OpType_ConvInt8;
            } else {
                dense_op->type       = OpType_Convolution;
            }
            dense_op->main.type  = OpParameter_Convolution2D;
            dense_op->main.value = dense.release();
            auto rank = _Rank(input);
            auto inputShape = _Shape(input, NCHW);
            auto inputL = _Unsqueeze(_Scalar<int>(num_input), {0});
            inputL.fix(VARP::CONSTANT);
            auto outputH = _Unsqueeze(_Scalar<int>(num_output), {0});
            outputH.fix(VARP::CONSTANT);
            VARP remainBegin;
            VARP inputELength;
            if (inputShapeUnknow) {
                remainBegin = _Minimum(_Scalar<int>(2), rank);
                inputELength = remainBegin - _Scalar<int>(1);
            } else {
                remainBegin = _Scalar<int>(2);
                inputELength = _Scalar<int>(1);
            }
            auto rankRemain = _Unsqueeze(rank - remainBegin, {0});
            VARP inputE;
            VARP inputRemain = _Slice(inputShape, _Unsqueeze(_Scalar<int>(0), {0}), rankRemain);
            if (transposeA) {
                inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
                if (format == MNN_DATA_FORMAT_NHWC) {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputE, _Unsqueeze(_Scalar<int>(1), {0}), inputL}, 0), format);
                } else {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, inputE, _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                }
            } else {
                inputE = _Slice(inputShape, rankRemain, _Unsqueeze(inputELength, {0}));
                if (format == MNN_DATA_FORMAT_NHWC) {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}), inputL}, 0), format);
                } else {
                    input = _ReshapeF(input, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                }
            }
            EXPRP dense_expr;
            if (convertToConvInt8) {
                dense_expr = Expr::create(dense_op.get(), {input}, 1);
            } else if (weightDataPtr) {
                dense_expr = Expr::create(dense_op.get(), {input}, 1);
            } else {
                if (expr->inputs().size() > 2) {
                    dense_expr = Expr::create(dense_op.get(), {input, weight}, 1);
                } else {
                    dense_expr = Expr::create(dense_op.get(), {input, weight, expr->inputs()[2]}, 1);
                }
            }
            
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
    // Directly convert matmul with quantize linear to convint8
    {
        auto fold = [this](EXPRP expr) -> bool {
            auto config = Global<modelConfig>::Get();
            auto version = config->targetVersion;
            if (version < 1.1f) {
                // For target version < 1.1 , don't support matmul + bias fuse
                return false;
            }
            if (!expr->get()) {
                return false;
            }
            if (expr->get()->type() != OpType_BinaryOp && expr->get()->type() != OpType_MatMul) {
                return false;
            }
            if (expr->get()->type() != OpType_BinaryOp && expr->get()->main_as_BinaryOp() && expr->get()->main_as_BinaryOp()->opType() != BinaryOpOperation_ADD) {
                return false;
            }
            VARP matmul_var;
            EXPRP matmul_expr;
            VARP bias_var = nullptr;
            bool matmulAddBias = true;
            // First, get matmul_expr
            if (expr->get()->type() == OpType_BinaryOp) {
                matmul_var = expr->inputs().at(0);
                matmul_expr = matmul_var->expr().first;
                if (matmul_expr->get() == nullptr) {
                    return false;
                }
                if (expr->inputs().size() > 1) {
                    bias_var = expr->inputs().at(1);
                    if (matmul_var->expr().first->get() == nullptr || matmul_var->expr().first->get()->type() == OpType_Const) {
                        bias_var = expr->inputs()[0];
                        matmul_var = expr->inputs()[1];
                        matmul_expr = matmul_var->expr().first;
                    }
                }
                if (bias_var->getInfo() == nullptr) {
                    return false;
                }
                if (bias_var->expr().first->inputType() == VARP::InputType::INPUT) {
                    return false;
                }
                // conv -> reshape -> convert -> add
                if (matmul_expr->get() && matmul_expr->get()->type() == OpType_ConvertTensor) {
                    matmul_var = matmul_expr->inputs()[0];
                    matmul_expr = matmul_var->expr().first;
                    if (matmul_expr->get() && matmul_expr->get()->type() == OpType_Reshape) {
                        matmul_var = matmul_expr->inputs()[0];
                        matmul_expr = matmul_var->expr().first;
                    }
                }
                if (matmul_expr->inputs().size() != 8 && matmul_expr->inputs().size() != 9) { // matmul 8 input: (x,y,x_scale,x_zero,y_scale,y_zero,out_scale,out_zero,bias
                    return false;
                }
                if (matmul_var->linkNumber() > 1) {
                    return false;
                }
            } else {
                matmul_expr = std::move(expr);
                if (matmul_expr->inputs().size() != 8 && matmul_expr->inputs().size() != 9) {
                    return false;
                }
                matmulAddBias = false;
            } // finish getting matmul_expr

            // Second, get matmul parameters
            auto matmulOp = matmul_expr->get();
            auto matmul_input = matmul_expr->inputs().at(0);
            auto input = matmul_expr->inputs().at(0);
            auto weight = matmul_expr->inputs()[1];
            auto weightInfo = weight->getInfo();
            if (nullptr == matmulOp || matmulOp->type() != OpType_MatMul) {
                return false;
            }
            if (nullptr == weightInfo || weightInfo->dim.size() != 2 || weightInfo->type.bits != 8) {
                return false;
            }
            // Compute number_output
            auto transposeB = matmulOp->main_as_MatMul()->transposeB();
            auto transposeA = matmulOp->main_as_MatMul()->transposeA();
            auto needSqueezeB = false;
            auto needSqueezeA = false;
            if (weightInfo->dim.size() == 1) {
                weight = _Unsqueeze(weight, {1});
                needSqueezeB = true;
            }
            if (!transposeB) {
                weight = _Transpose(weight, {1, 0});
            }
            weightInfo = weight->getInfo();
            if (input->getInfo() && input->getInfo()->dim.size() <= 1) {
                input = _Unsqueeze(input, {0});
                needSqueezeA = true;
            }
            if (needSqueezeA && needSqueezeB) {
                MNN_ERROR("Invalid MatMul for one-dimension A and B\n");
                return false;
            }
            auto format = MNN::MNN_DATA_FORMAT_NCHW;
            if (config->model == modelConfig::TFLITE || config->model == modelConfig::TENSORFLOW) {
                format = MNN_DATA_FORMAT_NHWC;
            }
            int numberOutput = weightInfo->dim[0]; // need to check
            int numberInput = weightInfo->dim[1];

            if (matmulAddBias) {
                auto biasInfo = bias_var->getInfo();
                if (biasInfo->size != numberOutput) {
                    return false;
                }
            }
            auto matmulInput = matmul_expr->inputs().at(0);
            auto inputScale  = matmul_expr->inputs().at(2);
            auto inputZero   = matmul_expr->inputs().at(3);
            auto weightScale = matmul_expr->inputs().at(4);
            auto weightZero  = matmul_expr->inputs().at(5);
            auto outputScale = matmul_expr->inputs().at(6);
            auto outputZero  = matmul_expr->inputs().at(7);
            
            float input_zero          = inputZero->readMap<float>()[0];
            float input_scale         = inputScale->readMap<float>()[0];
            const float* weight_scale = weightScale->readMap<float>();
            const float* weight_zero  = weightZero->readMap<float>();
            float output_scale        = outputScale->readMap<float>()[0];
            int output_zero           = static_cast<float>(outputZero->readMap<float>()[0]);
            // Convint8
            std::unique_ptr<Convolution2DT> dense(new MNN::Convolution2DT);
            dense->common.reset(new MNN::Convolution2DCommonT);
            dense->common->inputCount  = numberInput;
            dense->common->outputCount = numberOutput;
            // quant info
            dense->symmetricQuan.reset(new QuantizedFloatParamT);
            dense->symmetricQuan->nbits = 8;
            dense->symmetricQuan->clampMin = -128;
            dense->symmetricQuan->clampMax = 127;
            dense->symmetricQuan->zeroPoint = static_cast<int8_t>(input_zero);
            dense->symmetricQuan->outputZeroPoint = static_cast<int8_t>(output_zero);
            // quantParameter
            dense->quanParameter.reset(new IDSTQuanT);
            dense->quanParameter->scaleIn = input_scale;
            dense->quanParameter->scaleOut = output_scale;
            dense->quanParameter->type = 4;
            dense->quanParameter->aMin = -128;
            dense->quanParameter->readType = numberOutput;
            dense->quanParameter->quantScale = 1.0f;
            dense->quanParameter->buffer.resize(weightInfo->size);
            ::memcpy(dense->quanParameter->buffer.data(), weight->readMap<int8_t>(), weightInfo->size * sizeof(int8_t));
            dense->bias.resize(numberOutput, 0);
            // quan alpha
            dense->quanParameter->alpha.resize(2 * numberOutput);
            for (int i = 0; i < numberOutput; ++i) {
                dense->quanParameter->alpha[2 * i] = (-1)*(weight_zero[i] + 128) * weight_scale[i];
                dense->quanParameter->alpha[2 * i + 1] = weight_scale[i];
            }

            if (matmul_expr->inputs().size() == 9) {
                bias_var = matmul_expr->inputs().at(8);
                auto bias_ptr = bias_var->readMap<float>();
                memcpy(dense->bias.data(), bias_ptr, sizeof(int32_t) * numberOutput);
            }

            // Third, build convint8 op
            std::unique_ptr<OpT> dense_op(new OpT);
            dense_op->type = OpType_ConvInt8;
            dense_op->main.type  = OpParameter_Convolution2D;
            dense_op->main.value = dense.release();
            auto rank = _Rank(input);
            auto inputShape = _Shape(input, NCHW);
            auto inputL = _Unsqueeze(_Scalar<int>(numberInput), {0});
            inputL.fix(VARP::CONSTANT);
            auto outputH = _Unsqueeze(_Scalar<int>(numberOutput), {0});
            outputH.fix(VARP::CONSTANT);
            VARP inputE;
            VARP inputRemain = _StridedSlice(inputShape, _Unsqueeze(_Scalar<int>(0), {0}), _Unsqueeze(rank - _Scalar<int>(2), {0}), _Unsqueeze(_Scalar<int>(1), {0}), 0, 0, 0, 0, 0);
            if (transposeA) {
                inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
            } else {
                inputE = _Slice(inputShape, _Unsqueeze(rank - _Scalar<int>(2), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
            }
            if (config->externalFile && weightInfo->size >= config->externalTreshold) {
                RemoveAndStoreParam(dense_op, config->externalFile, config->externalOffset);
            }
            float ta = 0, sa = 0, sqzb = 0;
            if (transposeA) {
                ta = 1.0f;
            }
            if (needSqueezeA) {
                sa = 1.0f;
            }
            if (needSqueezeB) {
                sqzb = 1.0f;
            }
            EXPRP dense_expr = Expr::create(dense_op.get(), {matmul_input, _Concat({inputRemain, inputE, outputH}, 0), _Const(sa), _Const(sqzb), _Const(ta)},  1);
            VARP output = Variable::create(dense_expr);
            // output->setName(matmul_expr->outputName(0));
            dense_expr->setName(matmul_expr->outputName(0) + "__matmul_converted");
            Expr::replace(matmul_expr, dense_expr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("MatMulInt8ToConvInt8", fold, PASS_PRIORITY_HIGH);
    }
}

static ConvertMatMulToConv2D g_convert_matmul_to_dense;

} // namespace Express
} // namespace MNN
