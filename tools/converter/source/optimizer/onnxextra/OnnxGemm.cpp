//
//  OnnxGemm.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "../merge/MergeHelpers.hpp"

namespace MNN {
namespace Express {
static VARP _MatMul_Int8(VARP a, VARP b, bool tranposeA, bool tranposeB, VARP scaleA, VARP zeroA, VARP scaleB, VARP zeroB, VARP ScaleOut, VARP ScaleZero, VARP bias = nullptr) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                   = OpParameter_MatMul;
    op->type                        = OpType_MatMul;
    op->main.value                  = new MatMulT;
    op->main.AsMatMul()->transposeA = tranposeA;
    op->main.AsMatMul()->transposeB = tranposeB;
    return (Variable::create(Expr::create(op.get(), {a, b, scaleA, zeroA, scaleB, zeroB, ScaleOut, ScaleZero, bias})));
}

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

class OnnxGemmTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        bool transA = false;
        bool transB = false;
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool op8 = false;

        auto extraParam    = op->main_as_Extra();
        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "transA") {
                transA = attr->i() > 0;
                continue;
            }
            if (key == "transB") {
                transB = attr->i() > 0;
                continue;
            }
            if (key == "alpha") {
                alpha = attr->f();
                continue;
            }
            if (key == "beta") {
                beta = attr->f();
                continue;
            }
        }
        auto X = inputs[0];
        auto Y = inputs[1];
        auto x_expr = X->expr().first;
        auto y_expr = Y->expr().first;
        auto Z = _MatMul(X, Y, transA, transB);
        if (x_expr->get() && y_expr->get() && x_expr->get()->type() == OpType_Int8ToFloat && y_expr->get()->type() == OpType_Int8ToFloat) {
            auto config = Global<modelConfig>::Get();
            if (helpers::IsConstant(y_expr)) {
                auto matmulOp = expr->get();
                auto weight = Y;
                auto input = X;
                auto weightInfo = weight->getInfo();
                auto transposeB = matmulOp->main_as_MatMul()->transposeB();
                auto transposeA = matmulOp->main_as_MatMul()->transposeA();
                auto needSqueezeB = false;
                auto needSqueezeA = false;
                bool inputShapeUnknow = false;
                if (input->getInfo() != nullptr) {
                    if (input->getInfo()->dim.size() <= 1) {
                        input = _Unsqueeze(input, {0});
                        needSqueezeA = true;
                    }
                } else {
                    inputShapeUnknow = true;
                }
                if (weightInfo->dim.size() == 1) {
                    weight = _Unsqueeze(weight, {1});
                    needSqueezeB = true;
                }
                if (!transposeB) {
                    weight = _Transpose(weight, {1, 0});
                }
                if (X->getInfo() && X->getInfo()->dim.size() <= 1) {
                    X = _Unsqueeze(X, {0});
                    needSqueezeA = true;
                }
                if (needSqueezeA && needSqueezeB) {
                    MNN_ERROR("Invalid MatMul for one-dimension A and B\n");
                    return nullptr;
                }
                auto format = MNN::MNN_DATA_FORMAT_NCHW;
                int oc = weight->getInfo()->dim[0];
                int ic = weight->getInfo()->dim[1];
                
                // quan parameters
                float inputScale  = X->expr().first->get()->main_as_QuantizedFloatParam()->tensorScale()->data()[0];
                float inputZero   = X->expr().first->get()->main_as_QuantizedFloatParam()->floatzeros() ->data()[0];
                auto  weightScale = Y->expr().first->get()->main_as_QuantizedFloatParam()->tensorScale()->data();
                auto  weightZero  = Y->expr().first->get()->main_as_QuantizedFloatParam()->floatzeros()->data();
                // conv op
                std::unique_ptr<Convolution2DT> conv(new MNN::Convolution2DT);
                conv->common.reset(new MNN::Convolution2DCommonT);
                conv->common->inputCount = ic;
                conv->common->outputCount = oc;
                // conv quant parameters
                conv->quanParameter.reset(new IDSTQuanT);
                conv->quanParameter->scaleIn = inputScale;
                conv->quanParameter->type = 4;
                conv->quanParameter->aMin = -128;
                conv->quanParameter->readType = oc;
                conv->quanParameter->quantScale = 1.f;
                conv->quanParameter->buffer.resize(Y->getInfo()->size);
                ::memcpy(conv->quanParameter->buffer.data(), weight->readMap<int8_t>(), Y->getInfo()->size);
                conv->quanParameter->alpha.resize(2 * oc);
                for (int i = 0; i < oc; ++i) {
                    conv->quanParameter->alpha[2 * i] = (-1 * weightZero[i] - 128.f) / weightScale[i]; // minval
                    conv->quanParameter->alpha[2 * i + 1] = weightScale[i];
                }
                // output expr
                auto outputExpr = expr->outputs().front().lock();
                auto outputScaleVar = outputExpr->inputs()[1];
                auto outputZero = _Const(0.f);
                if (outputExpr->inputs().size() > 2 && outputExpr->inputs()[2]->getInfo()) {
                    if (outputExpr->inputs()[2]->getInfo()->type.code == halide_type_int) {
                        outputZero = _Cast<float>(outputExpr->inputs()[2]);
                    } else {
                        outputZero = _Cast<float>(outputExpr->inputs()[2]) - _Const(128.f);
                    }
                }
                conv->quanParameter->scaleOut = outputScaleVar->readMap<float>()[0];
                conv->symmetricQuan.reset(new QuantizedFloatParamT);
                conv->symmetricQuan->nbits = 8;
                conv->symmetricQuan->clampMax = 127;
                conv->symmetricQuan->clampMin = -128;
                conv->symmetricQuan->zeroPoint = static_cast<int8_t>(inputZero);
                conv->symmetricQuan->outputZeroPoint = static_cast<int8_t>(outputZero->readMap<float>()[0]);
                conv->bias.resize(oc);
                if (inputs.size() > 2) {
                    memcpy(conv->bias.data(), inputs[2]->readMap<float>(), oc * sizeof(float));
                }
                
                std::unique_ptr<OpT> conv_op(new OpT);
                conv_op->type = OpType_Convolution;
                conv_op->main.type = OpParameter_Convolution2D;
                conv_op->main.value = conv.release();

                auto rank = _Rank(X);
                auto inputShape = _Shape(X, NCHW);
                auto inputL = _Unsqueeze(_Scalar<int>(ic), {0});
                inputL.fix(VARP::CONSTANT);
                auto outputH = _Unsqueeze(_Scalar<int>(oc), {0});
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
                    input = _ReshapeF(X, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, inputE, _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                    
                } else {
                    inputE = _Slice(inputShape, rankRemain, _Unsqueeze(inputELength, {0}));
                    input = _ReshapeF(X, _Concat({_Unsqueeze(_Scalar<int>(-1), {0}), inputL, _Unsqueeze(_Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0})}, 0), format);
                    
                }
                EXPRP dense_expr = Expr::create(conv_op.get(), {X}, 1);
                VARP output = Variable::create(dense_expr);
                output->setName(expr->outputName(0) + "__matmul_converted");
                output = _ConvertF(output, format);
                VARP reshapeVar = _ReshapeF(output, _Concat({inputRemain, inputE, outputH}, 0), format);
                if (needSqueezeA) {
                    reshapeVar = _Squeeze(reshapeVar, {0});
                }
                if (needSqueezeB) {
                    reshapeVar = _Squeeze(reshapeVar, {1});
                }
                reshapeVar->setName(expr->outputName(0));
                return reshapeVar->expr().first;
            }
            // input quant info
            auto y_int8 = y_expr->inputs().at(0);
            auto y_scale = y_expr->inputs().at(2);
            auto y_zero = y_expr->inputs().at(3);
            auto x_int8 = x_expr->inputs().at(0);
            auto x_scale = x_expr->inputs().at(2);
            auto x_zero = x_expr->inputs().at(3);
            // output quant info
            auto outputExpr = expr->outputs().front().lock();
            auto outputScaleVar = outputExpr->inputs()[1];
            auto outputZero = _Const(0.f);
            if (outputExpr->inputs().size() > 2 && outputExpr->inputs()[2]->getInfo()) {
                if (outputExpr->inputs()[2]->getInfo()->type.code == halide_type_int) {
                    outputZero = _Cast<float>(outputExpr->inputs()[2]);
                } else {
                    outputZero = _Cast<float>(outputExpr->inputs()[2]) - _Const(128.f);
                }
            }
            
            Z = _MatMul_Int8(X, y_int8, transA, transB, x_scale, x_zero, y_scale, y_zero, outputScaleVar, outputZero);
            if (inputs.size() > 2) {
                auto bias_expr = inputs[2]->expr().first;
                auto bias_int32 = bias_expr->inputs().at(1);
                Z = _MatMul_Int8(X, y_int8, transA, transB, x_scale, x_zero, y_scale, y_zero, outputScaleVar, outputZero, bias_int32);
            }
            Z->setName(expr->name());
            return Z->expr().first;
        }
        
        if (1.0f != alpha) {
            Z = Z * _Scalar<float>(alpha);
        }
        if (inputs.size() > 2) {
            auto B = inputs[2];
            if (1.0f != beta) {
                B = B * _Scalar<float>(beta);
            }
            Z = Z + B;
        }
        Z->setName(expr->name());

        return Z->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Gemm", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGemmTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
