//
//  Transformer.cpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Transformer.hpp"
#include "OpConverter.hpp"
#include "MNN_generated.h"
#include <MNN/expr/ExprCreator.hpp>
using namespace MNN::Express;
namespace MNN {
namespace Train {

bool TurnTrainable::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> p) {
    auto& trainInfo = mTrainInfo.bnVariables;
    auto exprs = Variable::getExecuteOrder(outputs);
    {
        auto isTraining = _Input({}, NCHW, halide_type_of<int>());
        isTraining->setName("is_training");
        trainInfo["is_training"] = isTraining;
        isTraining = _Cast<float>(isTraining);
        isTraining->setName("is_training_float");
        trainInfo["is_training_float"] = isTraining;
        trainInfo["one_float"] = _Scalar<float>(1.0f);
        trainInfo["bn_momentum"] = _Scalar<float>(mConfig.extraParams["BatchNorm"]["momentum"]->f);
        // Turn convolution be trainable convolution
        for (auto expr : exprs) {
            auto newExpr = OpConverter::convert(expr, mTrainInfo);
            if (newExpr.get() != expr.get()) {
                Expr::replace(expr, newExpr);
            }
        }
    }
    exprs                = Variable::getExecuteOrder(outputs);
    auto& noUpdateOps = mConfig.noUpdateOps;
    auto& onlyUpdateOps = mConfig.onlyUpdateOps;
    // Collect Const Variable and turn to Trainable
    for (auto v : exprs) {
        if (v->get() == nullptr && VARP::INPUT != v->inputType()) {
            auto name = v->name();
            auto info = v->outputInfo(0);
            if (halide_type_float != info->type.code) {
                continue;
            }

            bool update;
            if (!onlyUpdateOps.empty()) {
                update = false;
                for (auto limit : onlyUpdateOps) {
                    if (name.find(limit) != std::string::npos) {
                        update = true;
                        break;
                    }
                }
            } else {
                update = true;
                for (auto limit : noUpdateOps) {
                    if (name.find(limit) != std::string::npos) {
                        update = false;
                        break;
                    }
                }
            }
            
            auto va = Variable::create(v, 0);
            if (update && name != "") {
                va.fix(VARP::TRAINABLE);
                if (name.find("Weight") == std::string::npos && name.find("Bias") == std::string::npos) {
                    MNN_PRINT(">>>\ncheck mnn model if const '%s' is a learnable parameter in your original training model, ", name.c_str());
                    MNN_PRINT("if not, add it to transformConfig.json NoUpdateOps\n<<<\n");
                    va->setName(name + "_Weight");
                    va->expr().first->setName(va->name());
                }
                mTrainInfo.trainables.insert(std::make_pair(name, va->name()));
                MNN_PRINT("Add Variable: %s\n", va->name().c_str());
            } else {
                va.fix(VARP::CONSTANT);
            }
        }
    }
    return true;
}

std::shared_ptr<Express::Optimizer> Transformer::turnModelToTrainable(TrainConfig config) {
    std::shared_ptr<Express::Optimizer> res;
    res.reset(new TurnTrainable(std::move(config)));
    return res;
}

bool InferOptimizer::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    auto exprs = Variable::getExecuteOrder(outputs);

    // convert trainable to const
    for (auto& expr : exprs) {
        if (expr->inputs().size() == 0 && expr->inputType() == VARP::InputType::TRAINABLE) {
            auto newConst = Variable::create(expr);
            newConst.fix(VARP::InputType::CONSTANT);
            newConst->setName(expr->name());
            auto newExpr = newConst->expr().first;
            newExpr->setName(expr->name());
            Expr::replace(expr, newExpr);
        }
    }

    // merge bn after conv into conv
    // convert single bn to scale
    std::set<std::string> bnNames;
    std::string pattern1 = "_MNN_BN_after_conv_first_op";
    std::string pattern2 = "_MNN_single_BN_first_op";
    for (auto& expr : exprs) {
        if (expr->name().find(pattern1) != std::string::npos) {
            std::string bnName = expr->name();
            for (int i = 0; i < pattern1.size(); i++) {
                bnName.pop_back();
            }
            bnNames.insert(bnName);
        }
        if (expr->name().find(pattern2) != std::string::npos) {
            std::string bnName = expr->name();
            for (int i = 0; i < pattern2.size(); i++) {
                bnName.pop_back();
            }
            bnNames.insert(bnName);
        }
    }

    std::map<std::string, std::map<std::string, EXPRP>> bnInfo;
    for (auto& name : bnNames) {
        for (auto& expr : exprs) {
            auto inputs = expr->inputs();
            if (expr->name() == name) {
                bnInfo[name]["Self"] = expr;
            }
            if (inputs.size() == 0 && expr->name() == name + "_BN_RunningMean_Weight") {
                bnInfo[name]["RunningMean"] = expr;
            }
            if (inputs.size() == 0 && expr->name() == name + "_BN_RunningVariance_Weight") {
                bnInfo[name]["RunningVariance"] = expr;
            }
            if (inputs.size() == 0 && expr->name() == name + "_BN_Gamma_Weight") {
                bnInfo[name]["Gamma"] = expr;
            }
            if (inputs.size() == 0 && expr->name() == name + "_BN_Beta_Bias") {
                bnInfo[name]["Bias"] = expr;
            }
            if (inputs.size() == 0 && expr->name() == name + "_BN_Eps_Weight") {
                bnInfo[name]["Eps"] = expr;
            }
            if (expr->name() == name + pattern1) {
                bnInfo[name]["FirstOpAfterConv"] = expr;
            }
            if (expr->name() == name + pattern2) {
                bnInfo[name]["FirstOpSingleBN"] = expr;
            }
        }
    }
    for (auto& bn : bnInfo) {
        auto bnName = bn.first;
        auto info = bn.second;

        bool bnAfterConv = false;
        if (info.find("FirstOpAfterConv") != info.end()) {
            bnAfterConv = true;
        }

        auto rm = _Convert(Variable::create(info["RunningMean"]), NCHW);
        auto rv = _Convert(Variable::create(info["RunningVariance"]), NCHW);
        auto gamma = _Convert(Variable::create(info["Gamma"]), NCHW);
        auto bias = _Convert(Variable::create(info["Bias"]), NCHW);
        auto eps = Variable::create(info["Eps"]);

        auto s = _Sqrt(rv + eps);
        auto alpha = gamma / s;
        auto beta = bias - rm / s * gamma;

        if (bnAfterConv) {
            auto firstOp = info["FirstOpAfterConv"];
            auto convExpr = firstOp->inputs()[0]->expr().first;
            if (convExpr->get() == nullptr || convExpr->get()->type() != OpType_Convolution) {
                continue;
            }
            auto convInput = convExpr->inputs()[0];
            auto w = convExpr->inputs()[1];
            auto b = convExpr->inputs()[2];
            
            auto nw = w * _Reshape(alpha, {b->getInfo()->dim[0], 1, 1, 1});
            nw.fix(w->expr().first->inputType());
            nw->setName(w->name());
            auto nb = _Reshape(alpha, {b->getInfo()->dim}) * b + _Reshape(beta, b->getInfo()->dim);
            nb.fix(b->expr().first->inputType());
            nb->setName(b->name());

            std::vector<VARP> newInputs = {convInput, nw, nb};
            auto newConv = Expr::create(convExpr->extra(), std::move(newInputs));
            Expr::replace(info["Self"], newConv);
        } else {
            auto firstOp = info["FirstOpSingleBN"];
            auto inputs = firstOp->inputs();
            std::vector<float> scales, p;
            for (int i = 0; i < beta->getInfo()->size; i++) {
                scales.push_back(alpha->readMap<float>()[i]);
                p.push_back(beta->readMap<float>()[i]);
            }
            auto res = _Scale(inputs[0], beta->getInfo()->size, std::move(scales), std::move(p));
            res->setName(info["Self"]->name());
            Expr::replace(info["Self"], res->expr().first);
        }
    }

    exprs = Variable::getExecuteOrder(outputs);
    for (auto& iter : exprs) {
        auto op = iter->get();
        if (nullptr == op) {
            continue;
        }
        if (op->type() != OpType_ConvInt8 && op->type() != OpType_DepthwiseConvInt8) {
            continue;
        }
        auto inputExpr = iter->inputs()[0]->expr().first;
        if (inputExpr->get() == nullptr) {
            continue;
        }
        if (inputExpr->get()->type() != OpType_FloatToInt8) {
            continue;
        }
        auto subInputExpr = inputExpr->inputs()[0]->expr().first;
        if (subInputExpr->get() == nullptr) {
            continue;
        }
        if (subInputExpr->get()->type() != OpType_Int8ToFloat) {
            continue;
        }
        //MNN_PRINT("Find direct\n");
        std::vector<VARP> newInputs = subInputExpr->inputs();
        auto newExpr = Expr::create(iter->extra(), std::move(newInputs));
        Expr::replace(iter, newExpr);
    }
    return true;
}

std::shared_ptr<Express::Optimizer> Transformer::turnModelToInfer() {
    return std::shared_ptr<Optimizer>(new InferOptimizer);
}
} // namespace Train
} // namespace MNN
