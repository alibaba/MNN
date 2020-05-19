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
using namespace MNN::Express;
namespace MNN {
namespace Train {

class TurnTrainable : public Express::Optimizer {
public:
    TurnTrainable(Transformer::TrainConfig config) {
        mConfig = std::move(config);
    }
    virtual Cost onMeasure(const std::vector<VARP>& outputs,
                           std::shared_ptr<Parameters> parameters = nullptr) override {
        return Cost();
    }
    virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> p) override {
        auto exprs = Variable::getExecuteOrder(outputs);
        {
            // Turn convolution be trainable convolution
            for (auto expr : exprs) {
                auto newExpr = OpConverter::convert(expr);
                if (newExpr.get() != expr.get()) {
                    Expr::replace(expr, newExpr);
                }
            }
        }
        exprs                = Variable::getExecuteOrder(outputs);
        auto& variableLimits = mConfig.variableLimits;
        // Collect Const Variable and turn to Trainable
        for (auto v : exprs) {
            if (v->get() == nullptr && VARP::INPUT != v->inputType()) {
                auto name = v->name();
                auto info = v->outputInfo(0);
                if (halide_type_float != info->type.code) {
                    continue;
                }
                bool match = variableLimits.empty();
                for (auto limit : variableLimits) {
                    if (name.find(limit) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
                auto va = Variable::create(v, 0);
                if (match) {
                    MNN_PRINT("Add Variable: %s\n", name.c_str());
                    va.fix(VARP::TRAINABLE);
                } else {
                    va.fix(VARP::CONSTANT);
                }
            }
        }
        return true;
    }

private:
    Transformer::TrainConfig mConfig;
};

std::shared_ptr<Express::Optimizer> Transformer::turnModelToTrainable(TrainConfig config) {
    std::shared_ptr<Express::Optimizer> res;
    res.reset(new TurnTrainable(std::move(config)));
    return res;
}

class InferOptimizer : public Express::Optimizer {
public:
    InferOptimizer(){}
    virtual Cost onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) override {
        Cost c;
        return c;
    };

    virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) override {
        auto exprs = Variable::getExecuteOrder(outputs);
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
};

std::shared_ptr<Express::Optimizer> Transformer::turnModelToInfer() {
    return std::shared_ptr<Optimizer>(new InferOptimizer);
}
} // namespace Train
} // namespace MNN
