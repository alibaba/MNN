//
//  Transformer.cpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Transformer.hpp"
#include "OpConverter.hpp"
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
                    va.fix(VARP::CONST);
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

std::shared_ptr<Express::Optimizer> Transformer::turnModelToInfer() {
    return nullptr;
}
} // namespace Train
} // namespace MNN
