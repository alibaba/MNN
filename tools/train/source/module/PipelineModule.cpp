//
//  PipelineModule.cpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PipelineModule.hpp"
using namespace MNN::Express;
namespace MNN {
namespace Train {
class ExprModule : public Module {
public:
    ExprModule(EXPRP expr) {
        MNN_ASSERT(expr->get() != nullptr);
        mExpr   = expr;
        mInputs = expr->inputs();
        for (int i = 0; i < mInputs.size(); ++i) {
            auto inputExpr = mInputs[i]->expr().first;
            if (inputExpr->get() != nullptr) {
                mInputs[i] = nullptr;
                mInputIndexes.emplace_back(i);
                continue;
            }
            switch (inputExpr->inputType()) {
                case VARP::INPUT:
                    mInputs[i] = nullptr;
                    mInputIndexes.emplace_back(i);
                    break;
                case VARP::CONST:
                    break;
                case VARP::TRAINABLE:
                    addParameter(mInputs[i]);
                    break;
                default:
                    break;
            }
        }
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        MNN_ASSERT(mInputIndexes.size() == inputs.size());
        std::vector<VARP> tempInputs = mInputs;
        for (int i = 0; i < inputs.size(); ++i) {
            tempInputs[mInputIndexes[i]] = inputs[i];
        }
        std::vector<VARP> outputVars;
        auto newExpr = Expr::create(mExpr->extra(), std::move(tempInputs), mExpr->outputSize());
        for (int i = 0; i < mExpr->outputSize(); ++i) {
            outputVars.emplace_back(Variable::create(newExpr, i));
        }
        return outputVars;
    }
    const std::vector<int>& inputIndexes() const {
        return mInputIndexes;
    }

private:
    EXPRP mExpr;
    std::vector<VARP> mInputs;
    std::vector<int> mInputIndexes;
};
PipelineModule::PipelineModule(std::vector<VARP> inputs, std::vector<VARP> outputs, Transformer& transformFunction) {
    auto executeOrder = Variable::getExecuteOrder(outputs);
    // Set Indexes
    std::map<EXPRP, int> indexes;
    int currentIndexes = 0;
    for (auto expr : executeOrder) {
        indexes[expr] = currentIndexes;
        currentIndexes += expr->outputSize();
    }
    mInputIndexes.clear();
    mStack.resize(currentIndexes);
    for (auto v : inputs) {
        auto inputExpr = v->expr();
        mInputIndexes.emplace_back(indexes[inputExpr.first] + inputExpr.second);
    }

    // Create All SubModule
    for (auto expr : executeOrder) {
        if (expr->get() == nullptr) {
            continue;
        }
        auto moduleResult = transformFunction(expr);
        if (moduleResult.second == nullptr) {
            std::shared_ptr<Module> module(new ExprModule(expr));
            moduleResult.first  = ((ExprModule*)module.get())->inputIndexes();
            moduleResult.second = module;
        }
        auto subInputs        = expr->inputs();
        auto exprInputIndexes = moduleResult.first;
        std::vector<int> inputIndexes(exprInputIndexes.size());
        for (int i = 0; i < exprInputIndexes.size(); ++i) {
            auto inputExpr  = subInputs[exprInputIndexes[i]]->expr();
            inputIndexes[i] = indexes[inputExpr.first] + inputExpr.second;
        }

        std::vector<int> outputIndexes(expr->outputSize());
        for (int i = 0; i < outputIndexes.size(); ++i) {
            outputIndexes[i] = indexes[expr] + i;
        }
        mSubModules.emplace_back(std::make_tuple(moduleResult.second, inputIndexes, outputIndexes));
        registerModel({moduleResult.second});
    }
    mOutputIndexes.clear();
    for (auto output : outputs) {
        auto outputExpr = output->expr();
        mOutputIndexes.emplace_back(indexes[outputExpr.first] + outputExpr.second);
    }
}
std::vector<VARP> PipelineModule::onForward(const std::vector<VARP>& inputs) {
    for (int i = 0; i < mInputIndexes.size(); ++i) {
        mStack[mInputIndexes[i]] = inputs[i];
    }
    for (auto& m : mSubModules) {
        std::vector<VARP> tempInputs(std::get<1>(m).size());
        for (int i = 0; i < tempInputs.size(); ++i) {
            tempInputs[i] = mStack[std::get<1>(m)[i]];
        }
        std::vector<VARP> tempOutputs = std::get<0>(m)->onForward(tempInputs);
        MNN_ASSERT(tempOutputs.size() == std::get<2>(m).size());
        for (int i = 0; i < tempOutputs.size(); ++i) {
            mStack[std::get<2>(m)[i]] = tempOutputs[i];
        }
    }
    std::vector<VARP> outputs(mOutputIndexes.size());
    for (int i = 0; i < mOutputIndexes.size(); ++i) {
        outputs[i] = mStack[mOutputIndexes[i]];
    }
    return outputs;
}
void PipelineModule::onClearCache() {
    for (auto& v : mStack) {
        v = nullptr;
    }
}
} // namespace Train
} // namespace MNN
