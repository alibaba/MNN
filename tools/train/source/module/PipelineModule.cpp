//
//  PipelineModule.cpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PipelineModule.hpp"
#include "NN.hpp"
#include "MNN_generated.h"
#include <set>
#include <vector>
using namespace MNN::Express;
namespace MNN {
namespace Train {
#define PIPELINE_MODULE "_pipeline_module__"
class ExprModule : public Module {
public:
    ExprModule(EXPRP expr) {
        mExpr   = expr;
        mInputs = expr->inputs();
        auto op = mExpr->get();
        if (op) {
            auto typeName = EnumNameOpType(op->type());
            setType(typeName);
        }
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
                case VARP::CONSTANT:
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
        if (nullptr == mExpr->get()) {
            return {Variable::create(mExpr)};
        }
        std::vector<VARP> tempInputs = mInputs;
        for (int i = 0; i < inputs.size(); ++i) {
            tempInputs[mInputIndexes[i]] = inputs[i];
        }
        std::vector<VARP> outputVars;
        auto newExpr = Expr::create(mExpr->extra(), std::move(tempInputs), mExpr->outputSize());
        newExpr->setName(mExpr->name());
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

Module* PipelineModule::extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain) {
    std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction;
    if (fortrain) {
        transformFunction =
        [](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source));
            if (nullptr != m) {
                return std::make_pair(std::vector<int>{0}, m);
            }
            auto convExtracted = NN::Utils::ExtractConvolution(source);
            if (convExtracted.weight == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> module(NN::Conv(convExtracted));
            module->setName(source->name());
            return std::make_pair(std::vector<int>{0}, module);
        };
    } else {
        transformFunction = [](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source));
            if (nullptr != m) {
                return std::make_pair(std::vector<int>{0}, m);
            }
            return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        };
    }
    return new PipelineModule(inputs, outputs, transformFunction);
}

PipelineModule::PipelineModule(std::vector<VARP> inputs, std::vector<VARP> outputs, const Transformer& transformFunction) {
    setType(PIPELINE_MODULE);
    std::vector<EXPRP> executeOrder;
    std::set<EXPRP> inputExpr;
    for (auto v : inputs) {
        inputExpr.insert(v->expr().first);
    }
    for (auto output : outputs) {
        Expr::visit(output->expr().first,
        [&executeOrder, &inputExpr](EXPRP expr) {
            if (expr->visited()) {
                return false;
            }
            if (inputExpr.find(expr)!= inputExpr.end()) {
                expr->setVisited(true);
                executeOrder.emplace_back(expr);
                return false;
            }
            return true;
        },
        [&executeOrder](EXPRP expr) {
            //FUNC_PRINT_ALL(var->name().c_str(), s);
            if (!expr->visited()) {
                executeOrder.emplace_back(expr);
                expr->setVisited(true);
            }
            return true;
        });
    }
    for (auto expr : executeOrder) {
        expr->setVisited(false);
    }
    // Set Indexes
    std::map<EXPRP, int> indexes;
    int currentIndexes = 0;
    for (auto expr : executeOrder) {
        indexes[expr] = currentIndexes;
        currentIndexes += expr->outputSize();
    }
    std::set<EXPRP> inputSets;
    mInputIndexes.clear();
    mStack.resize(currentIndexes);
    for (auto v : inputs) {
        auto inputExpr = v->expr();
        mInputIndexes.emplace_back(indexes[inputExpr.first] + inputExpr.second);
        inputSets.insert(inputExpr.first);
    }

    // Create All SubModule
    for (auto expr : executeOrder) {
        if (inputSets.find(expr) != inputSets.end()) {
            continue;
        }
        std::pair<std::vector<int>, std::shared_ptr<Module> > moduleResult;
        if (!transformFunction) {
            moduleResult = std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        } else {
            moduleResult = transformFunction(expr);
        }
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
bool PipelineModule::turnQuantize(Module* module, const int bit, NN::FeatureScaleStatMethod featureScaleStatMethod, NN::ScaleUpdateMethod scaleUpdateMethod) {
    if (nullptr == module || module->type() != PIPELINE_MODULE) {
        MNN_ERROR("Invalide module for quantized\n");
        return false;
    }
    ((PipelineModule*)module)->toTrainQuant(bit, featureScaleStatMethod, scaleUpdateMethod);
    return true;
}

void PipelineModule::toTrainQuant(const int bits, NN::FeatureScaleStatMethod featureScaleStatMethod,
                                        NN::ScaleUpdateMethod scaleUpdateMethod) {
    std::vector<int> needEraseIndices;

    for (int i = 0; i < mSubModules.size(); i++) {
        auto& m = mSubModules[i];
        auto& theModule = std::get<0>(m);
        auto moduleType = theModule->type();
        auto& inputIndices = std::get<1>(m);
        auto& outputIndices = std::get<2>(m);

        if (moduleType == "Conv" && i < mSubModules.size() - 1) {
            auto& p1 = mSubModules[i+1];
            auto p1Module = std::get<0>(p1);
            auto& p1ModuleType = p1Module->type();
            auto& p1InputIndices = std::get<1>(p1);
            auto& p1OutputIndices = std::get<2>(p1);

            // only conv
            if ((p1ModuleType == "Conv") ||
                    (p1ModuleType != "BatchNorm" && p1ModuleType != "ReLU" && p1ModuleType != "ReLU6")) {
                theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
                registerModel({theModule});
                continue;
            }
            // conv + bn + ?
            if (p1ModuleType == "BatchNorm") {
                // make sure that they are connected
                MNN_ASSERT(outputIndices.size() == 1 && p1InputIndices.size() == 1);
                MNN_ASSERT(outputIndices[0] = p1InputIndices[0]);

                // last conv + bn
                if (i == mSubModules.size() - 2) {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                }
                // maybe there is a relu or relu6 after conv + bn
                auto& p2 = mSubModules[i+2];
                auto& p2Module = std::get<0>(p2);
                auto p2ModuleType = p2Module->type();
                auto& p2InputIndices = std::get<1>(p2);
                auto& p2OutputIndices = std::get<2>(p2);
                // only conv + bn
                if (p2ModuleType != "ReLU" && p2ModuleType != "ReLU6") {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                } else { // conv + bn + relu or conv + bn + relu6
                    // make sure that they are connected
                    MNN_ASSERT(p1OutputIndices.size() == 1 && p2InputIndices.size() == 1);
                    MNN_ASSERT(p1OutputIndices[0] = p2InputIndices[0]);

                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module, p2Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    registerModel({theModule});
                    outputIndices = p2OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    needEraseIndices.emplace_back(i + 2);
                    continue;
                }
            }
            // conv + relu or conv + relu6
            if (p1ModuleType == "ReLU" || p1ModuleType == "ReLU6") {
                // make sure that they are connected
                MNN_ASSERT(outputIndices.size() == 1 && p1InputIndices.size() == 1);
                MNN_ASSERT(outputIndices[0] = p1InputIndices[0]);

                theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                registerModel({theModule});
                outputIndices = p1OutputIndices;
                needEraseIndices.emplace_back(i + 1);
                continue;
            }
        }

        if (i == mSubModules.size() - 1 && moduleType == "Conv") {
            theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
            registerModel({theModule});
        }
    }

    // erase useless submodules
    const int eraseSize = needEraseIndices.size();
    int alreadyErasedCount = 0;
    for (int i = 0; i < eraseSize; i++) {
        auto position = needEraseIndices[i] - alreadyErasedCount;
        auto type = std::get<0>(mSubModules[position])->type();
        MNN_ASSERT(type == "BatchNorm" || type == "ReLU" || type == "ReLU6");
        mSubModules.erase(mSubModules.begin() + position);
        alreadyErasedCount++;
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
