//
//  Program.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Program.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <unordered_map>
#include <unordered_set>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
using namespace MNN;
#define UP_DIV(x) (((x) + 3) / 4)
#include "MNN_generated.h"
namespace MNN {
namespace Express {

void Program::createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists, MNN::OpT* op, const MNN::NetT* net, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes) {
    createUnit(varMap, inputIndexes, oplists, op, net->tensorName, invalidSet, extraInputIndexes, net);
}

void Program::createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists,
                    MNN::OpT* op, const std::vector<std::string>& tensorName, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes, const MNN::NetT* net, std::map<std::string, int> TensorDescribeName) {
    if (invalidSet.find(op) != invalidSet.end()) {
        return;
    }
    std::vector<VARP> inputVars;
    auto outputIndexes = op->outputIndexes;
    for (int j = 0; j < outputIndexes.size(); ++j) {
        if (varMap.find(outputIndexes[j]) != varMap.end()) {
            // Don't support multi op output to one index
            return;
        }
    }
    invalidSet.insert(op);
    for (auto input : op->inputIndexes) {
        if (input < 0) { // optional input
            inputVars.emplace_back(nullptr);
            continue;
        }
        if (varMap.find(input) == varMap.end()) {
            for (int j = 0; j < oplists.size(); ++j) {
                for (auto outputIndex : oplists[j]->outputIndexes) {
                    if (outputIndex == input) {
                        createUnit(varMap, inputIndexes, oplists, oplists[j].get(), tensorName, invalidSet, extraInputIndexes, net, TensorDescribeName);
                    }
                }
            }
            if (varMap.find(input) == varMap.end()) {
                extraInputIndexes.insert(input);
//                MNN_PRINT("Don't find input %d - %s for %s, turn to input\n", input, net->tensorName[input].c_str(),
//                          op->name.c_str());
                auto newInput = _Input({-1});
                newInput->setName(tensorName[input]);
                varMap[input] = newInput;
            }
        }
        inputVars.emplace_back(varMap[input]);
    }
    auto expr = Expr::create(op, inputVars, outputIndexes.size());
    expr->setName(op->name);
    for (int j = 0; j < outputIndexes.size(); ++j) {
        if (op->type == OpType_Input) {
            inputIndexes.emplace_back(outputIndexes[j]);
        }
        auto newVar = Variable::create(expr, j);
        newVar->setName(tensorName[outputIndexes[j]]);
        if (nullptr != net && !net->extraTensorDescribe.empty()) {
            auto& extraDescribes = net->extraTensorDescribe;
//            int idx = outputIndexes[j];
            if (TensorDescribeName.find(op->name) != TensorDescribeName.end()) {
                int idx = TensorDescribeName[op->name];
                float scale = extraDescribes[idx]->quantInfo->scale;
                float zero = extraDescribes[idx]->quantInfo->zero;
                newVar->writeScaleMap(scale, zero);
            }
        }
        varMap[outputIndexes[j]] = newVar;
    }
}

VARPS Program::input(const std::unordered_map<std::string, VARP>& inputs, bool lazy) {
    VARPS inputUpdate;
    return inputUpdate;
}

void Program::save(MNN::NetT* net) {
    // use origin input var to save into net
    for (auto& it : mOriginInputs) {
        auto& var = std::get<0>(it);
        var->setExpr(std::get<1>(it), std::get<2>(it));
    }
    Variable::save(mOutputs, net);
}

std::shared_ptr<Program> Program::create(const MNN::NetT* net, bool supportExtra, bool saveAllVars) {
    return create(net->oplists, net->tensorName, net->outputName, supportExtra, saveAllVars, net);
}

std::shared_ptr<Program> Program::create(const MNN::SubGraphProtoT* subgraph, bool supportExtra, bool saveAllVars) {
    std::vector<std::string> outputName;
    for (auto idx : subgraph->outputs) {
        outputName.push_back(subgraph->tensors[idx]);
    }
    return create(subgraph->nodes, subgraph->tensors, outputName, supportExtra, saveAllVars);
}

std::shared_ptr<Program> Program::create(const std::vector<std::unique_ptr<OpT>>& oplists, const std::vector<std::string>& tensorName, const std::vector<std::string>& outputName, bool supportExtra, bool saveAllVars, const MNN::NetT* net) {
    std::map<int, VARP> varMap;
    std::vector<int> inputIndexes;
    std::set<int> extraInputIndexes;
    std::map<std::string, int> TensorDescribeName;
    if (net && net->extraTensorDescribe.size() > 0) {
        for (int i = 0; i < net->extraTensorDescribe.size(); ++i) {
            TensorDescribeName.insert(std::make_pair(net->extraTensorDescribe[i]->name, i));
        }
    }
    for (int index = 0; index < oplists.size(); ++index) {
        std::set<OpT*> invalidSet;
        createUnit(varMap, inputIndexes, oplists, oplists[index].get(), tensorName, invalidSet, extraInputIndexes, net, TensorDescribeName);
    }
    std::map<std::string, VARP> outputs;
    for (auto& iter : varMap) {
        if (iter.second->linkNumber() == 0) {
            outputs.insert(std::make_pair(iter.second->name(), iter.second));
        }
    }
    for (auto& o : outputName) {
        int index = -1;
        for (int i=0; i<tensorName.size(); ++i) {
            if (tensorName[i] == o) {
                index = i;
                break;
            }
        }
        if (varMap.find(index) != varMap.end()) {
            auto var = varMap[index];
            outputs.insert(std::make_pair(var->name(), var));
        }
    }
    std::shared_ptr<Program> newProgram(new Program);
    Program& program = *newProgram;
    for (auto output : outputs) {
        program.mOutputs.emplace_back(output.second);
    }
    return newProgram;
}

void Program::updateVars(std::map<std::string, VARP> map, std::vector<std::string> tensorName) {
    for (auto& iter: mVars) {
        if (iter.first < tensorName.size() && iter.first >= 0) {
            auto name = tensorName[iter.first];
            if (map.find(name) != map.end()) {
                auto var_ = map[name];
                mVars[iter.first] = var_;
            }
        }
    }
}
} // namespace Express
} // namespace MNN
