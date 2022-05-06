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
    createUnit(varMap, inputIndexes, oplists, op, net->tensorName, invalidSet, extraInputIndexes);
}

void Program::createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists,
                    MNN::OpT* op, const std::vector<std::string>& tensorName, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes) {
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
                        createUnit(varMap, inputIndexes, oplists, oplists[j].get(), tensorName, invalidSet, extraInputIndexes);
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
        varMap[outputIndexes[j]] = newVar;
    }
}

VARPS Program::input(const std::unordered_map<std::string, VARP>& inputs, bool lazy) {
    VARPS inputUpdate;
    for (auto& it : mVars) {
        auto var = it.second;
        auto expr = var->expr().first;
        if (expr->get() != nullptr || expr->inputType() != VARP::INPUT) {
            continue;
        }
        if (inputs.count(var->name())) {
            VARP input = inputs.at(var->name());
            inputUpdate.emplace_back(var);
            if (lazy) {
                // only replace expr, not do getInfo, avoid unnecessary getInfo error
                // replace will override var(and expr)'s name, remain them so we can track input var
                // origin input var will be used when save program to net
                mOriginInputs.emplace_back(var, var->expr().first, var->expr().second);
                var->setExpr(input->expr().first, input->expr().second);
            } else {
                var->input(input);
            }
        }
    }
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
    return create(net->oplists, net->tensorName, net->outputName, supportExtra, saveAllVars);
}

std::shared_ptr<Program> Program::create(const MNN::SubGraphProtoT* subgraph, bool supportExtra, bool saveAllVars) {
    std::vector<std::string> outputName;
    for (auto idx : subgraph->outputs) {
        outputName.push_back(subgraph->tensors[idx]);
    }
    return create(subgraph->nodes, subgraph->tensors, outputName, supportExtra, saveAllVars);
}

std::shared_ptr<Program> Program::create(const std::vector<std::unique_ptr<OpT>>& oplists, const std::vector<std::string>& tensorName, const std::vector<std::string>& outputName, bool supportExtra, bool saveAllVars) {
    std::map<int, VARP> varMap;
    std::vector<int> inputIndexes;
    std::set<int> extraInputIndexes;
    for (int index = 0; index < oplists.size(); ++index) {
        std::set<OpT*> invalidSet;
        createUnit(varMap, inputIndexes, oplists, oplists[index].get(), tensorName, invalidSet, extraInputIndexes);
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
    if (saveAllVars) {
        program.mVars    = std::move(varMap);
    }
    for (auto output : outputs) {
        program.mOutputs.emplace_back(output.second);
    }
    return newProgram;
}
} // namespace Express
} // namespace MNN
