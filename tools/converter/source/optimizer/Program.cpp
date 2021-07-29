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

void Program::createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists,
                    MNN::OpT* op, const MNN::NetT* net, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes) {
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
        if (varMap.find(input) == varMap.end()) {
            for (int j = 0; j < oplists.size(); ++j) {
                for (auto outputIndex : oplists[j]->outputIndexes) {
                    if (outputIndex == input) {
                        createUnit(varMap, inputIndexes, oplists, oplists[j].get(), net, invalidSet, extraInputIndexes);
                    }
                }
            }
            if (varMap.find(input) == varMap.end()) {
                extraInputIndexes.insert(input);
//                MNN_PRINT("Don't find input %d - %s for %s, turn to input\n", input, net->tensorName[input].c_str(),
//                          op->name.c_str());
                auto newInput = _Input({-1});
                newInput->setName(net->tensorName[input]);
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
        newVar->setName(net->tensorName[outputIndexes[j]]);
        varMap[outputIndexes[j]] = newVar;
    }
}

void Program::input(const std::unordered_map<std::string, VARP>& inputs) {
    for (auto& it : mVars) {
        VARP var = it.second;
        if (var->expr().first->inputType() != VARP::INPUT) {
            continue;
        }
        if (inputs.count(var->name())) {
            VARP input = inputs.at(var->name());
            var->input(input);
        }
    }
}

std::shared_ptr<Program> Program::create(const MNN::NetT* net, bool supportExtra, bool saveAllVars) {
    std::map<int, VARP> varMap;
    std::vector<int> inputIndexes;
    std::set<int> extraInputIndexes;
    for (int index = 0; index < net->oplists.size(); ++index) {
        std::set<OpT*> invalidSet;
        createUnit(varMap, inputIndexes, net->oplists, net->oplists[index].get(), net, invalidSet, extraInputIndexes);
    }
    std::set<VARP> outputs;
    for (auto& iter : varMap) {
        if (iter.second->linkNumber() == 0) {
            outputs.insert(iter.second);
        }
    }
    for (auto& o : net->outputName) {
        int index = -1;
        for (int i=0; i<net->tensorName.size(); ++i) {
            if (net->tensorName[i] == o) {
                index = i;
                break;
            }
        }
        if (varMap.find(index) != varMap.end()) {
            outputs.insert(varMap[index]);
        }
    }
    std::shared_ptr<Program> newProgram(new Program);
    Program& program = *newProgram;
    if (saveAllVars) {
        program.mVars    = std::move(varMap);
    }
    for (auto output : outputs) {
        program.mOutputs.emplace_back(output);
    }
    return newProgram;
}
} // namespace Express
} // namespace MNN
