//
//  transformerExecution.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <MNN/expr/Module.hpp>
#include "OpGrad.hpp"
#include "Transformer.hpp"
#include "core/Macro.h"
#define USE_ELU
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace std;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./transformer.out temp.bin dst.bin config.json\n");
        return 0;
    }
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        FUNC_PRINT(document.HasParseError());
        FUNC_PRINT(document.IsArray());
        FUNC_PRINT(document.IsObject());
    }
    auto configObject = document.GetObject();
    std::vector<std::string> variableLimits;
    if (configObject.HasMember("Optimizor")) {
        auto optimizor = configObject["Optimizor"].GetObject();
        if (optimizor.HasMember("Variables")) {
            auto limitArray = optimizor["Variables"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                variableLimits.emplace_back(vIter->GetString());
                MNN_PRINT("Variabale contain : %s \n", vIter->GetString());
            }
        }
    }
    const char* inputModeFileName = argv[1];
    FUNC_PRINT_ALL(inputModeFileName, s);
    std::map<std::string, VARP> inputVars;
    std::map<std::string, VARP> outputVars;
    {
        auto inputsOutputs = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
        inputVars = inputsOutputs.first;
        outputVars = inputsOutputs.second;
    }
    Transformer::TrainConfig trainConfig;
    trainConfig.variableLimits = std::move(variableLimits);
    Transformer::turnModelToTrainable(trainConfig)->onExecute(Variable::mapToSequence(outputVars));
    if (configObject.HasMember("Shape")) {
        auto shapeArray = configObject["Shape"].GetObject();
        for (auto shapeIter = shapeArray.begin(); shapeIter != shapeArray.end(); shapeIter++) {
            auto dimArray = shapeIter->value.GetArray();
            std::vector<int> dims;
            for (auto dimIter = dimArray.begin(); dimIter != dimArray.end(); dimIter++) {
                dims.emplace_back(dimIter->GetInt());
            }
            FUNC_PRINT_ALL(shapeIter->name.GetString(), s);
            std::string key = shapeIter->name.GetString();
            for (auto& varIter : inputVars) {
                auto var = varIter.second;
                if (var->name() == key) {
                    var->resize(dims);
                    break;
                }
            }
        }
    }
    auto exprs = Variable::getExecuteOrder(Variable::mapToSequence(outputVars));

    // Collect Const Variable
    std::set<VARP> parameters;
    for (auto v : exprs) {
        if (v->get() == nullptr && VARP::TRAINABLE == v->inputType()) {
            auto va = Variable::create(v, 0);
            parameters.insert(va);
        }
    }
    for (auto p : parameters) {
        p.fix(VARP::CONSTANT);
    }

    VARP loss;
    bool hasLoss = configObject.HasMember("Loss");
    if (!hasLoss) {
        auto output      = outputVars.begin()->second;
        auto outputShape = output->getInfo();
        if (outputShape->order == NC4HW4) {
            auto outputName = output->name();
            output->setName(outputName + "Origin");
            output      = _Convert(output, NHWC);
            outputShape = output->getInfo();
            output->setName(outputName);
        }
        auto outputReal = _Input(outputShape->dim, outputShape->order);
        outputReal->setName(output->name() + "_Compare");
#ifdef USE_ELU
        auto sub = _Subtract(output, outputReal);
        sub->setName(output->name() + "_Sub");
        loss = (_ReduceSum(_Multiply(sub, sub), {}));
#else
        auto mul = _Multiply(_Log(output), outputReal);
        mul->setName(output->name() + "_Mul");
        loss = _Negative(_ReduceSum(mul, {}));
#endif
        auto l2 = _Const(0.0f);
        for (auto var : parameters) {
            l2 = l2 + (var * var).sum({});
        }
        loss = loss + _Multiply(l2, _Const(0.0005f));
        loss->setName("Loss");
        exprs = Variable::getExecuteOrder({loss});
    } else {
        std::string lossName = configObject["Loss"].GetObject()["op"].GetString();
        for (auto expr : exprs) {
            if (expr->name() == lossName) {
                loss = Variable::create(expr);
                break;
            }
        }
        for (auto iter : outputVars) {
            if (iter.first == lossName) {
                outputVars.erase(iter.first);
                break;
            }
        }
    }
    MNN_ASSERT(nullptr != loss);
    auto gradMap = OpGrad::grad(loss, parameters);
    // Make Update
    std::map<VARP, VARP> varUpdateMap;
    auto learningRate = _Input();
    learningRate->setName("LearningRate");
    for (auto iter : gradMap) {
        auto p = iter.first;
        auto q = iter.second;
        q      = _Subtract(p, _Multiply(q, learningRate));
        q->setName("update_" + p->name());
        varUpdateMap[p] = q;
    }
    std::unique_ptr<MNN::NetT> netStruct(new MNN::NetT);
    std::vector<VARP> resultOutputs;
    for (auto output : outputVars) {
        resultOutputs.emplace_back(output.second);
    }
    resultOutputs.emplace_back(loss);
    for (auto iter : varUpdateMap) {
        resultOutputs.emplace_back(iter.second);
    }
    Variable::save(resultOutputs, ".grad.mnn");
    Variable::save(resultOutputs, netStruct.get());
    for (int i = 0; i < netStruct->oplists.size(); ++i) {
        auto& op = netStruct->oplists[i];
        for (auto iter : varUpdateMap) {
            if (iter.second->name() == op->name) {
                for (int j = 0; j < netStruct->oplists.size(); ++j) {
                    auto& opSub = netStruct->oplists[j];
                    if (opSub->name == iter.first->name()) {
                        op->outputIndexes = opSub->outputIndexes;
                    }
                }
            }
        }
    }
    {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, netStruct.get());
        builder.Finish(offset);
        // TODO, use FileWriter instead
        FILE* f = fopen(argv[2], "wb");
        fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);
        fclose(f);
    }

    return 0;
}
