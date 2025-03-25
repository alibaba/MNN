//
//  backendTest.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include "MNN_generated.h"
#include <MNN/expr/Module.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "core/Session.hpp"
#include "rapidjson/document.h"
typedef std::vector<std::pair<std::string, std::vector<std::string>>> OUTPUTCONFIG;

static OUTPUTCONFIG _getAllOutputs(const MNN::Net* net, const MNN::Session* session) {
    auto info = session->getPipelineInfo(0);
    std::vector<std::pair<std::string, std::vector<std::string>>> res;
    auto tensorName = net->tensorName();
    auto oplist = net->oplists();
    if (nullptr == oplist || nullptr == tensorName) {
        FUNC_PRINT(1);
        return res;
    }
    for (int i=0; i<info.second.size(); ++i) {
        auto& unit = info.second[i];
        if (unit.type != MNN::Schedule::SEPARATE) {
            continue;
        }
        auto op = unit.op;
        if (op->type() == MNN::OpType_Const || op->type() == MNN::OpType_TrainableParam || op->type() == MNN::OpType_Input) {
            continue;
        }
        if (nullptr == op->outputIndexes() || op->outputIndexes()->size() == 0) {
            continue;
        }
        std::vector<std::string> outputNames(op->outputIndexes()->size());
        for (int v=0; v<op->outputIndexes()->size(); ++v) {
            auto index = op->outputIndexes()->data()[v];
            outputNames[v] = tensorName->GetAsString(index)->str();
        }
        res.emplace_back(std::make_pair(op->name()->str(), outputNames));
    }
    return res;
}
static std::vector<std::string> _getAllInputs(const MNN::Net* net) {
    auto tensorName = net->tensorName();
    auto oplist = net->oplists();
    std::vector<std::string> res;
    if (nullptr == oplist || nullptr == tensorName) {
        FUNC_PRINT(1);
        return res;
    }
    for (int i=0; i<oplist->size(); ++i) {
        auto op = oplist->GetAs<MNN::Op>(i);
        if (op->type() == MNN::OpType_Input) {
            auto index = op->outputIndexes()->data()[0];
            res.emplace_back(tensorName->GetAsString(index)->str());
        }
    }
    return res;
}

template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}

using namespace MNN;

static void _zeroInputs(const Interpreter* net, const Session* session) {
    // Set Other Inputs to Zero
    auto allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto inputTensor = iter.second;
        auto size = inputTensor->size();
        if (size <= 0) {
            continue;
        }
        MNN::Tensor tempTensor(inputTensor, inputTensor->getDimensionType());
        ::memset(tempTensor.host<void>(), 0, tempTensor.size());
        inputTensor->copyFromHostTensor(&tempTensor);
    }
}
static void compareForwadType(OUTPUTCONFIG outputNames, Interpreter* net, MNNForwardType expectType, MNNForwardType compareType, float tolerance,
                              const std::map<std::string, std::shared_ptr<Tensor>>& inputs, const std::string& stopOp, BackendConfig::PrecisionMode precision, int modeNum) {
    auto inputNames = _getAllInputs(MNN::GetNet(net->getModelBuffer().first));
    for (int v=0; v<outputNames.size(); ++v) {
        auto outputName = outputNames[v].second;
        auto opName = outputNames[v].first;
        MNN::ScheduleConfig expectConfig, compareConfig;
        BackendConfig backendConfig;
        backendConfig.precision = precision;
        expectConfig.type   = expectType;
        expectConfig.path.inputs = inputNames;
        expectConfig.path.outputs = outputName;
        expectConfig.saveTensors = outputName;
        expectConfig.path.mode = MNN::ScheduleConfig::Path::Tensor;

        compareConfig.type  = compareType;
        compareConfig.backendConfig = &backendConfig;
        compareConfig.mode = modeNum;
        compareConfig.path.inputs = inputNames;
        compareConfig.path.outputs = outputName;
        compareConfig.saveTensors = outputName;
        compareConfig.path.mode = MNN::ScheduleConfig::Path::Tensor;
        auto expectSession  = net->createSession(expectConfig);
        auto compareSession = net->createSession(compareConfig);
        _zeroInputs(net, expectSession);
        _zeroInputs(net, compareSession);
        for (auto& iter : inputs) {
            Tensor* expectInput = net->getSessionInput(expectSession, iter.first.empty() ? NULL : iter.first.c_str());
            expectInput->copyFromHostTensor(iter.second.get());
            Tensor* compareInput = net->getSessionInput(compareSession, iter.first.empty() ? NULL : iter.first.c_str());
            compareInput->copyFromHostTensor(iter.second.get());
        }
        net->runSession(expectSession);
        net->runSession(compareSession);
        bool allCorrect = true;
        bool outputValid = false;
        auto compare = [&]() {
            for(auto name : outputName) {
                auto expectTensor = net->getSessionOutput(expectSession, name.c_str());
                if (nullptr == expectTensor || expectTensor->host<void>() == nullptr) {
                    MNN_ERROR("Can't compare tensor: %s\n", name.c_str());
                    continue;
                }
                outputValid = true;
                auto compareTensor = net->getSessionOutput(compareSession, name.c_str());
                if (nullptr == compareTensor) {
                    MNN_ERROR("%d [%s] Tensor %s invalid\n", v, opName.c_str(), name.c_str());
                    allCorrect = false;
                    break;
                }
                auto correct      = TensorUtils::compareTensors(compareTensor, expectTensor, tolerance, true);
                if (!correct) {
                    MNN_PRINT("%d [%s] Op outputs %s is error\n", v, opName.c_str(), name.c_str());
                    allCorrect = false;
                    break;
                }
            }
        };
        compare();
        if (!outputValid) {
            net->releaseSession(expectSession);
            net->releaseSession(compareSession);
            continue;
        }

        if (allCorrect) {
            MNN_PRINT("Correct ! Run second pass\n");
        } else {
            return;
        }
        for (auto& iter : inputs) {
            Tensor* compareInput = net->getSessionInput(compareSession, iter.first.empty() ? NULL : iter.first.c_str());
            compareInput->copyFromHostTensor(iter.second.get());
        }
        net->runSession(compareSession);
        compare();
        if (allCorrect) {
            MNN_PRINT("Correct for %d, name=%s\n", v, opName.c_str());
        } else {
            return;
        }
        net->releaseSession(expectSession);
        net->releaseSession(compareSession);
    }
    MNN_PRINT("Correct !\n");
}

int main(int argc, const char* argv[]) {
    // read args
    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    const char* fileName = argv[1];

    auto type = MNN_FORWARD_CPU;
    if (argc > 2) {
        type = (MNNForwardType)stringConvert<int>(argv[2]);
    }
    MNN_PRINT("Test forward type: %d\n", type);

    float tolerance = 0.05f;
    if (argc > 3) {
        tolerance = stringConvert<float>(argv[3]);
    }
    MNN_PRINT("Tolerance Rate: %f\n", tolerance);

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName));
    net->setSessionMode(Interpreter::Session_Debug);

    // create session
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    std::map<std::string, std::shared_ptr<MNN::Tensor>> inputs;
    std::vector<std::string> inputNames;
    do {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << pwd << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        if (fileNames.fail()) {
            break;
        }
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            break;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
            }
        }
    } while (false);
    if (!inputNames.empty()) {
        MNN_PRINT("Find input.json, use inputs:");
        for (auto& n : inputNames) {
            MNN_PRINT(" %s, ", n.c_str());
        }
        MNN_PRINT("\n");
        for (auto name : inputNames) {
            auto inputTensor = net->getSessionInput(session, name.c_str());
            std::shared_ptr<MNN::Tensor> givenTensor(new Tensor(inputTensor, inputTensor->getDimensionType()));
            {
                std::ostringstream fileName;
                fileName << pwd << name << ".txt";
                std::ifstream input(fileName.str().c_str());
                MNN_ASSERT(!input.fail());

                int size_w = inputTensor->width();
                int size_h = inputTensor->height();
                int bpp    = inputTensor->channel();
                int batch  = inputTensor->batch();
                // auto backend = net->getBackend(session, inputTensor);
                // MNN_ASSERT(!input.fail());
                MNN_PRINT("Input: %d,%d,%d,%d\n", size_w, size_h, bpp, batch);
                auto inputData = givenTensor->host<float>();
                auto size      = givenTensor->size() / sizeof(float);
                for (int i = 0; i < size; ++i) {
                    input >> inputData[i];
                }
                inputs.insert(std::make_pair(name, givenTensor));
            }

        }
    } else {
        auto inputTensor = net->getSessionInput(session, NULL);
        std::shared_ptr<MNN::Tensor> givenTensor(new Tensor(inputTensor, inputTensor->getDimensionType()));
        {
            std::ostringstream fileName;
            fileName << pwd << "input_0"
                     << ".txt";
            std::ifstream input(fileName.str().c_str());

            int size_w = inputTensor->width();
            int size_h = inputTensor->height();
            int bpp    = inputTensor->channel();
            int batch  = inputTensor->batch();
            // auto backend = net->getBackend(session, inputTensor);
            // MNN_ASSERT(!input.fail());
            MNN_PRINT("Input: %d,%d,%d,%d\n", size_w, size_h, bpp, batch);
            auto inputData = givenTensor->host<float>();
            auto size      = givenTensor->size() / sizeof(float);
            for (int i = 0; i < size; ++i) {
                input >> inputData[i];
            }
            inputs.insert(std::make_pair("", givenTensor));
        }
    }
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    if (argc > 4) {
        precision = (BackendConfig::PrecisionMode)atoi(argv[4]);
    }
    FUNC_PRINT(precision);
    int modeNum = 1;
    if(argc > 5) {
        modeNum = atoi(argv[5]);//set gpu mode
    }
    FUNC_PRINT(modeNum);
    std::string stopOp = "";
    if (argc > 6) {
        stopOp = argv[6];
    }
    FUNC_PRINT_ALL(stopOp.c_str(), s);
    auto outputNames = _getAllOutputs(MNN::GetNet(net->getModelBuffer().first), session);

    net->releaseSession(session);
    compareForwadType(outputNames, net.get(), MNN_FORWARD_CPU, type, tolerance, inputs, stopOp, precision, modeNum);

    return 0;
}
