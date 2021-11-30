//
//  SequenceModuleTest.cpp
//  MNN
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
#include <cmath>
using namespace MNN::Express;
using namespace MNN;

static bool compareOutput(VARP output, const std::string& directName, const std::string& name, Dimensionformat dataFormat, int order) {
    auto info = output->getInfo();
    auto ptr = output->readMap<float>();
    if (nullptr == info || nullptr == ptr) {
        MNN_ERROR("TESTERROR ptr / info nullptr\n");
        return false;
    }
    std::string targetFileName;
    std::ifstream outputOrigin;
    // First find key
    {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << name <<".txt";
        targetFileName = outputFileOs.str();
        outputOrigin.open(targetFileName.c_str());
    }
    // Second find order
    if (outputOrigin.fail()) {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << order <<".txt";
        targetFileName = outputFileOs.str();
        outputOrigin.open(targetFileName.c_str());
    }
    if (info->order == NC4HW4 && info->dim.size() > 1) {
        output = _Convert(output, dataFormat);
        info = output->getInfo();
    }
    if (info->type.code != halide_type_float) {
        output = _Cast<float>(output);
        info = output->getInfo();
    }
    MNN_PRINT("%s: (", name.c_str());
    for (int i=0; i<info->dim.size(); ++i) {
        MNN_PRINT("%d, ", info->dim[i]);
    }
    MNN_PRINT(")\n");
    auto targetValue = _Input({info->dim}, info->order, info->type);
    auto targetPtr = targetValue->writeMap<float>();
    for (int i=0; i<info->size; ++i) {
        outputOrigin >> targetPtr[i];
    }
    auto absMax = _ReduceMax(_Abs(targetValue), {});
    absMax = _Maximum(absMax, _Scalar<float>(0.0001f));
    auto diff = _Abs(targetValue - output);
    auto diffAbsMax = _ReduceMax(diff);
    auto absMaxV = absMax->readMap<float>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
    if (absMaxV * 0.01f < diffAbsMaxV || std::isnan(absMaxV)) {
        MNN_ERROR("TESTERROR from %s value error : absMaxV:%f - DiffMax %f\n", targetFileName.c_str(), absMaxV, diffAbsMaxV);
        return false;
    }
    return true;
}
int main(int argc, char *argv[]) {
    if (argc < 5) {
        MNN_ERROR("Usage: ./SequenceModuleTest.out ${test.mnn} [forwardType] [shapeMutable] ${Dir} ${Dir1} ......\n");
        return 0;
    }
    std::string modelName = argv[1];
    auto type = (MNNForwardType)atoi(argv[2]);
    auto shapeMutable = atoi(argv[3]);
    MNN_PRINT("Test %s, type = %d\n", modelName.c_str(), type);
    // create session
    MNN::ScheduleConfig config;
    config.type      = type;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = 1;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    config.backendConfig     = &backendConfig;
    
    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = shapeMutable;
    mConfig.rearrange = true;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    std::shared_ptr<Module> net;

    for (int index = 4; index < argc; ++index) {
        MNN_PRINT("Test for %s\n", argv[index]);
        std::string directName = argv[index];
        rapidjson::Document document;
        std::map<std::string, float> inputInfo;
        std::map<std::string, std::vector<int>> inputShape;
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        std::ostringstream jsonNameOs;
        jsonNameOs << argv[index] << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            continue;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
                MNN_PRINT("%s\n", name.c_str());
                if (obj.HasMember("value")) {
                    float value = obj["value"].GetFloat();
                    inputInfo.insert(std::make_pair(name, value));
                }
                if (obj.HasMember("shape")) {
                    auto dims = obj["shape"].GetArray();
                    std::vector<int> shapes;
                    for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                        shapes.emplace_back(iter->GetInt());
                    }
                    inputShape.insert(std::make_pair(name, shapes));
                }
            }
        }
        if (document.HasMember("outputs")) {
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
        if (nullptr == net.get()) {
            net.reset(Module::load(inputNames, outputNames, modelName.c_str(), rtmgr, &mConfig));
            if (net == nullptr) {
                MNN_PRINT("Error: can't load module\n");
                return 0;
            }
        }
        auto mInfo = net->getInfo();

#define LOAD_DATA(TYPE)\
    if (inputInfo.find(inputName) != inputInfo.end()) {\
        auto value = inputInfo[inputName];\
        for (int i=0; i<info->size; ++i) {\
            ptr[i] = value;\
        }\
    } else {\
        std::ostringstream fileNameOs;\
        fileNameOs << directName << "/" << inputName << ".txt";\
        auto fileName = fileNameOs.str();\
        std::ifstream inputOs(fileName.c_str());\
        if (inputOs.fail()) {\
            MNN_ERROR("TESTERROR Can't open %s\n", fileName.c_str());\
            continue;\
        }\
        for (int i=0; i<info->size; ++i) {\
            inputOs >> ptr[i];\
        }\
    }
        std::vector<VARP> inputs(mInfo->inputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        // Load inputs
        for (int i=0; i<inputs.size(); ++i) {
            auto inputName = inputNames[i];
            // Resize
            auto shapeIter = inputShape.find(inputName);
            if (shapeIter != inputShape.end()) {
                auto s = shapeIter->second;
                inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
            }
            auto info = inputs[i]->getInfo();
            if (info->type == halide_type_of<float>()){
                auto ptr = inputs[i]->writeMap<float>();
                LOAD_DATA(float)
            } else {
                auto floatVar = _Input(info->dim, info->order, halide_type_of<float>());
                auto ptr = floatVar->writeMap<float>();
                LOAD_DATA(float)
                auto temp = _Cast(floatVar, info->type);
                inputs[i]->input(temp);
            }
            inputs[i] = _Convert(inputs[i], mInfo->inputs[i].order);
        }
#undef LOAD_DATA
        bool modelError = false;
        // Module Branch
        auto outputs = net->onForward(inputs);
        for (int i=0; i<outputNames.size(); ++i) {
            auto output = outputs[i];
            bool success = compareOutput(output, directName, outputNames[i], mInfo->defaultFormat, i);
            if (!success) {
                modelError = true;
                MNN_ERROR("Error for output %s\n", outputNames[i].c_str());
            }
        }
    }

    return 0;
}

