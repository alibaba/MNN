//
//  TestConvertResult.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "PostConverter.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include "common/MemoryFormater.h"
#include "cli.hpp"
using namespace MNN::Express;
using namespace MNN;

static bool compareOutput(VARP output, const std::string& directName, const std::string& name, Dimensionformat dataFormat, int order) {
    auto info = output->getInfo();
    auto ptr = output->readMap<float>();
    if (nullptr == info || nullptr == ptr) {
        MNN_ERROR("TESTERROR ptr / info nullptr\n");
        return false;
    }
    std::ifstream outputOrigin;
    // First find key
    {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << name <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    // Second find order
    if (outputOrigin.fail()) {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << order <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    MNN_ASSERT(!outputOrigin.fail());
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
    auto outputPtr = output->readMap<float>();
    auto diffAbsMax = _ReduceMax(diff);
    auto absMaxV = absMax->readMap<float>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
    if (absMaxV * 0.01f < diffAbsMaxV || std::isnan(absMaxV)) {
        MNN_ERROR("TESTERROR %s value error : absMaxV:%f - DiffMax %f\n", name.c_str(), absMaxV, diffAbsMaxV);

        MNN_PRINT("expected value\n");
        formatMatrix(targetValue->readMap<float>(), targetValue->getInfo()->dim);
        MNN_PRINT("real value\n");
        formatMatrix(output->readMap<float>(), output->getInfo()->dim);

        return false;
    }
    return true;
}
int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./TestConvertResult [Onnx, Tf, Tflite, Torch] ${Dir}\n");
        return 0;
    }
    std::string inputType = argv[1];
    std::string directName = argv[2];
    auto inputModel = modelConfig::ONNX;
    auto suffix = ".onnx";
    auto dataFormat = NCHW;
    if (inputType == "Tf") {
        inputModel = modelConfig::TENSORFLOW;
        suffix = ".pb";
        dataFormat = NHWC;
    } else if (inputType == "Tflite") {
        inputModel = modelConfig::TFLITE;
        suffix = ".tflite";
        dataFormat = NHWC;
    } else if (inputType == "Torch") {
        inputModel = modelConfig::TORCH;
        suffix = ".pt";
    }
    MNN_PRINT("Test %s\n", directName.c_str());
    std::string defaultCacheFile = ".___temp.mnn";
    {
        modelConfig modelPath;
        modelPath.model = inputModel;
        std::ostringstream modelNameOs;
        modelNameOs << directName << "/test" << suffix;
        modelPath.modelFile = modelNameOs.str();
        modelPath.MNNModel = defaultCacheFile;
        modelPath.keepInputFormat = true;
        Cli::convertModel(modelPath);
    }
    bool useControlFlow = false;
    rapidjson::Document document;
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    {
        std::ostringstream jsonNameOs;
        jsonNameOs << directName << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        if (document.HasMember("controlflow")) {
            useControlFlow = document["controlflow"].GetBool();
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
    }

    // create session
    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = 2;
    // If type not fount, let it failed
    config.backupType = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    // config.path.outputs.push_back("ResizeBilinear_2");
    // backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(1);
    // backendConfig.memory = BackendConfig::Memory_High;
    config.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = true;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    std::shared_ptr<Module> net(Module::load(inputNames, outputNames, defaultCacheFile.c_str(), rtmgr, &mConfig));
    auto mInfo = net->getInfo();
    std::vector<VARP> inputs(mInfo->inputs.size());
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
    // Load inputs
    for (int i=0; i<inputs.size(); ++i) {
        auto inputName = inputNames[i];
        // Resize
        auto shapeIter = inputShape.find(inputName);
        if (shapeIter != inputShape.end()) {
            auto s = shapeIter->second;
            inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
        } else {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        auto info = inputs[i]->getInfo();
        auto iter = inputInfo.find(inputNames[i]);
        if (iter != inputInfo.end()) {
            auto ptr = inputs[i]->writeMap<float>();
            for (int v=0; v<mInfo->inputs[i].size; ++v) {
                ptr[v] = iter->second;
            }
            continue;
        }
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
        auto name = outputNames[i];
        auto v = outputs[i];
        auto info = v->getInfo();
        if (nullptr == info) {
            continue;
        }
        if (info->order == NC4HW4 && info->dim.size() > 1) {
            v = _Convert(v, mInfo->defaultFormat);
        }
        if (info->type.code != halide_type_float) {
            v = _Cast<float>(v);
        }
        v.fix(VARP::CONSTANT);
        outputs[i] = v;
    }

    for (int i=0; i<outputNames.size(); ++i) {
        auto output = outputs[i];
        bool success = compareOutput(output, directName, outputNames[i], mInfo->defaultFormat, i);
        if (!success) {
            modelError = true;
            MNN_ERROR("Error for output %s\n", outputNames[i].c_str());
        }
    }

    if (modelError) {
        MNN_ERROR("Save mnn result to  .error director\n");
        for (int i=0; i<outputNames.size(); ++i) {
            auto v = outputs[i];
            auto name = outputNames[i];
            auto info = v->getInfo();
            if (nullptr == info) {
                continue;
            }
            if (info->order == NC4HW4 && info->dim.size() > 1) {
                v = _Convert(v, dataFormat);
            }
            if (info->type.code != halide_type_float) {
                v = _Cast<float>(v);
            }
            v.fix(VARP::CONSTANT);
            info = v->getInfo();
            std::ofstream _output((".error/" + name + ".txt").c_str());
            auto ptr = v->readMap<float>();
            for (int v=0; v<info->size; ++v) {
                _output << ptr[v] << "\n";
            }
            v->setName(name);
            outputs.emplace_back(v);
        }
        Variable::save(outputs, ".Error.mnn");
        return 0;
    }
    MNN_PRINT("TEST_SUCCESS\n");
    return 0;
}

