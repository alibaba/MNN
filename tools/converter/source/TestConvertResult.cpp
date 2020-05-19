//
//  TestConvertResult.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "PostConverter.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
using namespace MNN::Express;
int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./TestConvertResult Onnx dir\n");
        return 0;
    }
    std::string directName = argv[2];
    MNN_PRINT("Test %s\n", directName.c_str());
    std::string defaultCacheFile = ".___temp.mnn";
    {
        std::ostringstream modelNameOs;
        modelNameOs << directName << "/test.onnx";
        std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
        onnx2MNNNet(modelNameOs.str().c_str(), "Test", netT);
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, false);
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, newNet.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();

        std::ofstream output(defaultCacheFile.c_str(), std::ofstream::binary);
        output.write((const char*)bufferOutput, sizeOutput);
    }
    rapidjson::Document document;
    std::map<std::string, float> inputInfo;
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
    auto varMap = Variable::loadMap(defaultCacheFile.c_str());
    for (auto inputName : inputNames) {
        if (varMap.find(inputName) == varMap.end()) {
            MNN_ERROR("TESTERROR Can't find var: %s\n", inputName.c_str());
            continue;
        }
        varMap[inputName] = _ChangeInputFormat(varMap[inputName], NCHW);
        auto info = varMap[inputName]->getInfo();
        auto ptr = varMap[inputName]->writeMap<float>();
        if (inputInfo.find(inputName) != inputInfo.end()) {
            auto value = inputInfo[inputName];
            for (int i=0; i<info->size; ++i) {
                ptr[i] = value;
            }
        } else {
            std::ostringstream fileNameOs;
            fileNameOs << directName << "/" << inputName << ".txt";
            auto fileName = fileNameOs.str();
            std::ifstream inputOs(fileName.c_str());
            if (inputOs.fail()) {
                MNN_ERROR("TESTERROR Can't open %s\n", fileName.c_str());
                continue;
            }
            for (int i=0; i<info->size; ++i) {
                inputOs >> ptr[i];
            }
        }
    }
    for (int i=0; i<outputNames.size(); ++i) {
        auto name = outputNames[i];
        if (varMap.find(name) == varMap.end()) {
            MNN_ERROR("TESTERROR, Can't find var: %s\n", name.c_str());
            return 0;
        }
        auto output = varMap[name];
        auto info = output->getInfo();
        auto ptr = output->readMap<float>();
        if (nullptr == info || nullptr == ptr) {
            MNN_ERROR("TESTERROR ptr / info nullptr\n");
            return 0;
        }
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << i <<".txt";
        std::ifstream outputOrigin(outputFileOs.str().c_str());
        if (info->order == NC4HW4) {
            output = _Convert(output, NCHW);
            info = output->getInfo();
        }
        auto targetValue = _Input({info->dim}, info->order, info->type);
        auto targetPtr = targetValue->writeMap<float>();
        for (int i=0; i<info->size; ++i) {
            outputOrigin >> targetPtr[i];
        }
        auto absMax = _ReduceMax(_Abs(targetValue), {});
        auto diff = _Abs(targetValue - output);
        auto diffAbsMax = _ReduceMax(diff);
        auto absMaxV = absMax->readMap<float>()[0];
        auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
        if (absMaxV * 0.01f < diffAbsMaxV) {
            MNN_ERROR("TESTERROR %s value error : absMaxV:%f - DiffMax %f\n", name.c_str(), absMaxV, diffAbsMaxV);
            return 0;
        }
    }
    MNN_PRINT("TEST_SUCCESS\n");
    return 0;
}

