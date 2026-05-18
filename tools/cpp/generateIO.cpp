#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include <iostream>
#include <sstream>

#include <limits>

static void saveInputOutputs(const MNN::Express::Module::Info* info, std::vector<MNN::Express::VARP> inputs,
                             std::vector<MNN::Express::VARP> outputs, const std::string& outputDir) {
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i = 0; i < info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i = 0; i < info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }

    std::string inputPath = MNNFilePathConcat(outputDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(outputDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./generateLlmIO model inputJson outputDir externalFilePath\n");
        return 1;
    }

    std::string ExternalFilePath;
    std::string modelPath = std::string(argv[1]);
    std::string inputJson = argv[2];
    std::string outputDir = argv[3];
    if (argc >= 5) {
        ExternalFilePath = argv[4];
    }

    rapidjson::Document document;
    std::ifstream fileNames(inputJson.c_str());
    std::ostringstream output;
    output << fileNames.rdbuf();
    auto outputStr = output.str();
    document.Parse(outputStr.c_str());
    if (document.HasParseError()) {
        MNN_ERROR("Invalid json\n");
        return 0;
    }
    int shapeIndex = 0;
    std::shared_ptr<MNN::Express::Module> net;

    if (document.HasMember("configs")) {
        if (!(MNNCreateDir(outputDir.c_str()))) {
            MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
        }
        auto configsArray = document["configs"].GetArray();
        for (auto& configObj : configsArray) {
            std::map<std::string, float> inputInfo;
            std::map<std::string, std::string> inputType;
            std::vector<std::string> inputNames;
            std::vector<std::string> outputNames;
            std::map<std::string, std::vector<int>> inputShape;
            std::vector<MNN::Express::VARP> inputs;
            std::vector<MNN::Express::VARP> outputs;
            if (configObj.HasMember("inputs")) {
                auto inputsInfo = configObj["inputs"].GetArray();
                for (auto iter = inputsInfo.begin(); iter != inputsInfo.end(); iter++) {
                    auto obj = iter->GetObject();
                    std::string type = "float";
                    std::string name = obj["name"].GetString();
                    inputNames.emplace_back(name);
                    if (obj.HasMember("type")) {
                        type = obj["type"].GetString();
                        inputType.insert(std::make_pair(name, type));
                    }
                    if (obj.HasMember("value")) {
                        float value;
                        if (type == "int") {
                            value = (float)obj["value"].GetInt();
                        } else {
                            value = obj["value"].GetFloat();
                        }
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
            if (configObj.HasMember("outputs")) {
                auto array = configObj["outputs"].GetArray();
                for (auto iter = array.begin(); iter != array.end(); iter++) {
                    std::string name = iter->GetString();
                    outputNames.emplace_back(name);
                }
            }

            // Load Model.
            if (net.get() == nullptr) {
                MNN::ScheduleConfig config;
                std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(
                    MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
                if (ExternalFilePath.length() > 0) {
                    rtmgr->setExternalFile(ExternalFilePath.c_str());
                }
                net.reset(MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr),
                          MNN::Express::Module::destroy);
            }

            auto mInfo = net->getInfo();
            // create input
            inputs.resize(mInfo->inputs.size());
            for (int i = 0; i < inputs.size(); ++i) {
                inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
            }
            // Load inputs
            for (int i = 0; i < inputs.size(); ++i) {
                auto inputName = inputNames[i];
                std::string type = "float";
                auto typeIter = inputType.find(inputName);
                if (typeIter != inputType.end()) {
                    type = typeIter->second;
                }
                // Resize
                auto shapeIter = inputShape.find(inputName);
                if (shapeIter != inputShape.end()) {
                    auto s = shapeIter->second;
                    inputs[i] = _Input(s, mInfo->inputs[i].order, mInfo->inputs[i].type);
                }
                auto info = inputs[i]->getInfo();
                if (inputInfo.find(inputName) != inputInfo.end()) {
                    auto value = inputInfo[inputName];
                    if (type == "int") {
                        auto ptr = inputs[i]->writeMap<int>();
                        for (int i = 0; i < info->size; ++i) {
                            ptr[i] = (int)value;
                        }
                    } else {
                        auto ptr = inputs[i]->writeMap<float>();
                        for (int i = 0; i < info->size; ++i) {
                            ptr[i] = (float)value;
                        }
                    }
                }
            }

            std::string outputDirtmp = MNNFilePathConcat(outputDir, std::to_string(shapeIndex++));
            if (!(MNNCreateDir(outputDirtmp.c_str()))) {
                MNN_PRINT("Failed to create dir %s.\n", outputDirtmp.c_str());
            }

            outputs = net->onForward(inputs);
            saveInputOutputs(net->getInfo(), inputs, outputs, outputDirtmp);
        }
    }

    return 0;
}
