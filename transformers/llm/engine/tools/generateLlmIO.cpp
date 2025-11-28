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

#include <limits>

static void saveInputOutputs(const MNN::Express::Module::Info* info, std::vector<MNN::Express::VARP> inputs, std::vector<MNN::Express::VARP> outputs, const std::string & outputDir, int index) {
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i=0; i<info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i=0; i<info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, std::to_string(index));
    if (!(MNNCreateDir(subDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

static void createInputsForLLM(int seqLen, int hiddenSize, const std::string& attentionMaskType, bool lastLogit, std::vector<MNN::Express::VARP>& inputs) {
    if (attentionMaskType != "float") {
        MNN_ERROR("Don't support Attention Mask Type other than 'float', currently.\n");
        return;
    }

    MNN::Express::VARP inputIdx = MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
    float * inputIdxData = inputIdx->writeMap<float>();
    for (int i = 0; i < seqLen * hiddenSize; ++i) {
        inputIdxData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputIdx);

    MNN::Express::VARP attentionMask =  MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    float * attentionMaskData = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            attentionMaskData[i * seqLen + j] = (j > i) * std::numeric_limits<float>::lowest();
        }
    }
    inputs.push_back(attentionMask);

    MNN::Express::VARP positionIds = MNN::Express::_Input({seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    int * positionIdsData = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; i++) {
        positionIdsData[i] = i;
    }
    inputs.push_back(positionIds);

    int logitsIndexValue = lastLogit ? -1 : 0;
    MNN::Express::VARP logitsIndex = MNN::Express::_Const((const void *) &logitsIndexValue, {1}, MNN::Express::NHWC, halide_type_of<int>());
    inputs.push_back(logitsIndex);

    return;
}

static void generateForLLM(const std::string& modelPath, const std::string& outputDir, const std::string& jsonPath, int blockSize) {
    std::shared_ptr<MNN::Express::Module> net;
    std::vector<std::string> inputNames = {"input_ids", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames = {"logits"};

    int hiddenSize;
    std::string attentionMaskType;
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            MNN_ERROR("Failed to open JSON config file: %s.\n", jsonPath.c_str());
            return;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if (doc.HasParseError() || !doc.IsObject()) {
            MNN_ERROR("Failed to parse JSON config file: %s.\n", jsonPath.c_str());
            return;
        }

        if (!doc.HasMember("hidden_size") || !doc["hidden_size"].IsInt()) {
            MNN_ERROR("'hidden_size' not found or not an integer in %s\n", jsonPath.c_str());
            return;
        }
        hiddenSize = doc["hidden_size"].GetInt();

        if (!doc.HasMember("attention_mask") || !doc["attention_mask"].IsString()) {
            MNN_ERROR("'attention_mask' not found or not a string in %s\n", jsonPath.c_str());
            return;
        }
        attentionMaskType = doc["attention_mask"].GetString();
    }

    // Load Model.
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());
    net.reset(MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr), MNN::Express::Module::destroy);

    {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForLLM(blockSize, hiddenSize, attentionMaskType, false, inputs);
        outputs = net->onForward(inputs);
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, blockSize);
    }

    {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForLLM(1, hiddenSize, attentionMaskType, true, inputs);
        outputs = net->onForward(inputs);
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
    }

    return;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateLlmIO model/config.json outputDir [blocksize]\n");
        MNN_PRINT("This program generates IO test data, i.e. input.mnn and output.mnn, for a given llm model, assuming standard inputs ('inputs_ids', 'attention_mask', 'position_ids', 'logits_index') and standard outputs('logits').\n");
        return 1;
    }

    srand(time(NULL));
    int blockSize = 128;
    if (argc >= 4) {
        blockSize = atoi(argv[3]);
    }
    FUNC_PRINT(blockSize);

    std::string modelPath = std::string(argv[1]) + "/llm.mnn";
    std::string llmConfigPath = std::string(argv[1]) + "/llm_config.json";
    FUNC_PRINT_ALL(modelPath.c_str(), s);
    FUNC_PRINT_ALL(llmConfigPath.c_str(), s);
    std::string outputDir = argv[2];

    if (!(MNNCreateDir(outputDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    generateForLLM(modelPath, outputDir, llmConfigPath, blockSize);

    return 0;
}
