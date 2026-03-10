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

    MNN::Express::VARP positionIds = MNN::Express::_Input({1, seqLen}, MNN::Express::NCHW, halide_type_of<int>());
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

static void createInputsForEmbedding(int seqLen, int hiddenSize, const std::string& attentionMaskType, std::vector<MNN::Express::VARP>& inputs) {
    MNN::Express::VARP inputEmbeds = MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
    float* inputEmbedsData = inputEmbeds->writeMap<float>();
    for (int i = 0; i < seqLen * hiddenSize; ++i) {
        inputEmbedsData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputEmbeds);

    MNN::Express::VARP attentionMask = MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    float* attentionMaskData = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            if (attentionMaskType == "float") {
                attentionMaskData[i * seqLen + j] = (j > i) ? std::numeric_limits<float>::lowest() : 0.0f;
            } else {
                attentionMaskData[i * seqLen + j] = 1.0f;
            }
        }
    }
    inputs.push_back(attentionMask);

    MNN::Express::VARP positionIds = MNN::Express::_Input({1, seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    int* positionIdsData = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        positionIdsData[i] = i;
    }
    inputs.push_back(positionIds);
}

static bool isEmbeddingModel(const rapidjson::Document& doc) {
    if (doc.HasMember("output_names") && doc["output_names"].IsArray()) {
        for (auto iter = doc["output_names"].Begin(); iter != doc["output_names"].End(); ++iter) {
            if (iter->IsString() && std::string(iter->GetString()) == "sentence_embeddings") {
                return true;
            }
        }
    }
    auto modelType = std::string(doc.HasMember("model_type") && doc["model_type"].IsString() ? doc["model_type"].GetString() : "");
    if (modelType == "bert" || modelType == "new" || modelType == "qwen3") {
        return true;
    }
    return false;
}

static bool generateForModel(const std::string& modelPath, const std::string& outputDir, const std::string& jsonPath, int blockSize) {
    std::shared_ptr<MNN::Express::Module> net;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    bool isEmbedding = false;

    int hiddenSize;
    std::string attentionMaskType;
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            MNN_ERROR("Failed to open JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if (doc.HasParseError() || !doc.IsObject()) {
            MNN_ERROR("Failed to parse JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }

        if (!doc.HasMember("hidden_size") || !doc["hidden_size"].IsInt()) {
            MNN_ERROR("'hidden_size' not found or not an integer in %s\n", jsonPath.c_str());
            return false;
        }
        hiddenSize = doc["hidden_size"].GetInt();

        if (!doc.HasMember("attention_mask") || !doc["attention_mask"].IsString()) {
            MNN_ERROR("'attention_mask' not found or not a string in %s\n", jsonPath.c_str());
            return false;
        }
        attentionMaskType = doc["attention_mask"].GetString();

        isEmbedding = isEmbeddingModel(doc);
    }

    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());

    if (isEmbedding) {
        inputNames = {"input_ids", "attention_mask", "position_ids"};
        outputNames = {"sentence_embeddings"};
    } else {
        inputNames = {"input_ids", "attention_mask", "position_ids", "logits_index"};
        outputNames = {"logits"};
    }
    net.reset(MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr), MNN::Express::Module::destroy);
    if (nullptr == net.get()) {
        MNN_ERROR("Failed to load module for QNN IO generation as %s model.\n", isEmbedding ? "embedding" : "llm");
        return false;
    }

    {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        if (isEmbedding) {
            createInputsForEmbedding(blockSize, hiddenSize, attentionMaskType, inputs);
        } else {
            createInputsForLLM(blockSize, hiddenSize, attentionMaskType, false, inputs);
        }
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, blockSize);
    }

    if (!isEmbedding) {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForLLM(1, hiddenSize, attentionMaskType, true, inputs);
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run decode forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
    }

    if (isEmbedding) {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForEmbedding(1, hiddenSize, attentionMaskType, inputs);
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run single token embedding forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateLlmIO model/config.json outputDir [blocksize]\n");
        MNN_PRINT("This program generates IO test data for QNN export. It supports both generation models and embedding models exported by llmexport.\n");
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

    if (!generateForModel(modelPath, outputDir, llmConfigPath, blockSize)) {
        return 1;
    }

    return 0;
}
