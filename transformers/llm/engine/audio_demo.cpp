//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>

static void saveVar(MNN::Express::VARP var, std::string name) {
    auto ptr = var->readMap<float>();
    auto size = var->getInfo()->size;
    std::ofstream os(name.c_str());
    for (int i=0; i<size; ++i) {
        os << ptr[i] << std::endl;
    }
}
static inline int layershift(int input_id, int layer, int stride = 4160, int shift=152000) {
    return input_id + shift + layer * stride;
}

static std::vector<std::vector<int>> _getInputIdsFromLogits(MNN::Transformer::Llm* llm, MNN::Express::VARP logits, int texSize, int audioSize) {
    auto info = logits->getInfo();
    auto basicOffset = info->size - info->dim[2];
    std::vector<std::vector<int>> decodeIds(8);
    decodeIds[7] = {llm->sample(logits, {}, basicOffset, texSize)};
    for (int i=0; i<7; ++i) {
        int id = llm->sample(logits, {}, basicOffset + texSize + i * audioSize, audioSize);
        id = id + texSize + i * audioSize;
        decodeIds[i] = {id};
    }
    for (int i=0; i<8; ++i) {
        FUNC_PRINT(decodeIds[i][0]);
    }
    return decodeIds;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " config.json audio.txt" << std::endl;
        return 0;
    }
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    std::shared_ptr<MNN::Express::Module> audioEncode(MNN::Express::Module::load({"audio_features"}, {"audio_embed"}, "/Users/xtjiang/alicnn/AliNNPrivate/build/whisper_mlp_fp16.mnn"));
    MNN::Express::VARP audio = MNN::Express::_Input({1, 98, 768}, MNN::Express::NCHW, halide_type_of<float>());
    {
        std::ifstream is(argv[2]);
        auto size = audio->getInfo()->size;
        auto ptr = audio->writeMap<float>();
        for (int i=0; i<size; ++i) {
            double t;
            is >> t;
            ptr[i] = t;
        }
    }
//    saveVar(audio, "audio_before.txt");
    audio = audioEncode->onForward({audio})[0];
    int T = audio->getInfo()->dim[1];
    int input_a = 4098, eoa=4096, answer_a=4099, pad_a = 4097;
    std::vector<std::vector<int>> inputIds(8);
    for (int i=0; i<7; ++i) {
        inputIds[i].resize(T+3);
        inputIds[i][0] = layershift(input_a, i);
        for (int j=0; j<T; ++j) {
            inputIds[i][j+1] = layershift(pad_a, i);
        }
        inputIds[i][T+1] = layershift(eoa, i);
        inputIds[i][T+2] = layershift(answer_a, i);
    }
    auto& inputIdsT = inputIds[7];
    inputIdsT.resize(T+3);
    inputIdsT[0] = 151938;
    for (int i=0; i<T; ++i) {
        inputIdsT[i+1] = 151937;
    }
    inputIdsT[T+1] = 151936;
    inputIdsT[T+2] = 151939;
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    llm->load();
    llm->generate_init();
    llm->switchMode(MNN::Transformer::Llm::Prefill);
    int texSize = 152000;
    int audioSize = 4160;
    // Prefill
    {
        MNN::Express::VARP embeddings;
        auto audioPtr = audio->readMap<float>();
        auto audioStride = audio->getInfo()->dim[1] * audio->getInfo()->dim[2];
        saveVar(audio, "audio.txt");
        for (int i=0; i<inputIds.size(); ++i) {
            auto embed = llm->embedding(inputIds[i]);
            if (i != 7) {
                // Replace by audio
                auto dstPtr = (float*)embed->readMap<float>() + audio->getInfo()->dim[2] * 1;
                ::memcpy(dstPtr, audioPtr, audioStride * sizeof(float));
            }
            if (0 == i) {
                embeddings = embed;
            } else {
                embeddings = embeddings + embed;
            }
        }
        embeddings = embeddings * MNN::Express::_Scalar<float>(1.0f/8.0f);
        saveVar(embeddings, "embeddings.txt");

        auto info = embeddings->getInfo();
        auto inputPos = MNN::Express::_Input({1, T+3}, MNN::Express::NCHW, halide_type_of<int>());
        for (int i=0; i<T+3; ++i) {
            inputPos->writeMap<int>()[i] = i;
        }
        llm->setKVCacheInfo(inputIds[0].size(), 0);
        auto logits = llm->forwardRaw(embeddings, llm->gen_attention_mask(T+3), inputPos);
        {
            auto info = logits->getInfo();
            auto basicOffset = info->size - info->dim[2];
            auto ptr = logits->readMap<float>() + basicOffset;

            std::ofstream os("logit.txt");
            for (int i=0; i<info->dim[2]; ++i) {
                os << ptr[i] << std::endl;
            }
        }
        inputIds = _getInputIdsFromLogits(llm.get(), logits, texSize, audioSize);
    }
    llm->switchMode(MNN::Transformer::Llm::Decode);
    std::vector<MNN::Express::VARP> embeddingsVec(inputIds.size());
    for (int index=0; index<1; ++index) {
        llm->setKVCacheInfo(1, 0);
        MNN::Express::VARP embeddings;
        for (int i=0; i<inputIds.size(); ++i) {
            auto emb = llm->embedding(inputIds[i]);
            if (0 == i) {
                embeddings = emb;
            } else {
                embeddings = embeddings + emb;
            }
        }
        embeddings = embeddings * MNN::Express::_Scalar<float>(1.0f/8.0f);
        auto inputPos = MNN::Express::_Input({1, 1}, MNN::Express::NCHW, halide_type_of<int>());
        inputPos->writeMap<int>()[0] = T+3 + index;
        auto logits = llm->forwardRaw(embeddings, llm->gen_attention_mask(1), inputPos);
        saveVar(logits, "logit_decode.txt");
        auto size = logits->getInfo()->size;
        auto ptr = logits->readMap<float>();
        auto basicOffset = logits->getInfo()->size - logits->getInfo()->dim[2];
        auto texId = llm->sample(logits, {}, basicOffset, texSize);
        inputIds = _getInputIdsFromLogits(llm.get(), logits, texSize, audioSize);
        FUNC_PRINT_ALL(llm->tokenizer_decode(inputIds[7][0]).c_str(), s);
    }
    
    return 0;
}
