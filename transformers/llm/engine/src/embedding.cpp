//
//  embedding.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "prompt.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"

namespace MNN {
using namespace Express;
namespace Transformer {

float Embedding::dist(VARP var0, VARP var1) {
    auto distVar = _Sqrt(_ReduceSum(_Square(var0 - var1)));
    auto dist    = distVar->readMap<float>()[0];
    return dist;
}

float Embedding::cos_sim(VARP var0, VARP var1) {
    auto innerProd = _ReduceSum(_Multiply(var0, var1))->readMap<float>()[0];
    auto len0 = _Sqrt(_ReduceSum(_Square(var0)))->readMap<float>()[0];
    auto len1 = _Sqrt(_ReduceSum(_Square(var1)))->readMap<float>()[0];
    auto sim  = innerProd / (len0 * len1);
    return sim;
}

Embedding* Embedding::createEmbedding(const std::string& config_path, bool load) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Embedding* embedding = new Embedding(config);
    if (load) {
        embedding->load();
    }
    return embedding;
}

Embedding::Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {
}

int Embedding::dim() const {
    return mConfig->hidden_size();
}

void Embedding::load() {
    initRuntime();
    printf("load tokenizer\n");
    std::cout << mConfig->tokenizer_file() << std::endl;
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    printf("load tokenizer Done\n");
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    // 2. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    auto model_path            = mConfig->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    mModules.resize(1);
    mModules[0].reset(Module::load({"input_ids", "attention_mask", "position_ids"}, {"sentence_embeddings"},
                                   model_path.c_str(), mRuntimeManager, &module_config));
    MNN_PRINT("Done!\n");
}

VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len           = ids.size();
    auto inputs_ids          = embedding(ids);
    auto attention_mask      = gen_attention_mask(prompt_len);
    auto position_ids        = gen_position_ids(prompt_len);
    auto outputs             = mModules[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

VARP Embedding::txt_embedding(const std::string& txt) {
    return ids_embedding(tokenizer_encode(txt));
}

VARP Embedding::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<float>());
    auto ptr = attention_mask->writeMap<float>();
    if (mConfig->attention_mask() == "float") {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
            }
        }
    } else {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = 1.0;
            }
        }
    }
    return attention_mask;
}

VARP Embedding::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr          = position_ids->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = i;
    }
    return position_ids;
}

}
}
