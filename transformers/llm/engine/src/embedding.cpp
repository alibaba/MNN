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

bool Embedding::load() {
    if (mConfig->config_.document.HasMember("load_disk_embedding_only") && mConfig->config_.document["load_disk_embedding_only"].GetBool()) {
        mDiskEmbedding.reset(new DiskEmbedding(mConfig));
        return true;
    }

    initRuntime();
    printf("load tokenizer\n");
    std::cout << mConfig->tokenizer_file() << std::endl;
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    printf("load tokenizer Done\n");
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    // 2. load model
    Module::Config module_config;
    if(mConfig->backend_type() == "npu") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    auto model_path            = mConfig->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    mModule.reset(Module::load({"input_ids", "attention_mask", "position_ids"}, {"sentence_embeddings"},
                                   model_path.c_str(), mRuntimeManager, &module_config));
    if (nullptr == mModule.get()) {
        return false;
    }
    MNN_PRINT("Done!\n");
    return true;
}

std::vector<Express::VARP> Embedding::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    return mModule->onForward({hiddenState, mask, inputPos});
}

VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len           = ids.size();
    auto inputs_ids          = embedding(ids);
    auto attention_mask      = gen_attention_mask(prompt_len);
    auto position_ids        = gen_position_ids(prompt_len);
    return forwardRaw(inputs_ids, attention_mask, position_ids)[0];
}

VARP Embedding::txt_embedding(const std::string& txt) {
    auto prompt = apply_chat_template(txt);
    return ids_embedding(tokenizer_encode(prompt));
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
