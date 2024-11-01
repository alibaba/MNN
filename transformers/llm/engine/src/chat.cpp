//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "app/chat/chat.hpp"
#include "llm/llm.hpp"
#include "sampler/sampler.hpp"
#include "prompt/prompt.hpp"
#include "evaluation/evaluation.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

Chat::Chat(std::string config_path) {
    // 0. create Executor
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    // 1. create Llm
    MNN_PRINT("LLM config path is: %s\n", config_path.c_str());
    mLlm.reset(Llm::createLLM(config_path));
    mLlm->load();
    mLlm->set_config("{\"tmp_path\":\"tmp\"}");
    MNN_PRINT("Llm loaded!\n");
    // 2. create Sampler
    mSampler.reset(Sampler::createSampler(mLlm.get(), config_path));
    MNN_PRINT("Sampler initiated!\n");
    // 3. create PromptLib
    mPromptLib.reset(PromptLib::createPromptLib("BaseChat", config_path));
    MNN_PRINT("PromptLib initiated!\n");
}

void Chat::generate_init() {
    // init status
    mLlm->generate_init();
    if (!mLlm->reuse_kv()) {
        // no reuse, clear cached tokens. A brand-new prompt starts!
        mSampler->reset();
        mLlm->reset();
    }
}

std::string Chat::generate(const std::string& prompt, std::ostream* os, const char* end_with) {
    if (prompt.empty()) { return ""; }
    if (!end_with) { end_with = "\n"; }
    generate_init();
    // std::cout << "# prompt : " << prompt << std::endl;
    auto input_ids = mLlm->tokenizer(prompt);
    // printf("input_ids (%lu): ", input_ids.size()); for (auto id : input_ids) printf("%d, ", id); printf("\n");
    struct TimePerformance* time_perf = new struct TimePerformance;
    std::string out_str = mSampler->sample(input_ids, os, end_with, time_perf);
    delete time_perf;
    return out_str;
}

bool Chat::getPrompt(bool from_file, std::istream* is, std::string& user_str) {
    if (!from_file) std::cout << "\nQ: ";
    return (bool)std::getline(*is, user_str);
}

void Chat::getAnswer(std::string user_str, std::ostream* os, const char* end_with) {
    mPromptLib->appendUserPrompt(user_str);
    auto assistant_str = generate(mPromptLib->getLLMInput(), os, end_with);
    mPromptLib->appendLLMOutput(assistant_str);
}

void Chat::chat_init(std::string systemPrompt) {
    chat_reset();
    mPromptLib->appendSystemPrompt(systemPrompt);
}

void Chat::chat_reset() {
    mPromptLib->reset();
    mSampler->reset();
}

void Chat::chat(bool session_by_line, bool from_file, std::istream* is, std::ostream* os, const char* end_with, 
                    std::string exit_token, std::string reset_token) {
    // handle system prompt
    chat_init("You are a helpful assistant!\n");
    std::string user_str;
    while (getPrompt(from_file, is, user_str)) {
        // whether to end
        if (user_str == exit_token) {
            chat_reset();
            break;
        }
        // whether to reset
        if (session_by_line || user_str == reset_token) {
            chat_init("You are a helpful assistant!\n");
            if (!from_file) std::cout << "\nreset done." << std::endl;
            continue;
        }
        // get answer
        (*os) << "\nA: " << std::flush;
        getAnswer(user_str, os, end_with);
        (*os) << std::endl;
    }
    chat_reset();
}

} // Transformer
} // MNN
