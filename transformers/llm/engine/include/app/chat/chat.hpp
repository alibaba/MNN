//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#ifndef CHAT_Hpp
#define CHAT_Hpp


#include "llm/llm.hpp"
#include "sampler/sampler.hpp"
#include "prompt/prompt.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>


using namespace MNN::Express;
namespace MNN {
namespace Transformer {

class MNN_PUBLIC Chat {
protected:
    std::unique_ptr<Llm> mLlm;
    std::unique_ptr<Sampler> mSampler;
    std::unique_ptr<PromptLib> mPromptLib;
    bool getPrompt(bool from_file, std::istream* is, std::string& user_str);
    void generate_init();
    std::string generate(const std::string& prompt, std::ostream* os = &std::cout, const char* end_with = "\n");
public:
    Chat(std::string config_path);
    void getAnswer(std::string user_str, std::ostream* os, const char* end_with);
    void chat_init(std::string systemPrompt = "You are a helpful assistant!\n");
    void chat_reset();
    void chat(bool session_by_line = false, bool from_file = false, 
              std::istream* is = &std::cin, std::ostream* os = &std::cout, 
              const char* end_with = "\n", std::string exit_prompt = "/exit", std::string reset_token = "/reset");
    ~Chat() {}
};


} // Transformer
} // MNN


#endif