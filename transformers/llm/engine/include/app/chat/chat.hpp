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
    void generate_init();
    std::string generate(const std::string& prompt, std::ostream* os = &std::cout, const char* end_with = "\n");
public:
    Chat(std::string config_path);
    void chat(std::istream* is = &std::cin, std::ostream* os = &std::cout, const char* end_with = "\n");
    ~Chat() {}
};


} // Transformer
} // MNN


#endif