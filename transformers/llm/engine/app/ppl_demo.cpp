//
//  ppl_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::Transformer;
static void trace_prepare(Llm* llm) {
    MNN_PRINT("Prepare for resize opt Begin\n");
    llm->trace(true);
    std::ostringstream cacheOs;
    llm->response("Hello", &cacheOs);
    MNN_PRINT("Prepare for resize opt End\n");
    llm->trace(false);
    llm->reset();
}

// parse json

static int ppl_eval(Llm* llm, std::string prompt_file, std::ofstream* perfOS) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    // ppl evaluation
    std::vector<float> ppls = llm->perplexity(prompt_file, perfOS);
    float mean_ppl = 0.f;
    for (int j = 0; j < ppls.size(); ++j) mean_ppl += ppls[j];
    mean_ppl /= ppls.size();
    std::cout << mean_ppl << std::endl;
    return 0;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " config.json ppl-prompt.txt [perf.txt]" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    {
        AUTOTIME;
        llm->load();
    }
    {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    std::string prompt_file = argv[2];
    std::unique_ptr<std::ofstream> perfOS(nullptr);
    if (argc == 4) { perfOS.reset(new std::ofstream(argv[3])); }
    return ppl_eval(llm.get(), prompt_file, perfOS.get());
}
