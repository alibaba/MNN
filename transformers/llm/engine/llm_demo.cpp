//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>

static void trace_prepare(Llm* llm) {
    MNN_PRINT("Prepare for resize opt Begin\n");
    std::vector<std::string> prompts = {
        "Hello",
    };
    llm->trace(true);
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    // llm->warmup();
    for (int i = 0; i < prompts.size(); i++) {
        std::ostringstream cacheOs;
        llm->response(prompts[i], &cacheOs);
        prompt_len += llm->prompt_len_;
        decode_len += llm->gen_seq_len_;
        prefill_time += llm->prefill_us_;
        decode_time += llm->decode_us_;
        llm->reset();
    }
    MNN_PRINT("Prepare for resize opt End\n");
    llm->trace(false);
}
static void benchmark(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        prompts.push_back(prompt);
    }
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    // llm->warmup();
    for (int i = 0; i < prompts.size(); i++) {
        llm->response(prompts[i]);
        prompt_len += llm->prompt_len_;
        decode_len += llm->gen_seq_len_;
        prefill_time += llm->prefill_us_;
        decode_time += llm->decode_us_;
        llm->reset();
    }
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", prompt_len);
    printf("decode tokens num  = %d\n", decode_len);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}


int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir <forwardtype> <preicionmemory> <prompt.txt>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    int forwardType = 0;
    if (argc >= 3) {
        std::istringstream os(argv[2]);
        os >> forwardType;
    }
    int memoryprecision = 10;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> memoryprecision;
    }
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir, "auto", forwardType, memoryprecision));
    {
        AUTOTIME;
        llm->load(model_dir);
    }
    {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    if (argc < 5) {
        llm->chat();
    }
    std::string prompt_file = argv[4];
    benchmark(llm.get(), prompt_file);
    return 0;
}
