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
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace MNN::Transformer;


std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    if (prompts.size() < 3) {
        MNN_ERROR("Need larger than 3 inputs\n");
        return 0;
    }
    auto& state = llm->getState();
    int initSize = 2;
    if (max_token_number <= 0) {
        max_token_number = 512;
    }
    MNN_PRINT("Prefill\n");
    std::vector<size_t> history;
    for (int i = 0; i < 3; i++) {
        const auto& prompt = prompts[i];
        llm->response(prompt, &std::cout, nullptr, 0);
        history.emplace_back(llm->getCurrentHistory());
    }
    MNN_PRINT("\n");
    
    MNN_PRINT("[LLM Test: Erase 1]\n");
    llm->eraseHistory(history[0], history[1]);
    llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
    while (!llm->stoped() && state.gen_seq_len_ < max_token_number) {
        llm->generate(1);
    }
    MNN_PRINT("\n[LLM Test End]\n");

    llm->eraseHistory(0, 0);
    history.clear();
    for (int i = 0; i < 3; i++) {
        const auto& prompt = prompts[i];
        llm->response(prompt, &std::cout, nullptr, 0);
        history.emplace_back(llm->getCurrentHistory());
    }
    MNN_PRINT("[LLM Test: Erase 2]\n");
    llm->eraseHistory(history[1], history[2]);
    llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
    while (!llm->stoped() && state.gen_seq_len_ < max_token_number) {
        llm->generate(1);
    }
    MNN_PRINT("\n[LLM Test End]\n");
    MNN_PRINT("[LLM Test For Init]\n");
    llm->reset();
    llm->eraseHistory(0, 0);
    llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
    while (!llm->stoped() && state.gen_seq_len_ < max_token_number) {
        llm->generate(1);
    }
    MNN_PRINT("\n[LLM Test End]\n");
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    return benchmark(llm, prompts, max_token_number);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << std::endl;
        return 0;
    }
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }
    int max_token_number = -1;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> max_token_number;
    }
    std::string prompt_file = argv[2];
    return eval(llm.get(), prompt_file, max_token_number);
}
