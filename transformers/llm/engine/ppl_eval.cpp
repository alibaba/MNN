//
//  llm_demo.cpp
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
#include <evaluation/perplexity.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/StateCacheManager.hpp>
using namespace MNN::Transformer;
static void trace_prepare(Llm* llm) {
    MNN_PRINT("Prepare for resize opt Begin\n");
    std::vector<std::string> prompts = {
        "Hello",
    };
    llm->trace(true);
    for (int i = 0; i < prompts.size(); i++) {
        std::ostringstream cacheOs;
        llm->response(prompts[i], &cacheOs);
    }
    MNN_PRINT("Prepare for resize opt End\n");
    llm->trace(false);
    llm->reset();
}

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

static int benchmark(Llm* llm, std::vector<std::string>& prompts) {
    for (int i = 0; i < prompts.size(); i++) {
        auto& prompt = prompts[i];
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        llm->response(prompt);
    }
    return 0;
}

static std::vector<std::string> entire(Llm* llm, std::string prompt_file) {
    // split by line
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    prompts.push_back("");
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        // concatenate.
        prompts.back() += prompt + "\n";
    }
    return prompts;
}

static std::vector<std::string> row(Llm* llm, std::string prompt_file) {
    // split by line
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    return prompts;
}

static std::vector<std::string> wiki(Llm* llm, std::string prompt_file) {
    // split wiki text into " = " first-level column.
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r' || prompt.back() == '\n') {
            prompt.pop_back();
        }
        if (prompt.size() < 4) continue;
        if ((prompts.size() == 0) \
             || (prompt.size() >= 4 \
                 && prompt.at(0) == ' ' \
                 && prompt.at(1) == '=' \
                 && prompt.at(2) == ' ' \
                 && prompt.at(3) != '=')) {
            // first-level column.
            prompts.push_back(prompt);
        } else {
            // concatenate.
            prompts.back() += "\n" + prompt;
        }
    }
    return prompts;
}

static int eval(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::vector<std::string> prompts = wiki(llm, prompt_file);
    std::cout << "prompt file loading ready!" << std::endl;
    if (prompts.empty()) {
        return 1;
    }
    // ceval
    PPLMeasurer measurer(llm, MNN::Express::ExecutorScope::Current()->getStateCacheManager(), prompts);
    std::vector<float> ppls = measurer.perplexity();
    float mean_ppl = 0.f;
    for (int j = 0; j < ppls.size(); ++j) mean_ppl += ppls[j];
    mean_ppl /= ppls.size();
    std::cout << mean_ppl << std::endl;
    return 0;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " config.json ppl-prompt.txt <prompt-type>" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    {
        AUTOTIME;
        llm->load();
    }
    if (true) {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    std::string prompt_file = argv[2];
    return eval(llm.get(), prompt_file);
}
