#pragma once
#include <string>
#include <vector>
#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

struct LLMBenchMarkInstance {
    std::string model;
    int n_prompt;
    int n_gen;
    std::vector<uint64_t> samples_ns;
};

struct LLMBenchMarkOptions {
    bool progress;
    int reps;
};

// Common benchmark function using llm_bench.cpp approach
void RunBenchmarkTest(MNN::Transformer::Llm* llm, const LLMBenchMarkInstance& instance);

} // namespace Transformer
} // namespace MNN
